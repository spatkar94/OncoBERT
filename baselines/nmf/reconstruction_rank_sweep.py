"""Dense, validation-only NMF rank sweep with paired uncertainty and elbow analysis."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def atomic_json(value, path: Path):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sample_mse(x, w, h, batch_size=2048):
    result = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        residual = x[start:stop].astype(np.float64) - w[start:stop].astype(np.float64) @ h.astype(np.float64)
        result[start:stop] = np.mean(residual * residual, axis=1)
    return result


def fit_rank(train, validation, rank, seed, max_iter, tol):
    model = NMF(
        n_components=rank, init="nndsvda", solver="cd", beta_loss="frobenius",
        random_state=seed, max_iter=max_iter, tol=tol, shuffle=False,
    )
    begun = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        train_w = model.fit_transform(csr_matrix(train))
        validation_w = model.transform(csr_matrix(validation))
    train_errors = sample_mse(train, train_w, model.components_)
    validation_errors = sample_mse(validation, validation_w, model.components_)
    record = {
        "rank": rank,
        "train_mse": float(train_errors.mean()),
        "validation_mse": float(validation_errors.mean()),
        "iterations": int(model.n_iter_),
        "seconds": time.monotonic() - begun,
        "convergence_warning": any(issubclass(item.category, ConvergenceWarning) for item in caught),
        "warnings": [str(item.message) for item in caught],
    }
    return model, record, validation_errors


def segmented_elbow(ranks, errors):
    """Internal breakpoint minimizing two-line SSE in normalized rank/log-error space."""
    x = (ranks - ranks.min()) / (ranks.max() - ranks.min())
    y = np.log(errors)
    y = (y - y.min()) / (y.max() - y.min())
    candidates = []
    for split in range(2, len(ranks) - 2):
        left = np.polyfit(x[:split + 1], y[:split + 1], 1)
        right = np.polyfit(x[split:], y[split:], 1)
        sse = np.square(y[:split + 1] - np.polyval(left, x[:split + 1])).sum()
        sse += np.square(y[split:] - np.polyval(right, x[split:])).sum()
        candidates.append((float(sse), int(ranks[split])))
    return min(candidates)[1], candidates


def paired_bootstrap(error_matrix, bootstrap_replicates, seed):
    """Bootstrap adjacent relative MSE reductions using paired validation samples."""
    rng = np.random.default_rng(seed)
    n_samples, n_ranks = error_matrix.shape
    observed = 1 - error_matrix.mean(axis=0)[1:] / error_matrix.mean(axis=0)[:-1]
    draws = np.empty((bootstrap_replicates, n_ranks - 1), dtype=np.float64)
    for start in range(0, bootstrap_replicates, 50):
        stop = min(start + 50, bootstrap_replicates)
        indices = rng.integers(0, n_samples, size=(stop - start, n_samples))
        means = error_matrix[indices].mean(axis=1)
        draws[start:stop] = 1 - means[:, 1:] / means[:, :-1]
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    return observed, lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--practical-threshold", type=float, default=.01)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(args.profiles, allow_pickle=False))
    x = data["mutations"].astype(np.float32)
    assert x.shape[1] == 200 and np.isin(x, [0, 1]).all()
    train_indices = np.flatnonzero(data["split"] == "train")
    validation_indices = np.flatnonzero(data["split"] == "validation")
    test_indices = np.flatnonzero(data["split"] == "test")
    assert not set(data["patients"][train_indices]) & set(data["patients"][validation_indices])
    assert not set(data["patients"][train_indices]) & set(data["patients"][test_indices])
    ranks = np.array(sorted(set(args.ranks)), dtype=int)
    assert ranks.min() > 0 and ranks.max() <= x.shape[1]

    rows, error_columns = [], []
    for rank in ranks:
        destination = args.output / f"rank_{rank}"
        destination.mkdir(exist_ok=True)
        metrics_path = destination / "metrics.json"
        errors_path = destination / "validation_sample_mse.npy"
        model_path = destination / "nmf.joblib"
        if metrics_path.exists() and errors_path.exists() and model_path.exists():
            record = json.loads(metrics_path.read_text())
            validation_errors = np.load(errors_path)
            assert record["rank"] == rank and len(validation_errors) == len(validation_indices)
        else:
            model, record, validation_errors = fit_rank(
                x[train_indices], x[validation_indices], int(rank), args.seed, args.max_iter, args.tol
            )
            joblib.dump(model, model_path)
            np.save(errors_path, validation_errors)
            atomic_json(record, metrics_path)
        rows.append(record)
        error_columns.append(validation_errors)
        atomic_json(rows, args.output / "rank_metrics.json")
        print(json.dumps(record), flush=True)

    error_matrix = np.column_stack(error_columns)
    observed, lower, upper = paired_bootstrap(error_matrix, args.bootstrap_replicates, args.seed + 1)
    marginal = pd.DataFrame({
        "from_rank": ranks[:-1], "to_rank": ranks[1:],
        "relative_validation_mse_reduction": observed,
        "bootstrap_95ci_lower": lower, "bootstrap_95ci_upper": upper,
    })
    marginal.to_csv(args.output / "marginal_improvements.csv", index=False)

    errors = error_matrix.mean(axis=0)
    train_errors = np.array([row["train_mse"] for row in rows], dtype=float)
    pd.DataFrame({
        "rank": ranks,
        "train_reconstruction_mse": train_errors,
        "validation_reconstruction_mse": errors,
    }).to_csv(args.output / "reconstruction_error_by_k.csv", index=False)
    segmented_rank, segmented_candidates = segmented_elbow(ranks, errors)
    # Plateau requires every subsequent step to be confidently below the practical threshold.
    plateau_rank = None
    for index, rank in enumerate(ranks[:-1]):
        if np.all(upper[index:] < args.practical_threshold):
            plateau_rank = int(rank)
            break
    selection_status = "plateau_found" if plateau_rank is not None else "no_validation_plateau_in_grid"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(ranks, train_errors, "o-", color="#888888", markersize=4, label="Training MSE")
    axes[0].plot(ranks, errors, "o-", color="#3572b0", markersize=4, label="Validation MSE")
    axes[0].axvline(segmented_rank, color="#c44e52", linestyle="--", label=f"Segmented elbow: K={segmented_rank}")
    if plateau_rank is not None:
        axes[0].axvline(plateau_rank, color="#55a868", linestyle=":", label=f"<1% plateau: K={plateau_rank}")
    axes[0].set(xlabel="NMF rank K", ylabel="Validation reconstruction MSE", title="Validation reconstruction curve")
    axes[0].grid(alpha=.2); axes[0].legend(frameon=False)
    centers = ranks[1:]
    axes[1].errorbar(centers, 100 * observed, yerr=[100 * (observed-lower), 100 * (upper-observed)],
                     fmt="o-", capsize=2, color="#8172b3")
    axes[1].axhline(100 * args.practical_threshold, color="#c44e52", linestyle="--", label="1% practical threshold")
    axes[1].set(xlabel="Higher rank K", ylabel="MSE reduction from previous K (%)",
                title="Paired marginal validation improvement")
    axes[1].grid(alpha=.2); axes[1].legend(frameon=False)
    fig.suptitle("Shared-200 NMF rank sweep", fontsize=15)
    fig.text(.5, .01, f"Training-only fitting; paired bootstrap over {len(validation_indices):,} validation samples. "
             "Test split not transformed or evaluated during rank selection.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.18, top=.86, wspace=.25)
    fig.savefig(args.output / "nmf_rank_elbow.pdf", bbox_inches="tight")
    fig.savefig(args.output / "nmf_rank_elbow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "complete": True,
        "selection_status": selection_status,
        "plateau_rank": plateau_rank,
        "practical_threshold_relative_mse": args.practical_threshold,
        "plateau_rule": "Smallest K for which upper paired-bootstrap 95% CI is below threshold for every later adjacent step",
        "segmented_regression_elbow_rank": segmented_rank,
        "segmented_regression_role": "Descriptive elbow only; not substituted for absent plateau",
        "ranks": ranks.tolist(),
        "train_mse": train_errors.tolist(),
        "validation_mse": errors.tolist(),
        "bootstrap_replicates": args.bootstrap_replicates,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples_untouched": len(test_indices),
        "test_evaluated": False,
        "seed": args.seed,
        "segmented_candidates": segmented_candidates,
    }
    atomic_json(summary, args.output / "elbow_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

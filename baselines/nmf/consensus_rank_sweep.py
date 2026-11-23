"""Consensus-NMF stability and reconstruction analysis for one candidate rank K."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def atomic_json(value, path: Path) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruction_mse(x: np.ndarray, w: np.ndarray, h: np.ndarray,
                       batch_size: int = 2048) -> float:
    squared_error = 0.0
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        residual = (x[start:stop].astype(np.float64)
                    - w[start:stop].astype(np.float64) @ h.astype(np.float64))
        squared_error += float(np.square(residual).sum())
    return squared_error / float(x.size)


def consensus_subset(validation: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Choose the same nonzero validation profiles for every candidate K."""
    eligible = np.flatnonzero(validation.sum(axis=1) > 0)
    if len(eligible) < size:
        raise ValueError(f"Only {len(eligible)} nonzero validation profiles; requested {size}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=size, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--consensus-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    if not 2 <= args.rank <= 30:
        raise ValueError("This experiment requires K in [2, 30]")
    destination = args.output / f"k_{args.rank:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / "metrics.json"
    if final_path.exists():
        print(final_path.read_text(), flush=True)
        return

    data = dict(np.load(args.profiles, allow_pickle=False))
    x = data["mutations"].astype(np.float32)
    if x.shape[1] != 200 or not np.isin(x, [0, 1]).all():
        raise ValueError("Expected a binary shared-200 mutation matrix")
    train_indices = np.flatnonzero(data["split"] == "train")
    validation_indices = np.flatnonzero(data["split"] == "validation")
    test_indices = np.flatnonzero(data["split"] == "test")
    if set(data["patients"][train_indices]) & set(data["patients"][validation_indices]):
        raise ValueError("Patient leakage between training and validation")

    train = x[train_indices]
    validation = x[validation_indices]
    subset = consensus_subset(validation, args.consensus_samples, args.seed + 100_003)
    train_sparse = csr_matrix(train)
    validation_sparse = csr_matrix(validation)
    consensus = np.zeros((len(subset), len(subset)), dtype=np.float32)
    run_rows = []
    begun_all = time.monotonic()

    for restart in range(args.restarts):
        run_seed = args.seed + restart
        model = NMF(
            n_components=args.rank,
            init="random",
            solver="cd",
            beta_loss="frobenius",
            random_state=run_seed,
            max_iter=args.max_iter,
            tol=args.tol,
            shuffle=False,
        )
        begun = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            train_w = model.fit_transform(train_sparse)
            validation_w = model.transform(validation_sparse)
        labels = np.argmax(validation_w[subset], axis=1)
        consensus += (labels[:, None] == labels[None, :])
        row = {
            "restart": restart,
            "seed": run_seed,
            "train_mse": reconstruction_mse(train, train_w, model.components_),
            "validation_mse": reconstruction_mse(validation, validation_w, model.components_),
            "iterations": int(model.n_iter_),
            "seconds": time.monotonic() - begun,
            "occupied_components_in_consensus_subset": int(np.unique(labels).size),
            "convergence_warning": any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ),
            "warnings": [str(item.message) for item in caught],
        }
        run_rows.append(row)
        atomic_json(run_rows, destination / "restart_metrics.json")
        print(json.dumps({"rank": args.rank, **row}), flush=True)

    consensus /= float(args.restarts)
    np.fill_diagonal(consensus, 1.0)
    condensed_distance = squareform(1.0 - consensus, checks=False).astype(np.float64)
    hierarchy = linkage(condensed_distance, method="average")
    cophenetic_correlation, _ = cophenet(hierarchy, condensed_distance)
    if not np.isfinite(cophenetic_correlation):
        raise ValueError("Cophenetic correlation is not finite")

    train_mse = np.array([row["train_mse"] for row in run_rows])
    validation_mse = np.array([row["validation_mse"] for row in run_rows])
    result = {
        "complete": True,
        "rank": args.rank,
        "cophenetic_correlation": float(cophenetic_correlation),
        "train_reconstruction_mse_mean": float(train_mse.mean()),
        "train_reconstruction_mse_std": float(train_mse.std(ddof=1)),
        "validation_reconstruction_mse_mean": float(validation_mse.mean()),
        "validation_reconstruction_mse_std": float(validation_mse.std(ddof=1)),
        "restarts": args.restarts,
        "initialization": "independent random NMF initialization",
        "solver": "coordinate descent; Frobenius objective",
        "consensus_definition": "fraction of restarts assigning each pair to the same argmax-W component",
        "cophenetic_definition": "Pearson correlation between consensus distances and average-linkage cophenetic distances",
        "consensus_samples": len(subset),
        "consensus_sampling": "fixed random subset of nonzero validation profiles, identical for every K",
        "consensus_subset_validation_indices": subset.tolist(),
        "training_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples_untouched": len(test_indices),
        "test_evaluated": False,
        "seed": args.seed,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "seconds": time.monotonic() - begun_all,
        "input_sha256": sha256(args.profiles),
        "restart_metrics": run_rows,
    }
    atomic_json(result, final_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

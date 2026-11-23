"""Evaluate one NMF rank on the canonical uniform-15% masked-state task."""
from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def atomic_json(value, path: Path) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def write_rows(rows, path: Path) -> None:
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_masks(root: Path, split: str, expected_rows: np.ndarray) -> np.ndarray:
    saved = np.load(root / f"{split}_masks.npz", allow_pickle=False)
    np.testing.assert_array_equal(saved["sample_rows"], expected_rows)
    masks = saved["masks"].astype(bool)
    if masks.shape != (len(expected_rows), 200) or not np.all(masks.sum(axis=1) == 30):
        raise ValueError(f"Invalid fixed {split} masks")
    return masks


def infer_visible_coefficients(x: np.ndarray, masks: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Solve each held-out sample's NNLS problem using visible genes only."""
    coefficients = np.zeros((len(x), len(components)), dtype=np.float32)
    for row in range(len(x)):
        visible = ~masks[row]
        target = x[row, visible].astype(np.float64)
        if target.any():
            coefficients[row] = nnls(
                components[:, visible].T.astype(np.float64), target,
                maxiter=max(3 * len(components), 100),
            )[0]
    return coefficients


def masked_arrays(x: np.ndarray, masks: np.ndarray, scores: np.ndarray):
    rows, genes = np.nonzero(masks)
    return x[rows, genes].astype(np.uint8), scores[rows, genes], genes.astype(np.int16)


def metrics(labels: np.ndarray, raw_scores: np.ndarray, probabilities: np.ndarray,
            gene_ids: np.ndarray, genes: np.ndarray):
    per_gene = []
    for index, gene in enumerate(genes):
        chosen = gene_ids == index
        y = labels[chosen]
        positives = int(y.sum())
        count = int(len(y))
        ap = float(average_precision_score(y, raw_scores[chosen])) if 0 < positives < count else None
        per_gene.append({
            "gene": str(gene), "masked_entries": count,
            "mutated_entries": positives,
            "prevalence": float(y.mean()) if count else None,
            "average_precision": ap,
        })
    supported = [row for row in per_gene if row["average_precision"] is not None]
    predictions = probabilities >= .5
    positives = labels == 1
    negatives = ~positives
    result = {
        "masked_entries": int(len(labels)),
        "mutation_prevalence": float(labels.mean()),
        "masked_reconstruction_mse": float(np.square(labels - raw_scores).mean()),
        "calibrated_binary_cross_entropy": float(log_loss(labels, probabilities, labels=[0, 1])),
        "mutation_micro_auprc": float(average_precision_score(labels, raw_scores)),
        "mutation_macro_auprc": float(np.mean([row["average_precision"] for row in supported])),
        "genes_with_macro_auprc": len(supported),
        "mutation_auroc": float(roc_auc_score(labels, raw_scores)),
        "binary_top1": float((predictions == positives).mean()),
        "binary_balanced_accuracy": float(.5 * (
            predictions[positives].mean() + (~predictions[negatives]).mean()
        )),
    }
    return result, per_gene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    destination = args.output / f"k_{args.rank:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    final_path = destination / "masked_reconstruction_metrics.json"
    if final_path.exists():
        print(final_path.read_text(), flush=True)
        return

    data = dict(np.load(args.prepared / "profiles.npz", allow_pickle=False))
    x = data["mutations"].astype(np.float32)
    parts = {name: np.flatnonzero(data["split"] == name)
             for name in ("train", "validation", "test")}
    validation_masks = load_masks(args.prepared, "validation", parts["validation"])
    test_masks = load_masks(args.prepared, "test", parts["test"])

    model = NMF(
        n_components=args.rank, init="nndsvda", solver="cd", beta_loss="frobenius",
        random_state=args.seed, max_iter=args.max_iter, tol=args.tol, shuffle=False,
    )
    begun = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(csr_matrix(x[parts["train"]]))
    components = model.components_.astype(np.float32)
    joblib.dump(model, destination / "nmf.joblib")

    validation_w = infer_visible_coefficients(x[parts["validation"]], validation_masks, components)
    validation_scores = validation_w @ components
    validation_labels, validation_raw, validation_genes = masked_arrays(
        x[parts["validation"]], validation_masks, validation_scores
    )
    # Calibration is learned only from validation masked predictions. It is monotonic,
    # so it cannot alter AUPRC; it only makes binary cross-entropy interpretable.
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(validation_raw[:, None], validation_labels)
    validation_probabilities = calibrator.predict_proba(validation_raw[:, None])[:, 1]
    validation_metrics, validation_per_gene = metrics(
        validation_labels, validation_raw, validation_probabilities,
        validation_genes, data["genes"],
    )

    test_w = infer_visible_coefficients(x[parts["test"]], test_masks, components)
    test_scores = test_w @ components
    test_labels, test_raw, test_genes = masked_arrays(x[parts["test"]], test_masks, test_scores)
    test_probabilities = calibrator.predict_proba(test_raw[:, None])[:, 1]
    test_metrics, test_per_gene = metrics(
        test_labels, test_raw, test_probabilities, test_genes, data["genes"],
    )
    write_rows(validation_per_gene, destination / "validation_per_gene.csv")
    write_rows(test_per_gene, destination / "test_per_gene.csv")
    np.savez_compressed(
        destination / "test_masked_predictions.npz",
        labels=test_labels, raw_reconstruction_scores=test_raw.astype(np.float32),
        calibrated_mutation_probabilities=test_probabilities.astype(np.float32),
        gene_indices=test_genes,
    )

    result = {
        "complete": True,
        "rank": args.rank,
        "fit": {
            "training_samples": len(parts["train"]), "iterations": int(model.n_iter_),
            "seconds_total": time.monotonic() - begun,
            "convergence_warning": any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ),
        },
        "masking": "Canonical uniform 15%; exactly 30/200 genes; identical validation/test masks as neural models",
        "coefficient_inference": "Per-sample NNLS using only 170 visible genes; masked values are missing, never replaced by zero",
        "calibration": {
            "method": "unregularized one-dimensional logistic calibration",
            "fit_split": "validation masked entries only",
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            "note": "Monotonic calibration changes BCE/probabilities but not AUPRC ranking",
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "test_split_used_for_model_or_calibrator_fitting": False,
        "seed": args.seed,
    }
    atomic_json(result, final_path)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

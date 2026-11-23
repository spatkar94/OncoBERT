"""Compute paired patient-bootstrap CIs for rank-vs-set cancer retrieval.

This reproduces the exact held-out linear-probe protocol used to create the
point estimates in ``focused_three_state_rank_gat_set_benchmark_20261030``.
The same bootstrap patient indices are used for every model and for top-1 and
top-3 accuracy, so uncertainty is directly comparable across bars.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from benchmark_patient_cancer_types import valid_labels


ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "experiments/final_shared200_five_encoder_benchmark_20261029"
SET = ROOT / "experiments/candidate_three_state_logofunc_shared200_v20"
FOCUSED = ROOT / "experiments/focused_three_state_rank_gat_set_benchmark_20261030"
OUTPUT = ROOT / "experiments/rank_vs_oncobert_set_benchmark_shared200_v20"

MODELS = {
    "rank_esm2": FINAL / "canonical_assets/representations/rank_esm2/patients.h5ad",
    "rank_genept": FINAL / "canonical_assets/representations/rank_genept/patients.h5ad",
    "rank_string": FINAL / "canonical_assets/representations/rank_string/patients.h5ad",
    "three_state_logofunc": (
        SET / "evaluation/representation/three_state_logofunc/patients.h5ad"
    ),
}
COVARIATES = (
    FINAL / "canonical_assets/covariates/matched_train_test_covariates.csv.gz"
)
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42


def bootstrap_ci(correct: np.ndarray, seed: int) -> tuple[float, float]:
    """Percentile CI after resampling held-out patients with replacement."""
    correct = np.asarray(correct, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    # Work in batches to avoid materializing a large replicate-by-patient array.
    for start in range(0, BOOTSTRAP_REPLICATES, 100):
        stop = min(start + 100, BOOTSTRAP_REPLICATES)
        draw = rng.integers(0, len(correct), size=(stop - start, len(correct)))
        estimates[start:stop] = correct[draw].mean(axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(COVARIATES, index_col="patient_id")
    valid = valid_labels(raw) & raw[["assay", "institution", "log1p_tmb"]].notna().all(axis=1)
    frame = raw.loc[valid].copy()
    train = frame.split.eq("train").to_numpy()
    test = frame.split.eq("test").to_numpy()
    major = frame.loc[train, "cancer_type"].value_counts().head(12).index
    labels = frame.cancer_type.where(frame.cancer_type.isin(major), "Other").to_numpy()

    rows = []
    predictions = pd.DataFrame(index=frame.index[test])
    predictions["cancer_type"] = labels[test]

    for model, path in MODELS.items():
        obj = ad.read_h5ad(path)
        index = pd.Index(obj.obs_names).get_indexer(frame.index)
        if (index < 0).any():
            raise ValueError(f"{model}: missing {(index < 0).sum()} benchmark patients")
        matrix = np.asarray(obj.X)[index].astype(np.float64)
        scaler = StandardScaler().fit(matrix[train])
        classifier = RidgeClassifier(
            alpha=1, class_weight="balanced", solver="cholesky"
        ).fit(scaler.transform(matrix[train]), labels[train])
        scores = classifier.decision_function(scaler.transform(matrix[test]))
        truth = np.array([
            np.flatnonzero(classifier.classes_ == value)[0] for value in labels[test]
        ])
        ranking = np.argsort(scores, axis=1)[:, ::-1]
        top1 = ranking[:, 0] == truth
        top3 = (ranking[:, :3] == truth[:, None]).any(axis=1)
        top1_ci = bootstrap_ci(top1, BOOTSTRAP_SEED)
        top3_ci = bootstrap_ci(top3, BOOTSTRAP_SEED)
        predictions[f"{model}_top1_correct"] = top1.astype(np.uint8)
        predictions[f"{model}_top3_correct"] = top3.astype(np.uint8)
        rows.append({
            "model": model,
            "top1_accuracy": top1.mean(),
            "top1_ci_lower": top1_ci[0],
            "top1_ci_upper": top1_ci[1],
            "top3_accuracy": top3.mean(),
            "top3_ci_lower": top3_ci[0],
            "top3_ci_upper": top3_ci[1],
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "train_patients": int(train.sum()),
            "test_patients": int(test.sum()),
        })
        del obj, matrix, scores, ranking

    result = pd.DataFrame(rows)
    original = pd.read_csv(FOCUSED / "representation/cancer_type_top1_top3.csv")
    expected = original.set_index("model").loc[result.model]
    np.testing.assert_allclose(result.top1_accuracy, expected.top1_accuracy, atol=1e-12)
    np.testing.assert_allclose(result.top3_accuracy, expected.top3_accuracy, atol=1e-12)
    result.to_csv(OUTPUT / "cancer_type_retrieval_patient_bootstrap.csv", index=False)
    predictions.to_csv(OUTPUT / "cancer_type_retrieval_patient_predictions.csv.gz")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

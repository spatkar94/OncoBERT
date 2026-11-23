#!/usr/bin/env python3
"""Patient-clustered bootstrap CIs for embedding-dimension ablation metrics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get(
    "ONCOBERT_EXPERIMENT_ROOT",
    Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments",
))
EXPERIMENT = ROOT / "experiments/three_state_set_dim_ablation_shared200_v20"
RESULTS = EXPERIMENT / "results"
PROFILES = ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/profiles.npz"
MASKS = ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/test_masks.npz"
FUNCTIONAL_STATES = ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/functional_states.npy"
RUNS = {
    32: EXPERIMENT / "runs/dim32_seed42/test_predictions.npz",
    64: EXPERIMENT / "runs/dim64_seed42/test_predictions.npz",
    128: ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/runs/three_state_seed42/test_predictions.npz",
    256: EXPERIMENT / "runs/dim256_seed42/test_predictions.npz",
}
REPLICATES = 2_000
SEED = 42


def weighted_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    entry_patients: np.ndarray,
    bootstrap_counts: np.ndarray,
    chunk_size: int = 64,
) -> np.ndarray:
    """Exact weighted AP for clustered bootstrap replicates, including ties."""
    order = np.argsort(-scores, kind="stable")
    labels = labels[order].astype(np.float64)
    scores = scores[order]
    patients = entry_patients[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    result = np.empty(len(bootstrap_counts), dtype=np.float64)
    for first in range(0, len(bootstrap_counts), chunk_size):
        last = min(first + chunk_size, len(bootstrap_counts))
        weights = bootstrap_counts[first:last, patients].astype(np.float64)
        positive = np.add.reduceat(weights * labels, starts, axis=1)
        total = np.add.reduceat(weights, starts, axis=1)
        cumulative_positive = np.cumsum(positive, axis=1)
        cumulative_total = np.cumsum(total, axis=1)
        precision = np.divide(
            cumulative_positive, cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        positive_total = cumulative_positive[:, -1]
        result[first:last] = np.divide(
            np.sum(precision * positive, axis=1), positive_total,
            out=np.zeros_like(positive_total), where=positive_total > 0,
        )
    return result


def main() -> None:
    profiles = np.load(PROFILES, allow_pickle=False)
    masks = np.load(MASKS, allow_pickle=False)
    sample_rows = masks["sample_rows"].astype(np.int64)
    mask = masks["masks"].astype(bool)
    # ``profiles['states']`` is the earlier binary profile; the three-state
    # models were trained against the dedicated functional-state matrix.
    expected_labels = np.load(FUNCTIONAL_STATES, mmap_mode="r")[sample_rows][mask]
    sample_patients = profiles["patients"][sample_rows]
    unique_patients, sample_patient_index = np.unique(
        sample_patients, return_inverse=True
    )
    entry_patients = np.repeat(sample_patient_index, mask.sum(axis=1))

    # Every embedding dimension uses identical resampled patients.
    rng = np.random.default_rng(SEED)
    draws = rng.integers(
        0, len(unique_patients), size=(REPLICATES, len(unique_patients)),
        dtype=np.int32,
    )
    bootstrap_counts = np.zeros(
        (REPLICATES, len(unique_patients)), dtype=np.uint16
    )
    for index in range(REPLICATES):
        bootstrap_counts[index] = np.bincount(
            draws[index], minlength=len(unique_patients)
        ).astype(np.uint16)
    del draws

    rows = []
    replicate_tables = []
    for dimension, path in RUNS.items():
        saved = np.load(path, allow_pickle=False)
        labels = saved["labels"].astype(np.uint8)
        probabilities = saved["probabilities"].astype(np.float64)
        genes = saved["gene_indices"].astype(np.int16)
        np.testing.assert_array_equal(labels, expected_labels)
        if len(labels) != len(entry_patients):
            raise ValueError(f"Prediction/mask length mismatch for dim={dimension}")

        state_ap = np.column_stack([
            weighted_average_precision(
                labels == state, probabilities[:, state], entry_patients,
                bootstrap_counts,
            )
            for state in range(3)
        ])
        three_state = state_ap.mean(axis=1)

        binary_labels = labels > 0
        binary_scores = 1.0 - probabilities[:, 0]
        gene_ap = np.empty((REPLICATES, 200), dtype=np.float64)
        for gene in range(200):
            selected = genes == gene
            gene_ap[:, gene] = weighted_average_precision(
                binary_labels[selected], binary_scores[selected],
                entry_patients[selected], bootstrap_counts,
            )
        binary = gene_ap.mean(axis=1)
        replicate_tables.append(pd.DataFrame({
            "embedding_dim": dimension,
            "bootstrap_replicate": np.arange(REPLICATES, dtype=np.int32),
            "three_state_macro_auprc": three_state,
            "binary_mutation_macro_auprc": binary,
        }))
        rows.append({
            "embedding_dim": dimension,
            "patients": len(unique_patients),
            "bootstrap_replicates": REPLICATES,
            "bootstrap_seed": SEED,
            "three_state_patient_bootstrap_95ci_lower": float(np.quantile(three_state, .025)),
            "three_state_patient_bootstrap_95ci_upper": float(np.quantile(three_state, .975)),
            "binary_patient_bootstrap_95ci_lower": float(np.quantile(binary, .025)),
            "binary_patient_bootstrap_95ci_upper": float(np.quantile(binary, .975)),
        })
        print(rows[-1], flush=True)

    intervals = pd.DataFrame(rows).sort_values("embedding_dim")
    RESULTS.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(RESULTS / "patient_bootstrap_macro_auprc.csv", index=False)
    pd.concat(replicate_tables, ignore_index=True).to_csv(
        RESULTS / "patient_bootstrap_macro_auprc_replicates.csv", index=False
    )
    metrics_path = RESULTS / "dimension_ablation_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    new_columns = [
        "three_state_patient_bootstrap_95ci_lower",
        "three_state_patient_bootstrap_95ci_upper",
        "binary_patient_bootstrap_95ci_lower",
        "binary_patient_bootstrap_95ci_upper",
    ]
    metrics = metrics.drop(columns=[column for column in new_columns if column in metrics])
    metrics = metrics.merge(
        intervals[["embedding_dim"] + new_columns],
        on="embedding_dim", how="left", validate="one_to_one",
    )
    metrics.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Nested-CV linear Cox probes in MSK-CHORD NSCLC with ICI index therapy.

The endpoint is overall survival measured from the start of the first
post-sequencing index regimen.  All models use the same patients, folds,
outcomes, and inner hyperparameter-selection procedure.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv


ORIGINAL = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments"))
REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
SET_ROOT = (ORIGINAL / "experiments/candidate_three_state_logofunc_shared200_v20/"
            "evaluation/chord_os")
GAT_ROOT = (ORIGINAL / "experiments/candidate_mutation_projector_dim10_mean_shared200_v20/"
            "evaluation/chord_os")
NMF_ROOT = ORIGINAL / "experiments/chord_os_nmf13_three_state_gat_set_shared200_v20"
COHORT = (ORIGINAL / "experiments/chord_nsclc_ici_clusters_clean/"
          "ici_vs_nonici_analysis_cohort.tsv.gz")
OUTPUT = REVISION / "artifacts/chord_nsclc_ici_index_os_benchmark"

from chord_os_utils import evaluate  # noqa: E402
from survival_utils import nested_fit_raw  # noqa: E402


MODELS = {
    "nmf_k13": {
        "label": "NMF", "color": "#9c755f",
        "path": NMF_ROOT / "representations/nmf_k13.npy", "dimension": 13,
    },
    "mutation_projector_gat": {
        "label": "MutationProjector (GAT)", "color": "#59a14f",
        "path": GAT_ROOT / "representations/mutation_projector_gat10_three_state.npy",
        "dimension": 10,
    },
    "oncobert_set_transformer": {
        "label": "OncoBERT (Set Transformer)", "color": "#4e79a7",
        "path": SET_ROOT / "representations/functional_state.npy", "dimension": 128,
    },
    "exact_tmb": {
        "label": "Exact TMB", "color": "#e6ab02", "dimension": 1,
    },
}


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SET_ROOT / "prepared/patients.csv.gz", low_memory=False)
    cohort = pd.read_csv(COHORT, sep="\t", low_memory=False)
    cohort = cohort.loc[cohort.treatment.eq(1)].copy().reset_index(drop=True)
    if len(cohort) != cohort.PATIENT_ID.nunique():
        raise ValueError("ICI-index endpoint table must contain one row per patient")
    if not (cohort.duration.gt(0).all() and cohort.event.isin([0, 1, False, True]).all()):
        raise ValueError("Invalid treatment-start survival endpoint")

    frame_position = pd.Series(np.arange(len(frame)), index=frame.PATIENT_ID.astype(str))
    if not cohort.PATIENT_ID.astype(str).isin(frame_position.index).all():
        raise ValueError("ICI-index patients are not fully represented in the CHORD embedding frame")
    rows = cohort.PATIENT_ID.astype(str).map(frame_position).to_numpy(int)

    y = Surv.from_arrays(
        cohort.event.to_numpy(dtype=bool), cohort.duration.to_numpy(dtype=float))
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = np.full(len(cohort), -1, dtype=np.int8)
    for fold, (_, test) in enumerate(splitter.split(np.arange(len(cohort)), y["event"])):
        folds[test] = fold
    fold_table = cohort[["PATIENT_ID", "SAMPLE_ID", "duration", "event"]].copy()
    fold_table["fold"] = folds
    fold_table.to_csv(OUTPUT / "fold_assignments.csv", index=False)

    features = {}
    for name, specification in MODELS.items():
        if name == "exact_tmb":
            values = np.log1p(frame.TMB_NONSYNONYMOUS.to_numpy(float))[:, None]
        else:
            values = np.load(specification["path"], mmap_mode="r")
        expected = (len(frame), specification["dimension"])
        if values.shape != expected:
            raise ValueError(f"{name} shape {values.shape}, expected {expected}")
        local = np.asarray(values[rows], dtype=np.float64)
        if not np.isfinite(local).all():
            raise ValueError(f"Non-finite values in {name}")
        features[name] = local

    records, predictions = [], []
    for name, specification in MODELS.items():
        for fold in range(5):
            test = np.flatnonzero(folds == fold)
            train = np.flatnonzero(folds != fold)
            risk, fit = nested_fit_raw(features[name], y, train, test, fold)
            metrics = evaluate(y[train], y[test], risk)
            records.append({
                "model": name, "model_label": specification["label"], "fold": fold,
                "train_patients": len(train), "test_patients": len(test),
                "train_deaths": int(y["event"][train].sum()),
                "test_deaths": int(y["event"][test].sum()),
                **{key: value for key, value in metrics.items()
                   if not isinstance(value, dict)},
                "selected_alpha": fit["selected_alpha"],
            })
            predictions.append(pd.DataFrame({
                "PATIENT_ID": cohort.PATIENT_ID.iloc[test].to_numpy(),
                "SAMPLE_ID": cohort.SAMPLE_ID.iloc[test].to_numpy(),
                "model": name, "fold": fold, "event": y["event"][test],
                "months_from_treatment_start": y["time"][test], "risk": risk,
            }))
            print(name, fold, f"C={metrics['harrell_c']:.4f}",
                  f"AUC12={metrics['auc_12']}", flush=True)

    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT / "fold_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT / "held_out_predictions.csv.gz", index=False)
    summaries = []
    for name, specification in MODELS.items():
        local = metrics.loc[metrics.model.eq(name)]
        row = {"model": name, "model_label": specification["label"]}
        for metric in ["harrell_c", "ipcw_c24", "auc_6", "auc_12", "auc_24"]:
            row[f"{metric}_mean"] = float(local[metric].mean())
            row[f"{metric}_sd"] = float(local[metric].std(ddof=1))
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "summary_metrics.csv", index=False)

    figure = plt.figure(figsize=(14.5, 5.8))
    grid = figure.add_gridspec(1, 2, width_ratios=[.95, 1.3], wspace=.28)
    c_axis, auc_axis = figure.add_subplot(grid[0]), figure.add_subplot(grid[1])
    positions = np.arange(len(summary))
    colors = [MODELS[name]["color"] for name in summary.model]
    bars = c_axis.bar(positions, summary.harrell_c_mean,
                      yerr=summary.harrell_c_sd, capsize=4, color=colors, width=.72)
    c_axis.bar_label(bars, labels=[f"{value:.3f}" for value in summary.harrell_c_mean],
                     padding=7, fontsize=10, fontweight="bold")
    c_axis.axhline(.5, color="#555555", ls="--", lw=1)
    c_axis.set_xticks(positions,
                      [value.replace(" (", "\n(") for value in summary.model_label],
                      fontsize=9)
    c_axis.set_ylabel("Held-out Harrell C-index")
    c_axis.set_ylim(.42, max(.72, float((summary.harrell_c_mean + summary.harrell_c_sd).max()+.06)))
    c_axis.set_title("A  Overall-survival discrimination", loc="left", weight="bold")
    c_axis.grid(axis="y", alpha=.16)
    c_axis.spines[["top", "right"]].set_visible(False)

    horizons = np.array([6, 12, 24])
    for row in summary.itertuples():
        auc_axis.errorbar(
            horizons, [row.auc_6_mean, row.auc_12_mean, row.auc_24_mean],
            yerr=[row.auc_6_sd, row.auc_12_sd, row.auc_24_sd], marker="o",
            capsize=4, lw=2, ms=6, color=MODELS[row.model]["color"],
            label=MODELS[row.model]["label"])
    auc_axis.axhline(.5, color="#555555", ls="--", lw=1)
    auc_axis.set_xticks(horizons)
    auc_axis.set_xlabel("Months from treatment start")
    auc_axis.set_ylabel("Held-out cumulative/dynamic AUROC")
    auc_axis.set_ylim(.42, .78)
    auc_axis.set_title("B  Time-dependent discrimination", loc="left", weight="bold")
    auc_axis.grid(alpha=.16)
    auc_axis.spines[["top", "right"]].set_visible(False)
    auc_axis.legend(frameon=False, fontsize=9, loc="lower right")

    figure.suptitle("MSK-CHORD NSCLC with ICI as index treatment", fontsize=17, y=.98)
    figure.text(
        .5, .012,
        f"n={len(cohort):,}; deaths={int(y['event'].sum()):,}. Mean ± SD across five shared held-out folds; "
        "three-fold inner tuning of raw-value elastic-net Cox models. Patient embeddings are arithmetic means "
        "across eligible day-zero samples; Exact TMB uses log(1 + TMB).",
        ha="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.07, right=.985, top=.86, bottom=.22)
    for suffix in ("pdf", "png", "svg"):
        figure.savefig(OUTPUT / f"nsclc_ici_index_os_benchmark.{suffix}", dpi=300,
                       bbox_inches="tight")
    plt.close(figure)

    protocol = {
        "complete": True,
        "cohort": "MSK-CHORD NSCLC with ICI-containing first post-sequencing index regimen",
        "patients": int(len(cohort)), "deaths": int(y["event"].sum()),
        "time_origin": "index treatment start", "models": list(MODELS),
        "outer_folds": 5, "inner_folds": 3, "seed": 42,
        "model": "elastic-net Cox linear probe; l1_ratio=0.1",
        "preprocessing": "raw stored representations; no scaling or normalization",
        "patient_representation": "arithmetic mean of all eligible day-zero sample embeddings per patient; post-baseline specimens excluded",
        "uncertainty": "mean and sample SD across five held-out folds",
        "auc_horizons_months": [6, 12, 24],
    }
    atomic_json(protocol, OUTPUT / "protocol.json")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

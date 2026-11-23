#!/usr/bin/env python3
"""Sensitivity of the Cluster-2 × ICI interaction to Leiden settings.

The reference phenotype is the saved GENIE resolution-0.5 Cluster 2.  For
each cached Leiden solution, we identify the cluster with maximum patient
Jaccard overlap with that phenotype, transfer that binary membership to
CHORD with a class-balanced linear probe, and refit the exact saved weighted
NSCLC ICI-versus-non-ICI Cox interaction model.
"""

from __future__ import annotations

import json
from pathlib import Path
import os

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (balanced_accuracy_score, jaccard_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments"))
ROBUST = ROOT / "experiments/genie_leiden_robustness"
GENIE = (ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/"
         "evaluation/representation/three_state_logofunc/patients.h5ad")
CHORD = (ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/"
         "evaluation/chord_os")
COHORT = ROOT / "experiments/chord_nsclc_ici_clusters_clean/ici_vs_nonici_analysis_cohort.tsv.gz"
OUTPUT = ROOT / "experiments/cluster2_ici_leiden_sensitivity_20261107"
REFERENCE_CLUSTER = "2"
REFERENCE_RESOLUTION = 0.5
REFERENCE_SEED = 42


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def interaction(frame: pd.DataFrame, membership: np.ndarray) -> dict:
    local = frame[["duration", "event", "treatment", "overlap_weight"]].copy()
    local["in_cluster"] = np.asarray(membership, dtype=int)
    local["treatment_x_cluster"] = local.treatment * local.in_cluster
    support = (local.groupby(["treatment", "in_cluster"]).agg(
        patients=("event", "size"), deaths=("event", "sum"))
        .reindex(pd.MultiIndex.from_product([[0, 1], [0, 1]],
                                            names=["treatment", "in_cluster"]),
                 fill_value=0))
    result = {
        "ici_cluster_patients": int(support.loc[(1, 1), "patients"]),
        "ici_cluster_deaths": int(support.loc[(1, 1), "deaths"]),
        "nonici_cluster_patients": int(support.loc[(0, 1), "patients"]),
        "nonici_cluster_deaths": int(support.loc[(0, 1), "deaths"]),
        "evaluable": bool((support.patients >= 15).all() and (support.deaths >= 5).all()),
    }
    if not result["evaluable"]:
        return result
    model = CoxPHFitter(penalizer=.001).fit(
        local, duration_col="duration", event_col="event",
        weights_col="overlap_weight", robust=True)
    item = model.summary.loc["treatment_x_cluster"]
    result.update({
        "interaction_ratio_of_hazard_ratios": float(item["exp(coef)"]),
        "interaction_ci_low": float(item["exp(coef) lower 95%"]),
        "interaction_ci_high": float(item["exp(coef) upper 95%"]),
        "interaction_p_value": float(item["p"]),
        "same_favorable_ici_direction": bool(item["exp(coef)"] < 1),
    })
    return result


def calibrated_binary_transfer(gx: np.ndarray, cx: np.ndarray,
                               target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit a binary Ridge probe and calibrate its cutoff on held-out GENIE.

    A zero cutoff is inappropriate here because class balancing intentionally
    changes the implied prevalence.  Maximizing held-out Jaccard restores a
    membership-specific cutoff while leaving the linear score untouched.
    """
    target = np.asarray(target, dtype=int)
    split = StratifiedShuffleSplit(n_splits=1, test_size=.20, random_state=42)
    train, test = next(split.split(gx, target))
    probe = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(
        gx[train], target[train])
    score = np.asarray(probe.decision_function(gx[test]), float)
    precision, recall, thresholds = precision_recall_curve(target[test], score)
    # precision/recall have one extra terminal element without a threshold.
    f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                   out=np.zeros_like(thresholds),
                   where=(precision[:-1]+recall[:-1]) > 0)
    best = int(np.nanargmax(f1))
    cutoff = float(thresholds[best])
    heldout = score >= cutoff
    probe.fit(gx, target)
    membership = np.asarray(probe.decision_function(cx), float) >= cutoff
    metrics = {
        "transfer_cutoff": cutoff,
        "heldout_balanced_accuracy": float(balanced_accuracy_score(target[test], heldout)),
        "heldout_jaccard": float(jaccard_score(target[test], heldout)),
        "heldout_precision": float(precision_score(target[test], heldout)),
        "heldout_recall": float(recall_score(target[test], heldout)),
    }
    return membership, metrics


def plot_results(results: pd.DataFrame, validation: dict) -> None:
    use = results[results.evaluable].copy()
    resolutions = sorted(use.resolution.unique())
    colors = {0.10: "#9ecae1", 0.25: "#6baed6", 0.50: "#e15759",
              0.75: "#3182bd", 1.00: "#08519c"}
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8))

    ax = axes[0]
    for resolution in resolutions:
        local = use[use.resolution.eq(resolution)]
        jitter = np.linspace(-.025, .025, len(local))
        ax.scatter(np.full(len(local), resolution) + jitter, local.best_jaccard,
                   s=28, alpha=.72, color=colors[resolution])
        ax.plot(resolution, local.best_jaccard.median(), marker="_", markersize=18,
                markeredgewidth=2.4, color="black")
    ax.set(xlabel="Leiden resolution", ylabel="Best-match Jaccard with reference Cluster 2",
           title="A  Phenotype recovery", ylim=(0, 1.02))

    ax = axes[1]
    ax.axhline(1, color=".35", linestyle="--", linewidth=1)
    for resolution in resolutions:
        local = use[use.resolution.eq(resolution)]
        jitter = np.linspace(-.025, .025, len(local))
        ax.scatter(np.full(len(local), resolution) + jitter,
                   local.interaction_ratio_of_hazard_ratios,
                   s=30, alpha=.72, color=colors[resolution])
        ax.plot(resolution, local.interaction_ratio_of_hazard_ratios.median(),
                marker="_", markersize=18, markeredgewidth=2.4, color="black")
    ax.set_yscale("log")
    ax.set(xlabel="Leiden resolution", ylabel="Treatment × cluster ratio of hazard ratios",
           title="B  ICI-interaction sensitivity")

    ax = axes[2]
    summary = use.groupby("resolution").agg(
        same_direction=("same_favorable_ici_direction", "mean"),
        nominal_significant=("interaction_p_value", lambda x: np.mean(x < .05)),
        median_jaccard=("best_jaccard", "median")).reindex(resolutions)
    positions = np.arange(len(summary))
    ax.bar(positions-.18, 100*summary.same_direction, width=.36,
           color="#4e79a7", label="Same favorable-ICI direction")
    ax.bar(positions+.18, 100*summary.nominal_significant, width=.36,
           color="#f28e2b", label="Nominal p<0.05")
    ax.set_xticks(positions, [f"{x:g}" for x in summary.index])
    ax.set(xlabel="Leiden resolution", ylabel="Sensitivity runs (%)",
           title="C  Direction and significance", ylim=(0, 105))
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", alpha=.18)
    fig.text(.5, .015,
             "Twenty random seeds per resolution. Alternative clusters are matched to the saved r=0.5 Cluster 2 by GENIE patient overlap; "
             "membership is transferred to the fixed CHORD cohort with a class-balanced linear probe. Black ticks show medians.",
             ha="center", fontsize=8.8, color=".28")
    fig.tight_layout(rect=(0, .055, 1, 1))
    save(fig, "cluster2_ici_leiden_sensitivity")

    # A compact effect-size forest with the full seed range (a sensitivity
    # interval, not a sampling confidence interval).
    summary_rows = []
    for resolution, local in use.groupby("resolution"):
        summary_rows.append({
            "resolution": resolution,
            "median": local.interaction_ratio_of_hazard_ratios.median(),
            "low": local.interaction_ratio_of_hazard_ratios.min(),
            "high": local.interaction_ratio_of_hazard_ratios.max(),
            "same_direction": local.same_favorable_ici_direction.mean(),
            "p_lt_0p05": (local.interaction_p_value < .05).mean(),
        })
    summary_frame = pd.DataFrame(summary_rows).sort_values("resolution")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(summary_frame))
    for i, row in enumerate(summary_frame.itertuples()):
        ax.errorbar(row.median, i,
                    xerr=[[row.median-row.low], [row.high-row.median]], fmt="o",
                    color=colors[row.resolution], ecolor=colors[row.resolution],
                    capsize=3, linewidth=1.8)
        ax.text(row.high*1.04, i,
                f"direction {100*row.same_direction:.0f}%; p<0.05 {100*row.p_lt_0p05:.0f}%",
                va="center", fontsize=9)
    ax.axvline(1, color=".3", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y, [f"Resolution {x:g}" for x in summary_frame.resolution])
    ax.invert_yaxis()
    ax.set(xlabel="Treatment × cluster ratio of hazard ratios",
           title="Cluster-2 ICI interaction across Leiden specifications")
    ax.grid(axis="x", alpha=.2)
    fig.text(.5, .015,
             "Points: median across 20 seeds; whiskers: full seed range (sensitivity range, not a 95% CI). Values below 1 favor ICI relatively more within the cluster.",
             ha="center", fontsize=8.7, color=".3")
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, "cluster2_ici_effect_sensitivity_forest")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    recovery = pd.read_csv(ROBUST / "cluster2_recovery_across_resolutions.tsv", sep="\t")
    saved = np.load(ROBUST / "leiden_solutions.npz")
    data = ad.read_h5ad(GENIE, backed="r")
    genie_x = np.asarray(data.X, np.float32)
    reference = data.obs["leiden_r0p5"].astype(str).to_numpy().eq(REFERENCE_CLUSTER) if hasattr(data.obs["leiden_r0p5"].astype(str).to_numpy(), 'eq') else (data.obs["leiden_r0p5"].astype(str).to_numpy() == REFERENCE_CLUSTER)
    data.file.close()
    reference = np.asarray(reference, bool)
    chord_x_all = np.load(CHORD / "representations/functional_state.npy", mmap_mode="r")
    frame = pd.read_csv(COHORT, sep="\t", low_memory=False)
    chord_x = np.asarray(chord_x_all[frame.chord_row.to_numpy(int)], np.float32)

    scaler = StandardScaler().fit(genie_x)
    gx = scaler.transform(genie_x)
    cx = scaler.transform(chord_x)

    # Validate the binary-transfer formulation against the original transferred
    # multiclass membership used in the published interaction analysis.
    reference_chord, reference_metrics = calibrated_binary_transfer(
        gx, cx, reference.astype(int))
    original_chord = frame.assigned_cluster.astype(str).eq(REFERENCE_CLUSTER).to_numpy()
    validation = {
        **reference_metrics,
        "chord_binary_vs_original_multiclass_jaccard": float(jaccard_score(original_chord, reference_chord)),
        "chord_binary_vs_original_multiclass_precision": float(precision_score(original_chord, reference_chord)),
        "chord_binary_vs_original_multiclass_recall": float(recall_score(original_chord, reference_chord)),
        "original_cluster2_chord_patients": int(original_chord.sum()),
        "binary_transfer_cluster2_chord_patients": int(reference_chord.sum()),
    }

    rows = []
    for item in recovery.itertuples(index=False):
        resolution = float(item.resolution)
        seed = int(item.seed)
        key = f"r{resolution:g}_s{seed}"
        labels = np.asarray(saved[key], int)
        target = labels == int(item.best_candidate_cluster_code)
        membership, transfer_metrics = calibrated_binary_transfer(
            gx, cx, target.astype(int))
        result = interaction(frame, membership)
        rows.append({
            "resolution": resolution, "seed": seed,
            "candidate_cluster_code": int(item.best_candidate_cluster_code),
            "reference_cluster_size": int(item.reference_size),
            "candidate_cluster_size": int(item.candidate_size),
            "intersection": int(item.intersection),
            "best_jaccard": float(item.best_jaccard),
            "reference_recall": float(item.reference_recall),
            "candidate_precision": float(item.candidate_precision),
            "transferred_chord_cluster_patients": int(membership.sum()),
            **transfer_metrics,
            **result,
        })
        print(f"r={resolution:g} seed={seed:2d} J={item.best_jaccard:.3f} "
              f"CHORD n={membership.sum():4d} RHR={result.get('interaction_ratio_of_hazard_ratios', np.nan):.3f} "
              f"p={result.get('interaction_p_value', np.nan):.3g}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "cluster2_ici_leiden_sensitivity.tsv", sep="\t", index=False)
    aggregate = results[results.evaluable].groupby("resolution").agg(
        runs=("seed", "size"),
        jaccard_median=("best_jaccard", "median"),
        jaccard_min=("best_jaccard", "min"),
        jaccard_max=("best_jaccard", "max"),
        interaction_median=("interaction_ratio_of_hazard_ratios", "median"),
        interaction_min=("interaction_ratio_of_hazard_ratios", "min"),
        interaction_max=("interaction_ratio_of_hazard_ratios", "max"),
        same_direction_fraction=("same_favorable_ici_direction", "mean"),
        nominal_p_lt_0p05_fraction=("interaction_p_value", lambda x: np.mean(x < .05)),
    ).reset_index()
    aggregate.to_csv(OUTPUT / "cluster2_ici_leiden_sensitivity_summary.tsv", sep="\t", index=False)
    (OUTPUT / "binary_transfer_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = {
        "complete": True,
        "reference_cluster": REFERENCE_CLUSTER,
        "reference_resolution": REFERENCE_RESOLUTION,
        "reference_seed": REFERENCE_SEED,
        "resolutions": sorted(results.resolution.unique().tolist()),
        "seeds_per_resolution": int(results.groupby("resolution").seed.nunique().min()),
        "runs": int(len(results)),
        "cluster_correspondence": "maximum patient-level Jaccard overlap in GENIE",
        "transfer": "StandardScaler + class-balanced binary RidgeClassifier(alpha=1)",
        "survival_model": "saved overlap-weighted NSCLC ICI-versus-non-ICI Cox treatment-by-cluster interaction",
        "validation": validation,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_results(results, validation)
    print("\nBinary transfer validation:\n", json.dumps(validation, indent=2))
    print("\nSensitivity summary:\n", aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

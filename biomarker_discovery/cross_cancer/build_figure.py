#!/usr/bin/env python3
"""Consolidate external cluster-2 OS and harmonized ICI response findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.patches import Patch
from scipy.stats import fisher_exact


BASE = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments"))
IMPACT = BASE / "experiments/cluster2_msk_impact_ici_os"
RESPONSE = Path(os.environ.get("ONCOBERT_ICI_RESPONSE_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "response_benchmark"))
CHORD_ASSIGNMENTS = (BASE / "experiments/genie_leiden_transfer_chord_ici_os_r0p5/"
                     "chord_ici_assigned_clusters.tsv.gz")
OTHER_RESPONSE_ASSIGNMENTS = (BASE / "experiments/cluster2_other_immunotherapy_response/"
                              "patient_cluster_assignments.tsv.gz")
NSCLC_RESPONSE_ASSIGNMENTS = (BASE / "experiments/cluster2_external_nsclc_ici_validation/"
                              "external_nsclc_cluster_assignments.tsv.gz")
NEW_RESPONSE_ASSIGNMENTS = (
    BASE / "experiments/cluster2_new_cbioportal_ici_validation_20261105_pog_dcb/"
    "patient_cluster_assignments_and_response.tsv.gz"
)
OUTPUT = BASE / "experiments/cluster2_impact_and_ici_response_consolidated_20261103"

BLUE = "#4e79a7"
RED = "#d62728"
GRAY = "#9c9c9c"

COHORT_NAMES = {
    "blca_iatlas_imvigor210_2017": "IMvigor210",
    "brca_iatlas_anders_2022": "Anders 2022",
    "ccrcc_dfci_2019": "DFCI 2019",
    "gbm_columbia_2019": "Columbia 2019",
    "mel_iatlas_liu_2019": "Liu 2019",
    "mel_ucla_2016": "UCLA 2016",
    "nsclc_mskcc_2015": "MSK 2015",
    "nsclc_mskcc_2018": "MSK 2018",
    "paad_iatlas_prince_2022": "PRINCE",
    "rcc_iatlas_immotion150_2018": "IMmotion150",
    "skcm_dfci_2015": "DFCI 2015",
    "skcm_mskcc_2014": "MSK 2014",
    "blca_bcan_hcrn_2022": "BCAN/HCRN 2022",
    "mel_iatlas_riaz_nivolumab_2017": "Riaz 2017",
}


def load_impact(exclude_all_nsclc: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(IMPACT / "patient_cluster2_survival_table.tsv.gz", sep="\t")
    chord = pd.read_csv(CHORD_ASSIGNMENTS, sep="\t", dtype=str)
    chord_nsclc_ids = set(chord.loc[
        chord.CANCER_TYPE.eq("Non-Small Cell Lung Cancer"), "PATIENT_ID"
    ].dropna())
    frame = frame.loc[~frame.patient_id.astype(str).isin(chord_nsclc_ids)].copy()
    if exclude_all_nsclc:
        frame = frame.loc[
            ~frame.CANCER_TYPE.eq("Non-Small Cell Lung Cancer")
        ].copy()
    frame["duration"] = frame.duration.clip(lower=1e-3)
    return frame


def panel_a(axis, frame: pd.DataFrame, subtitle: str) -> dict:
    for value, label, color in [
        (1, "Cluster 2", RED),
        (0, "Remaining clusters", GRAY),
    ]:
        local = frame.loc[frame.cluster2.eq(value)]
        KaplanMeierFitter(
            label=f"{label} (n={len(local):,}, events={int(local.event.sum()):,})"
        ).fit(local.duration, local.event).plot_survival_function(
            ax=axis, ci_show=True, ci_alpha=.14, color=color, linewidth=2.2,
            show_censors=False,
        )

    cluster2 = frame.loc[frame.cluster2.eq(1)]
    remaining = frame.loc[frame.cluster2.eq(0)]
    logrank = logrank_test(
        cluster2.duration, remaining.duration,
        event_observed_A=cluster2.event,
        event_observed_B=remaining.event,
    )
    model = CoxPHFitter(penalizer=.001).fit(
        frame[["duration", "event", "cluster2"]],
        duration_col="duration", event_col="event", robust=True,
    )
    row = model.summary.loc["cluster2"]
    stats = {
        "patients": int(len(frame)),
        "events": int(frame.event.sum()),
        "cluster2_patients": int(frame.cluster2.sum()),
        "hazard_ratio": float(row["exp(coef)"]),
        "ci_low": float(row["exp(coef) lower 95%"]),
        "ci_high": float(row["exp(coef) upper 95%"]),
        "cox_p": float(row.p),
        "logrank_p": float(logrank.p_value),
    }
    axis.text(
        .97, .06,
        f"Unadjusted HR {stats['hazard_ratio']:.2f} "
        f"({stats['ci_low']:.2f}–{stats['ci_high']:.2f})\n"
        f"log-rank p={stats['logrank_p']:.3g}",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=12,
        bbox={"facecolor": "white", "edgecolor": ".82", "alpha": .92},
    )
    title = (
        "A  MSK Targeted sequencing immunotherapy\n(other cancer types)"
        if subtitle == "All NSCLC patients excluded"
        else "A  MSK-IMPACT pan-cancer ICI cohort\n" + subtitle
    )
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=10)
    axis.set_xlabel("Months from ICI Treatment Start")
    axis.set_ylabel("Overall-survival probability")
    axis.set_ylim(0, 1.03)
    axis.grid(axis="y", alpha=.18)
    axis.legend(frameon=False, loc="upper right", fontsize=11)
    return stats


def panel_b(axis, exclude_all_nsclc: bool, subtitle: str) -> pd.DataFrame:
    results = pd.read_csv(
        IMPACT / "per_cancer_cluster2_cox_excluding_chord_nsclc.tsv", sep="\t"
    )
    if exclude_all_nsclc:
        results = results.loc[
            ~results.cancer_type.eq("Non-Small Cell Lung Cancer")
        ].copy()
    shown = results.sort_values("hazard_ratio", ascending=False).reset_index(drop=True)
    shown["display"] = shown.cancer_type.replace({
        "Non-Small Cell Lung Cancer": "NSCLC",
        "Cancer of Unknown Primary": "Unknown primary",
    })
    shown["display_with_n"] = [
        rf"{label}  ($\mathbf{{n={int(count):,}}}$)"
        for label, count in zip(shown.display, shown.patients)
    ]
    y = np.arange(len(shown))
    axis.errorbar(
        shown.hazard_ratio, y,
        xerr=np.vstack([
            shown.hazard_ratio - shown.ci_low,
            shown.ci_high - shown.hazard_ratio,
        ]),
        fmt="o", markersize=7, color=BLUE, ecolor=BLUE,
        elinewidth=1.8, capsize=3,
    )
    axis.axvline(1, color="black", linestyle="--", linewidth=1.2)
    for ypos, row in enumerate(shown.itertuples()):
        axis.text(row.ci_high * 1.07, ypos, f"p={row.p_value:.3g}",
                  va="center", fontsize=11, clip_on=False)
    axis.set_xscale("log")
    axis.set_yticks(y, shown.display_with_n)
    axis.set_xlabel("Hazard Ratio (95% CI)")
    title = (
        "B  MSK Targeted sequencing immunotherapy\n"
        "cancer-type specific associations"
        if exclude_all_nsclc
        else "B  Cancer-specific associations\n" + subtitle
    )
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=10)
    axis.grid(axis="x", which="both", alpha=.2)
    axis.set_xlim(shown.ci_low.min() * .72, shown.ci_high.max() * 1.55)
    return results


def panel_c(axis, simplified_title: bool = False) -> pd.DataFrame:
    distribution = pd.read_csv(RESPONSE / "evaluated_cohort_distribution.tsv", sep="\t")
    other = pd.read_csv(OTHER_RESPONSE_ASSIGNMENTS, sep="\t")
    nsclc = pd.read_csv(NSCLC_RESPONSE_ASSIGNMENTS, sep="\t")
    new = pd.read_csv(NEW_RESPONSE_ASSIGNMENTS, sep="\t")
    new = new.loc[new.study_id.isin([
        "blca_bcan_hcrn_2022", "mel_iatlas_riaz_nivolumab_2017",
    ])].copy()
    new_distribution = (new.groupby("study_id", as_index=False)
                         .agg(patients=("target", "size"),
                              responders=("target", "sum"),
                              response_rate=("target", "mean")))
    new_distribution["nonresponders"] = (
        new_distribution.patients - new_distribution.responders)
    new_distribution["cancer_group"] = new_distribution.study_id.map({
        "blca_bcan_hcrn_2022": "Bladder cancer",
        "mel_iatlas_riaz_nivolumab_2017": "Melanoma",
    })
    distribution = pd.concat([distribution, new_distribution], ignore_index=True)
    distribution = distribution.sort_values("response_rate", ascending=False).reset_index(drop=True)
    assignments = pd.concat([
        other[["study_id", "target", "cluster2"]],
        nsclc[["study_id", "target", "cluster2"]],
        new[["study_id", "target", "cluster2"]],
    ], ignore_index=True)
    assignments["cluster2"] = assignments.cluster2.astype(bool)

    rows = []
    for study in distribution.study_id:
        local = assignments.loc[assignments.study_id.eq(study)]
        for selected, group in [(True, "Cluster 2"), (False, "Remaining clusters")]:
            subset = local.loc[local.cluster2.eq(selected)]
            total = len(subset); responders = int(subset.target.sum())
            rate = responders / total if total else np.nan
            rows.append({"study_id": study, "group": group, "patients": total,
                         "responders": responders, "response_rate": rate})
    rates = pd.DataFrame(rows)

    # Test the cluster-2 versus remaining-cluster response association within
    # each evaluable cohort using an unadjusted two-sided Fisher exact test.
    tests = []
    for study, local in assignments.groupby("study_id"):
        cluster2 = local.loc[local.cluster2]
        remaining = local.loc[~local.cluster2]
        if len(cluster2) and len(remaining):
            table = [[int(cluster2.target.sum()), int(len(cluster2)-cluster2.target.sum())],
                     [int(remaining.target.sum()), int(len(remaining)-remaining.target.sum())]]
            odds_ratio, p_value = fisher_exact(table)
            tests.append({"study_id": study, "odds_ratio": odds_ratio,
                          "fisher_p": p_value})
    tests = pd.DataFrame(tests)
    tests["p_significant"] = tests.fisher_p.lt(.05)
    rates = rates.merge(tests, on="study_id", how="left")

    # Order by evidence for a cluster-2 response association. Cohorts without
    # both membership groups are non-estimable and appear last.
    distribution = (distribution.merge(
                        tests[["study_id", "fisher_p", "p_significant"]],
                        on="study_id", how="left")
                    .sort_values(["fisher_p", "response_rate"],
                                 ascending=[True, False], na_position="last")
                    .reset_index(drop=True))

    x = np.arange(len(distribution)); width = .36
    test_lookup = tests.set_index("study_id")
    for xpos, study in zip(x, distribution.study_id):
        if study in test_lookup.index and bool(test_lookup.loc[study, "p_significant"]):
            axis.axvspan(xpos-.48, xpos+.48, color="#fff2cc", alpha=.7, zorder=0)
    for selected, group, color, offset in [
        (True, "Cluster 2", RED, -width/2),
        (False, "Remaining clusters", GRAY, width/2),
    ]:
        shown = rates.loc[rates.group.eq(group)].set_index("study_id").loc[
            distribution.study_id
        ].reset_index()
        values = shown.response_rate.to_numpy(float)
        bars = axis.bar(x+offset, values, color=color, width=width, label=group)
        for bar, row in zip(bars, shown.itertuples()):
            if row.patients:
                axis.text(
                    bar.get_x()+bar.get_width()/2, min(row.response_rate+.018, 1.055),
                    f"{row.responders}/{row.patients}", ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold",
                )
            elif selected:
                axis.text(bar.get_x()+bar.get_width()/2, .018, "n=0",
                          ha="center", va="bottom", fontsize=7.5, color=RED)
    for xpos, study in zip(x, distribution.study_id):
        if study in test_lookup.index and bool(test_lookup.loc[study, "p_significant"]):
            axis.text(xpos, 1.085, f"★  p={test_lookup.loc[study, 'fisher_p']:.3g}",
                      ha="center", va="bottom", fontsize=8.5,
                      color="#8c2d04", fontweight="bold")
    cancer_short = {
        "Bladder cancer": "Bladder",
        "Breast cancer": "Breast",
        "Renal cell carcinoma": "Renal",
        "Pancreatic cancer": "Pancreatic",
    }
    labels = [
        f"{COHORT_NAMES.get(row.study_id, row.study_id)}\n"
        f"({cancer_short.get(row.cancer_group, row.cancer_group)})\n"
        f"(n={int(row.patients)})"
        for row in distribution.itertuples()
    ]
    axis.set_xticks(x, labels, rotation=0, ha="center")
    axis.tick_params(axis="x", labelsize=8.5, pad=7)
    axis.set_ylabel("Objective response rate")
    axis.set_ylim(0, 1.16)
    title = (
        "C  Objective response rates\n"
        "(other cancer type-specific immunotherapy cohorts)"
        if simplified_title
        else "C  Objective response by cluster membership\n"
             "Cohorts ordered by association P value"
    )
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=10)
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Patch(facecolor="#fff2cc", edgecolor="none",
                         label="p<0.05"))
    axis.legend(handles=handles, frameon=False, fontsize=10.5, loc="upper right")
    return rates


def main(exclude_all_nsclc: bool = False) -> None:
    output = (
        BASE / "experiments/cluster2_impact_and_ici_response_no_nsclc_20261106"
        if exclude_all_nsclc else OUTPUT
    )
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
    })
    impact = load_impact(exclude_all_nsclc=exclude_all_nsclc)
    subtitle = (
        "All NSCLC patients excluded"
        if exclude_all_nsclc
        else "Patients overlapping MSK-CHORD NSCLC excluded"
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(33.5, 8.8),
        gridspec_kw={"width_ratios": [1.05, .92, 2.2]},
        constrained_layout=True,
    )
    pan_cancer = panel_a(axes[0], impact, subtitle)
    cancer = panel_b(axes[1], exclude_all_nsclc, subtitle)
    response = panel_c(axes[2], simplified_title=exclude_all_nsclc)

    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"cluster2_impact_and_ici_response.{suffix}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    cancer.to_csv(output / "cancer_specific_unadjusted_hr.tsv", sep="\t", index=False)
    response.to_csv(output / "harmonized_ici_cohort_orr.tsv", sep="\t", index=False)
    manifest = {
        "impact_pan_cancer": pan_cancer,
        "impact_overlap_exclusion": (
            "All MSK-IMPACT NSCLC patients"
            if exclude_all_nsclc else "MSK-CHORD NSCLC patient IDs only"
        ),
        "response_cohorts": int(response.study_id.nunique()),
        "response_patients": int(response.groupby("study_id").patients.sum().sum()),
        "response_definition": "CR/PR or harmonized responder label",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude-all-msk-impact-nsclc",
        action="store_true",
        help="Remove every NSCLC patient from the MSK-IMPACT panels.",
    )
    args = parser.parse_args()
    main(exclude_all_nsclc=args.exclude_all_msk_impact_nsclc)

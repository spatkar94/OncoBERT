#!/usr/bin/env python3
"""Build a consolidated figure of CHORD cluster findings and external validation."""

from __future__ import annotations

import json
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.proportion import proportion_confint


ROOT = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments")) / "experiments"
CHORD = ROOT / "chord_nsclc_ici_clusters_clean"
OLD_EXTERNAL = ROOT / "cluster2_external_nsclc_ici_validation"
ALLEN = ROOT / "cluster2_external_mixed_allen_nsclc_validation"
OUTPUT = ROOT / "cluster2_consolidated_findings_20261103"

BLUE = "#4e79a7"
RED = "#d62728"
GRAY = "#9c9c9c"
ORANGE = "#f28e2b"


def km(axis, frame, membership, duration, event, title, annotate_p=True,
       ci=False, x_label="Months", y_label="Overall-survival probability"):
    groups = []
    for selected, label, color in [(True, "Cluster", RED),
                                    (False, "Remaining", GRAY)]:
        group = frame.loc[membership.eq(selected)].dropna(subset=[duration, event])
        groups.append(group)
        KaplanMeierFitter(label=(f"{label} "
                                 f"(n={len(group)}, events={int(group[event].sum())})")).fit(
            group[duration], group[event]).plot_survival_function(
                ax=axis, ci_show=ci, color=color, linewidth=2.0,
                ci_alpha=.13, show_censors=False)
    if annotate_p and all(len(group) for group in groups):
        test = logrank_test(groups[0][duration], groups[1][duration],
                            event_observed_A=groups[0][event],
                            event_observed_B=groups[1][event])
        axis.text(.97, .06, f"log-rank p={test.p_value:.3g}", transform=axis.transAxes,
                  ha="right", va="bottom", fontsize=11,
                  bbox={"facecolor": "white", "edgecolor": ".85", "alpha": .9})
    axis.set(title=title, xlabel=x_label, ylabel=y_label, ylim=(0, 1.03))
    axis.title.set_fontsize(14)
    axis.grid(axis="y", alpha=.18)
    axis.legend(frameon=False, fontsize=11, loc="upper right",
                handlelength=1.8, borderaxespad=.4)
    return groups


def panel_interaction(subfigure, interaction):
    axis = subfigure.subplots(1, 1)
    shown = interaction.sort_values(["fdr_q", "cluster"]).reset_index(drop=True)
    y = np.arange(len(shown))
    for ypos, row in zip(y, shown.itertuples()):
        estimate = row.interaction_ratio_of_hazard_ratios
        low = row.interaction_ci_low
        high = row.interaction_ci_high
        significant = bool(row.fdr_reject_0p05)
        color = "#d62728" if significant else BLUE
        axis.errorbar(estimate, ypos,
                      xerr=[[estimate-low], [high-estimate]],
                      fmt="o", markersize=6.5, color=color, ecolor=color,
                      elinewidth=1.7, capsize=3)
        weight = "bold" if significant else "normal"
        axis.text(high*1.07, ypos, f"q={row.fdr_q:.3g}", va="center",
                  fontsize=11, fontweight=weight, clip_on=False)
    axis.axvline(1, color="black", linestyle="--", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_yticks(y, [f"Cluster {value}" for value in shown.cluster])
    axis.invert_yaxis()
    axis.set_xlabel("Treatment × cluster ratio of hazard ratios (95% CI)")
    axis.set_title(
        "A  OncoBERT cluster interactions\n"
        "ICI versus non-ICI index treatment\n"
        "Values below 1 indicate relatively favorable survival with ICI",
        loc="left", fontsize=15, fontweight="bold", pad=12)
    axis.grid(axis="x", alpha=.2)
    axis.set_xlim(shown.interaction_ci_low.min()*.65,
                  shown.interaction_ci_high.max()*1.7)
    axis.text(.995, .02, "Red: FDR q<0.05", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=11, color="#d62728",
              fontweight="bold")


def panel_ab(subfigure, cohort, interaction, cluster, panel):
    axes = subfigure.subplots(1, 2, sharey=True)
    row = interaction.loc[interaction.cluster.eq(cluster)].iloc[0]
    member = cohort.assigned_cluster.astype(str).eq(str(cluster))
    for axis, treatment, title in zip(axes, [1, 0], ["ICI-treated", "Non-ICI-treated"]):
        local = cohort.loc[cohort.treatment.eq(treatment)].copy()
        km(axis, local, member.loc[local.index], "duration", "event", title,
           ci=True, x_label="Months from Treatment Start")
        axis.get_legend().get_texts()[0].set_text(
            axis.get_legend().get_texts()[0].get_text().replace("Cluster", f"Cluster {cluster}"))
        if axis is axes[1]:
            axis.set_ylabel("")
    axes[0].set_title("", loc="center")
    axes[0].set_title(
        f"{panel}  Cluster {cluster}: ICI-treated\n"
        f"Treatment×cluster interaction p={row.interaction_p_value:.3g}; "
        f"FDR q={row.fdr_q:.3g}",
        loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[1].set_title("Non-ICI-treated", fontsize=15, pad=10)


def panel_c(subfigure, cohort):
    """Multivariable OS model in ICI-index-treated NSCLC complete cases."""
    # Use explicit positions within this subfigure. Nested constrained-layout
    # grids allocated excessive space between the two KM axes because their
    # legends and y labels differed in width.
    axis = subfigure.add_axes([.10, .07, .43, .86])
    pdl1_axes = {
        "pdl1_positive": subfigure.add_axes([.60, .55, .38, .38]),
        "pdl1_negative": subfigure.add_axes([.60, .07, .38, .38]),
    }
    analysis = cohort.loc[
        cohort.pretreatment_pdl1.isin(["Yes", "No"])
    ].copy()
    analysis["cluster2"] = analysis.assigned_cluster.astype(str).eq("2").astype(int)
    log1p_tmb = np.log1p(analysis.TMB_NONSYNONYMOUS.astype(float))
    analysis["tmb_scaled"] = (
        (log1p_tmb - log1p_tmb.mean()) / log1p_tmb.std(ddof=0)
    )
    analysis["pdl1_positive"] = analysis.pretreatment_pdl1.eq("Yes").astype(int)
    columns = ["duration", "event", "cluster2", "tmb_scaled", "pdl1_positive"]
    analysis = analysis[columns].dropna().copy()
    analysis["duration"] = analysis.duration.clip(lower=1e-3)

    model = CoxPHFitter(penalizer=.001)
    model.fit(
        analysis,
        duration_col="duration",
        event_col="event",
        robust=True,
    )
    labels = {
        "cluster2": "Cluster 2\n(vs remaining clusters)",
        "tmb_scaled": "Scaled TMB\n(per 1-SD increase)",
        "pdl1_positive": "PD-L1 positive\n(vs negative)",
    }
    rows = []
    for covariate in ["cluster2", "tmb_scaled", "pdl1_positive"]:
        result = model.summary.loc[covariate]
        rows.append({
            "covariate": covariate,
            "label": labels[covariate],
            "hazard_ratio": float(result["exp(coef)"]),
            "ci_low": float(result["exp(coef) lower 95%"]),
            "ci_high": float(result["exp(coef) upper 95%"]),
            "p_value": float(result["p"]),
        })
    results = pd.DataFrame(rows)
    y = np.arange(len(results))
    axis.errorbar(
        results.hazard_ratio,
        y,
        xerr=np.vstack([
            results.hazard_ratio - results.ci_low,
            results.ci_high - results.hazard_ratio,
        ]),
        fmt="o",
        markersize=8,
        color=RED,
        ecolor=RED,
        elinewidth=2,
        capsize=4,
    )
    axis.axvline(1, color="black", linestyle="--", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_yticks(y, results.label)
    axis.invert_yaxis()
    for ypos, row in enumerate(results.itertuples()):
        axis.text(
            row.ci_high * 1.07,
            ypos,
            f"HR {row.hazard_ratio:.2f} ({row.ci_low:.2f}–{row.ci_high:.2f})\n"
            f"p={row.p_value:.3g}",
            va="center",
            fontsize=12,
            clip_on=False,
        )
    axis.set_xlim(results.ci_low.min() * .62, results.ci_high.max() * 2.35)
    axis.set_xlabel("Hazard ratio for overall survival (95% CI)")
    axis.set_title(
        "D  Multivariable Cox proportional-hazards analysis\n"
        f"ICI index-treated NSCLC · n={len(analysis):,}, deaths={int(analysis.event.sum()):,}",
        loc="left", fontsize=15, fontweight="bold", pad=10,
    )
    axis.grid(axis="x", which="both", alpha=.2)

    membership = cohort.assigned_cluster.astype(str).eq("2")
    for index, (key, value, title) in enumerate([
        ("pdl1_positive", "Yes", "PD-L1 positive"),
        ("pdl1_negative", "No", "PD-L1 negative"),
    ]):
        km_axis = pdl1_axes[key]
        local = cohort.loc[cohort.pretreatment_pdl1.eq(value)].copy()
        km(
            km_axis,
            local,
            membership.loc[local.index],
            "duration",
            "event",
            title,
            ci=True,
            x_label="Months from Treatment Start",
        )
        km_axis.get_legend().get_texts()[0].set_text(
            km_axis.get_legend().get_texts()[0].get_text().replace(
                "Cluster", "Cluster 2"
            )
        )
        if index:
            km_axis.set_ylabel("")

    results["patients"] = len(analysis)
    results["deaths"] = int(analysis.event.sum())
    results["model_concordance"] = float(model.concordance_index_)
    return results


def external_data():
    old = pd.read_csv(OLD_EXTERNAL / "external_nsclc_cluster_assignments.tsv.gz", sep="\t")
    old = old.rename(columns={"study_id": "cohort", "patient_id": "patient"})
    old["response"] = old.target.astype(int)
    old["duration"] = old.pfs_months
    old["event"] = old.pfs_event
    old = old[["cohort", "patient", "cluster2", "response", "duration", "event"]]
    allen = pd.read_csv(ALLEN / "nonoverlapping_nsclc_cluster_assignments.tsv", sep="\t")
    allen = allen[allen.RECIST.isin(["CR", "PR", "SD", "PD"])].copy()
    allen["cohort"] = "mixed_allen_2018"
    allen["patient"] = allen.PATIENT_ID
    allen["response"] = allen.RECIST.isin(["CR", "PR"]).astype(int)
    allen["duration"] = allen.PFS_MONTHS
    allen["event"] = allen.PFS_EVENT
    allen = allen[["cohort", "patient", "cluster2", "response", "duration", "event"]]
    combined = pd.concat([old, allen], ignore_index=True)
    if combined[["cohort", "patient"]].duplicated().any():
        raise ValueError("External validation contains duplicated cohort-patient keys")
    return combined


def response_plot(axis, frame):
    cohorts = ["nsclc_mskcc_2015", "nsclc_mskcc_2018", "mixed_allen_2018"]
    rows = []
    values, lower, upper = [], [], []
    labels, colors = ["Cluster 2", "Remaining clusters"], [RED, GRAY]
    for selected, group_label in zip([True, False], labels):
        local = frame.loc[frame.cluster2.eq(selected)]
        successes = int(local.response.sum()); total = len(local)
        value = successes / total
        lo, hi = proportion_confint(successes, total, alpha=.05, method="wilson")
        values.append(value); lower.append(lo); upper.append(hi)
        rows.append({"cohort": "Pooled", "group": group_label,
                     "responders": successes, "patients": total,
                     "response_rate": value, "ci_low": lo, "ci_high": hi})
    x = np.arange(2)
    bars = axis.bar(x, values, width=.58, color=colors)
    axis.errorbar(x, values,
                  yerr=np.vstack([np.asarray(values)-lower,
                                  np.asarray(upper)-values]),
                  fmt="none", color="black", linewidth=1.2, capsize=3)
    for bar, value, upper_bound, row in zip(bars, values, upper, rows):
        axis.text(bar.get_x()+bar.get_width()/2, upper_bound+.035,
                  f"{row['responders']}/{row['patients']} ({100*value:.1f}%)",
                  ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Estimate a common response odds ratio while allowing each cohort its own
    # baseline response prevalence.
    tables = []
    for cohort in cohorts:
        local = frame.loc[frame.cohort.eq(cohort)]
        c2 = local.loc[local.cluster2]
        remaining = local.loc[~local.cluster2]
        tables.append(np.array([[int(c2.response.sum()), int(len(c2)-c2.response.sum())],
                                [int(remaining.response.sum()),
                                 int(len(remaining)-remaining.response.sum())]]))
    stratified = StratifiedTable(tables)
    axis.text(.97, .96,
              f"Study-stratified OR {stratified.oddsratio_pooled:.2f}\n"
              f"95% CI {stratified.oddsratio_pooled_confint()[0]:.2f}–"
              f"{stratified.oddsratio_pooled_confint()[1]:.2f}; "
              f"p={stratified.test_null_odds().pvalue:.3g}",
              transform=axis.transAxes, ha="right", va="top", fontsize=11,
              bbox={"facecolor": "white", "edgecolor": ".85", "alpha": .9})
    axis.set(title="Pooled objective response (CR/PR)", ylabel="Response rate",
             xticks=x, xticklabels=labels, ylim=(0, 1.08))
    axis.grid(axis="y", alpha=.18)
    return pd.DataFrame(rows)


def pooled_pfs_plot(axis, frame):
    valid = frame.dropna(subset=["duration", "event"]).copy()
    valid["cluster2_int"] = valid.cluster2.astype(int)
    model = CoxPHFitter()
    model.fit(valid[["duration", "event", "cluster2_int", "cohort"]],
              duration_col="duration", event_col="event", strata=["cohort"])
    estimate = model.summary.loc["cluster2_int"]
    for selected, label, color in [(True, "Cluster 2", RED),
                                    (False, "Remaining", GRAY)]:
        local = valid.loc[valid.cluster2.eq(selected)]
        KaplanMeierFitter(label=f"{label} (n={len(local)}, events={int(local.event.sum())})").fit(
            local.duration, local.event).plot_survival_function(
                ax=axis, ci_show=True, ci_alpha=.13, color=color, linewidth=2.0)
    axis.text(.97, .96,
              f"Study-stratified HR {estimate['exp(coef)']:.2f}\n"
              f"95% CI {estimate['exp(coef) lower 95%']:.2f}–"
              f"{estimate['exp(coef) upper 95%']:.2f}; p={estimate.p:.3g}",
              transform=axis.transAxes, ha="right", va="top", fontsize=11,
              bbox={"facecolor": "white", "edgecolor": ".85", "alpha": .9})
    axis.set(title="Pooled external-cohort PFS", xlabel="PFS (months)",
             ylabel="Progression-free probability", ylim=(0, 1.03))
    axis.grid(axis="y", alpha=.18)
    axis.legend(frameon=False, fontsize=11, loc="lower left")
    return {
        "patients": len(valid), "events": int(valid.event.sum()),
        "cluster2_patients": int(valid.cluster2.sum()),
        "cluster2_events": int(valid.loc[valid.cluster2, "event"].sum()),
        "hazard_ratio": float(estimate["exp(coef)"]),
        "ci_low": float(estimate["exp(coef) lower 95%"]),
        "ci_high": float(estimate["exp(coef) upper 95%"]),
        "p_value": float(estimate.p),
    }


def panel_d(subfigure, external):
    axes = subfigure.subplots(1, 2)
    rates = response_plot(axes[0], external)
    pfs = pooled_pfs_plot(axes[1], external)
    axes[0].set_title("", loc="center")
    axes[0].set_title("E  Independent external validation\n"
                      "Pooled objective response (CR/PR)",
                      loc="left", fontsize=15, fontweight="bold", pad=10)
    axes[1].set_title("Pooled external-cohort PFS", fontsize=15, pad=10)
    return rates, pfs


def main():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    chord = pd.read_csv(CHORD / "ici_vs_nonici_analysis_cohort.tsv.gz", sep="\t")
    interaction = pd.read_csv(CHORD / "ici_vs_nonici_cluster_interactions.tsv", sep="\t")
    biomarker = pd.read_csv(
        CHORD / "cluster2_biomarker_strata/cluster2_biomarker_strata_cohort.tsv.gz", sep="\t")
    external = external_data()

    # The full-width interaction forest provides the inferential starting point;
    # subsequent panels show the two FDR-significant clusters and validation.
    # Panels B--E contain pairs or grids of KM axes, so allocate substantially
    # more horizontal space to those columns than to the single forest plot.
    fig = plt.figure(figsize=(46, 20), constrained_layout=True)
    outer = fig.add_gridspec(2, 3, width_ratios=[.48, 1.25, 1],
                             height_ratios=[.80, 1.20])
    panel_interaction(fig.add_subfigure(outer[:, 0]), interaction)
    panel_ab(fig.add_subfigure(outer[0, 1]), chord, interaction, 2, "B")
    panel_ab(fig.add_subfigure(outer[0, 2]), chord, interaction, 7, "C")
    multivariable_cox = panel_c(fig.add_subfigure(outer[1, 1]), biomarker)
    rates, pfs = panel_d(fig.add_subfigure(outer[1, 2]), external)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"cluster2_consolidated_findings.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)

    rates.to_csv(OUTPUT / "external_objective_response_rates.tsv", sep="\t", index=False)
    multivariable_cox.to_csv(
        OUTPUT / "ici_index_multivariable_cox_cluster2_tmb_pdl1.tsv",
        sep="\t", index=False,
    )
    external.to_csv(OUTPUT / "external_pooled_patient_cohort.tsv.gz", sep="\t", index=False)
    (OUTPUT / "external_pooled_pfs_result.json").write_text(
        json.dumps(pfs, indent=2) + "\n")
    manifest = {
        "panels": {
            "A": "MSK-CHORD ICI-versus-non-ICI treatment-by-cluster interaction forest",
            "B": "MSK-CHORD cluster 2 vs remaining, stratified by ICI exposure",
            "C": "MSK-CHORD cluster 7 vs remaining, stratified by ICI exposure",
            "D": "MSK-CHORD ICI-index-treated multivariable Cox model plus PD-L1 subgroup-specific Cluster 2 KM curves",
            "E": "Non-overlapping external NSCLC cohorts: objective response and pooled PFS",
        },
        "external_patient_count": len(external),
        "external_cohorts": external.groupby("cohort").size().astype(int).to_dict(),
        "pooled_pfs": pfs,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

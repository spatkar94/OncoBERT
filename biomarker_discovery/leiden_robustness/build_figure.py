#!/usr/bin/env python3
"""Plot Cluster 2–ICI phenotype robustness across Leiden resolutions and seeds."""

from __future__ import annotations

from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments"))
SENSITIVITY = ROOT / "experiments/cluster2_ici_leiden_sensitivity_20261107"
INPUT = SENSITIVITY / "cluster2_ici_leiden_sensitivity.tsv"
REFERENCE = (ROOT / "experiments/chord_nsclc_ici_clusters_clean/"
             "ici_vs_nonici_cluster_interactions.tsv")
OUTPUT = SENSITIVITY / "phenotype_robustness_figure"

RESOLUTIONS = (0.10, 0.25, 0.50, 0.75, 1.00)
BLUE = "#4E79A7"
LIGHT_BLUE = "#A0CBE8"
REFERENCE_COLOR = "#333333"
OVERALL_COLOR = "#6B6B6B"


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def clean_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)


def annotate_bar(ax: plt.Axes, rect: plt.Rectangle, numerator: int,
                 denominator: int, *, inside: bool = False) -> None:
    height = rect.get_height()
    if inside:
        y = height - 0.055
        va = "top"
        color = "white"
    else:
        y = height + 0.025
        va = "bottom"
        color = "black"
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        y,
        f"{100 * height:.0f}%\n({numerator}/{denominator})",
        ha="center", va=va, fontsize=9.2, fontweight="bold", color=color,
    )


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t")
    data = data[data.evaluable.astype(bool)].copy()
    data["nominally_significant"] = data.interaction_p_value.lt(0.05)

    reference = pd.read_csv(REFERENCE, sep="\t")
    original = reference.loc[reference.cluster.astype(str).eq("2")].iloc[0]
    original_hr = float(original.interaction_ratio_of_hazard_ratios)
    original_low = float(original.interaction_ci_low)
    original_high = float(original.interaction_ci_high)

    summary = data.groupby("resolution", sort=True).agg(
        solutions=("seed", "size"),
        median_interaction=("interaction_ratio_of_hazard_ratios", "median"),
        minimum_interaction=("interaction_ratio_of_hazard_ratios", "min"),
        maximum_interaction=("interaction_ratio_of_hazard_ratios", "max"),
        same_direction_n=("same_favorable_ici_direction", "sum"),
        same_direction_fraction=("same_favorable_ici_direction", "mean"),
        significant_n=("nominally_significant", "sum"),
        significant_fraction=("nominally_significant", "mean"),
        median_interaction_p_value=("interaction_p_value", "median"),
        minimum_interaction_p_value=("interaction_p_value", "min"),
        maximum_interaction_p_value=("interaction_p_value", "max"),
    ).reindex(RESOLUTIONS)

    pooled = pd.DataFrame({
        "resolution": ["Overall"],
        "solutions": [len(data)],
        "median_interaction": [data.interaction_ratio_of_hazard_ratios.median()],
        "minimum_interaction": [data.interaction_ratio_of_hazard_ratios.min()],
        "maximum_interaction": [data.interaction_ratio_of_hazard_ratios.max()],
        "same_direction_n": [int(data.same_favorable_ici_direction.sum())],
        "same_direction_fraction": [data.same_favorable_ici_direction.mean()],
        "significant_n": [int(data.nominally_significant.sum())],
        "significant_fraction": [data.nominally_significant.mean()],
        "median_interaction_p_value": [data.interaction_p_value.median()],
        "minimum_interaction_p_value": [data.interaction_p_value.min()],
        "maximum_interaction_p_value": [data.interaction_p_value.max()],
    })
    output_summary = summary.reset_index()
    output_summary["resolution"] = output_summary.resolution.map(lambda x: f"{x:g}")
    output_summary = pd.concat([output_summary, pooled], ignore_index=True)
    output_summary["original_interaction_ratio_of_hazard_ratios"] = original_hr
    output_summary["original_interaction_ci_low"] = original_low
    output_summary["original_interaction_ci_high"] = original_high

    fig = plt.figure(figsize=(14.2, 6.4))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.12, 1.0), wspace=0.28)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    # A: all 100 effect estimates, grouped by resolution (horizontal layout).
    positions = np.arange(len(RESOLUTIONS), dtype=float)
    values = [
        data.loc[data.resolution.eq(res), "interaction_ratio_of_hazard_ratios"].to_numpy()
        for res in RESOLUTIONS
    ]
    boxes = ax_a.boxplot(
        values, positions=positions, widths=0.52, patch_artist=True,
        vert=False,
        showfliers=False, whis=(0, 100),
        medianprops={"color": "white", "linewidth": 2.0},
        whiskerprops={"color": "0.35", "linewidth": 1.2},
        capprops={"color": "0.35", "linewidth": 1.2},
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.88)

    rng = np.random.default_rng(20261108)
    for index, vals in enumerate(values):
        jitter = rng.uniform(-0.17, 0.17, size=len(vals))
        ax_a.scatter(
            vals, index + jitter, s=24, color=BLUE, edgecolor="white",
            linewidth=0.45, alpha=0.95, zorder=3,
        )

    ax_a.axvline(1.0, color="0.35", linestyle=":", linewidth=1.4,
                 label="Null interaction (RHR=1)")
    ax_a.axvline(original_hr, color=REFERENCE_COLOR, linestyle="--", linewidth=2.0,
                 label=f"Original Cluster 2: RHR={original_hr:.3f}")
    ax_a.axvspan(
        original_low, original_high,
        color="0.6", alpha=0.10,
        label=f"Original 95% CI ({original_low:.3f}–{original_high:.3f})",
    )
    ax_a.set_yticks(positions, [f"{x:g}  (n=20)" for x in RESOLUTIONS])
    ax_a.set_ylim(-0.5, len(RESOLUTIONS) - 0.5)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.45, 1.045)
    ax_a.set(
        xlabel="Interaction ratio of hazard ratios",
        ylabel="Leiden resolution",
        title="A  Interaction-effect distribution",
    )
    legend_handles, legend_labels = ax_a.get_legend_handles_labels()
    clean_axis(ax_a)

    # B: the actual interaction p-value distributions at each resolution.
    p_values = [
        data.loc[data.resolution.eq(res), "interaction_p_value"].to_numpy()
        for res in RESOLUTIONS
    ]
    p_boxes = ax_b.boxplot(
        p_values, positions=positions, widths=0.52, patch_artist=True,
        vert=False, showfliers=False, whis=(0, 100),
        medianprops={"color": "white", "linewidth": 2.0},
        whiskerprops={"color": "0.35", "linewidth": 1.2},
        capprops={"color": "0.35", "linewidth": 1.2},
    )
    for patch in p_boxes["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.88)
    for index, vals in enumerate(p_values):
        jitter = rng.uniform(-0.17, 0.17, size=len(vals))
        ax_b.scatter(
            vals, index + jitter, s=24, color=BLUE, edgecolor="white",
            linewidth=0.45, alpha=0.95, zorder=3,
        )
    ax_b.axvline(0.05, color=REFERENCE_COLOR, linestyle="--", linewidth=1.8)
    ax_b.set_xscale("log")
    ax_b.set_yticks(positions, [f"{x:g}  (n=20)" for x in RESOLUTIONS])
    ax_b.set_ylim(-0.5, len(RESOLUTIONS) - 0.5)
    ax_b.invert_yaxis()
    ax_b.set_xlim(2e-4, 0.6)
    ax_b.set(
        xlabel="Interaction p-value (log scale)",
        ylabel="Leiden resolution",
        title="B  Interaction p-value distribution",
    )
    ax_b.text(
        0.061, -0.34, "p=0.05", ha="left", va="center", fontsize=9.2,
        color=REFERENCE_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    clean_axis(ax_b)

    # Keep the reference key outside the shaded data region and away from the
    # vertical reference lines.
    fig.legend(
        legend_handles, legend_labels, frameon=False, fontsize=9.0,
        loc="upper center", bbox_to_anchor=(0.31, 0.905), ncol=3,
        handlelength=2.4, columnspacing=1.5,
    )

    fig.suptitle(
        "Cluster 2–ICI interaction robustness across Leiden solutions",
        fontsize=18, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.015,
        "One hundred Leiden solutions: five resolutions × 20 random seeds. Interaction ratios below 1 indicate relatively favorable survival with ICI; "
        "Panel B shows the corresponding nominal treatment × cluster interaction p-values.",
        ha="center", va="bottom", fontsize=9.5, color="0.30",
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.81)
    save(fig, "cluster2_ici_phenotype_robustness")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_summary.to_csv(
        OUTPUT / "cluster2_ici_phenotype_robustness_summary.tsv",
        sep="\t", index=False,
    )


if __name__ == "__main__":
    main()

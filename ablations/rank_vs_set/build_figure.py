#!/usr/bin/env python3
"""Build the rank-versus-set retrieval and survival composite figure."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REVISION = Path(os.environ.get(
    "ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]
))
ARTIFACTS = REVISION / "artifacts"
OUTPUT = ARTIFACTS / "rank_vs_set_composite_benchmark"
REPRODUCED = REVISION / "reproduced_figures"

RETRIEVAL_MODELS = [
    "ESM2 rank", "GenePT rank", "STRING rank", "OncoBERT (Set Transformer)"
]
SURVIVAL_KEYS = ["rank_esm2", "rank_genept", "rank_string", "three_state_logofunc"]
DISPLAY = ["ESM2 rank", "GenePT rank", "STRING rank", "OncoBERT (Set Transformer)"]
COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
TOP_K = (1, 3, 5, 10)


def retrieval_panel(axis, table: pd.DataFrame, title: str, panel: str) -> None:
    data = table.set_index("model").loc[RETRIEVAL_MODELS]
    x = np.arange(len(TOP_K), dtype=float)
    width = 0.19
    for model_index, (model, color) in enumerate(zip(RETRIEVAL_MODELS, COLORS)):
        row = data.loc[model]
        values = np.array([row[f"top{k}_accuracy"] for k in TOP_K]) * 100
        lower = np.array([row[f"top{k}_ci_lower"] for k in TOP_K]) * 100
        upper = np.array([row[f"top{k}_ci_upper"] for k in TOP_K]) * 100
        bars = axis.bar(
            x + (model_index - 1.5) * width, values, width,
            yerr=np.vstack([values - lower, upper - values]),
            color=color, edgecolor="white", linewidth=0.6,
            error_kw={"ecolor": "#222222", "elinewidth": 1.1,
                       "capsize": 3, "capthick": 1.1},
            label=model, zorder=3,
        )
        axis.bar_label(
            bars, labels=[f"{value:.1f}" for value in values],
            padding=3, fontsize=9.6, rotation=90,
        )

    class_count = int(data.standardized_class_count.iloc[0])
    chance = 100 * np.asarray(TOP_K, dtype=float) / class_count
    axis.hlines(
        chance, x - 0.43, x + 0.43, color="black",
        linestyle=(0, (5, 3)), linewidth=1.7, zorder=4,
    )
    axis.set_xticks(x, [f"Top-{k}" for k in TOP_K])
    axis.set_ylabel("Held-out sample retrieval accuracy (%)", fontsize=12.5)
    axis.set_title(f"{panel}  {title} ({class_count} classes)", loc="left",
                   fontsize=15, fontweight="semibold", pad=10)
    axis.tick_params(axis="both", labelsize=12)
    axis.set_ylim(0, max(38 if class_count > 100 else 64,
                         100 * data.top10_ci_upper.max() + 6))
    axis.grid(axis="y", alpha=0.20, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def survival_panel(axis, table: pd.DataFrame) -> None:
    cancers = [
        "Non-Small Cell Lung Cancer", "Colorectal Cancer", "Breast Cancer",
        "Pancreatic Cancer", "Prostate Cancer",
    ]
    short = ["NSCLC", "Colorectal", "Breast", "Pancreatic", "Prostate"]
    x = np.arange(len(cancers), dtype=float)
    width = 0.19
    for model_index, (key, label, color) in enumerate(zip(SURVIVAL_KEYS, DISPLAY, COLORS)):
        subset = table.loc[table.model.eq(key)].set_index("cancer").loc[cancers]
        values = subset.harrell_c_mean.to_numpy()
        errors = subset.harrell_c_std.to_numpy()
        bars = axis.bar(
            x + (model_index - 1.5) * width, values, width,
            yerr=errors, color=color, edgecolor="white", linewidth=0.6,
            error_kw={"ecolor": "#222222", "elinewidth": 1.25,
                       "capsize": 4, "capthick": 1.25},
            label=label, zorder=3,
        )
        axis.bar_label(
            bars, labels=[f"{value:.3f}" for value in values],
            padding=5, fontsize=9.6, rotation=90,
        )
    axis.axhline(0.5, color="#555555", linestyle=(0, (5, 3)), linewidth=1.5)
    axis.set_xticks(x, short)
    axis.set_ylabel("Held-out Harrell C-index", fontsize=13)
    axis.set_ylim(0.45, 0.735)
    axis.set_title("C  MSK-CHORD overall-survival prediction", loc="left",
                   fontsize=15.5, fontweight="semibold", pad=10)
    axis.tick_params(axis="both", labelsize=12.5)
    axis.grid(axis="y", alpha=0.20, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    broad = pd.read_csv(
        ARTIFACTS / "standardized_sample_level_retrieval/genie/broad/model_topk_metrics.csv"
    )
    detailed = pd.read_csv(
        ARTIFACTS / "standardized_sample_level_retrieval/genie/detailed/model_topk_metrics.csv"
    )
    survival = pd.read_csv(ARTIFACTS / "slide07_rank_vs_set/chord_os_metrics.csv")

    figure = plt.figure(figsize=(22, 12.5), facecolor="white")
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.05, 1.72], height_ratios=[1, 1],
        left=0.070, right=0.985, bottom=0.080, top=0.775,
        wspace=0.19, hspace=0.34,
    )
    broad_axis = figure.add_subplot(grid[0, 0])
    detailed_axis = figure.add_subplot(grid[1, 0])
    survival_axis = figure.add_subplot(grid[:, 1])
    retrieval_panel(broad_axis, broad, "GENIE Broad Cancer Type Retrieval", "A")
    retrieval_panel(detailed_axis, detailed, "GENIE Detailed Cancer Type Retrieval", "B")
    survival_panel(survival_axis, survival)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white", label=label)
        for label, color in zip(DISPLAY, COLORS)
    ]
    handles.append(Line2D([0], [0], color="black", linestyle=(0, (5, 3)),
                          linewidth=1.7, label="Random / chance baseline"))
    figure.legend(
        handles=handles, ncol=5, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.922), fontsize=12,
        handlelength=1.8, columnspacing=1.45,
    )
    figure.suptitle(
        "Rank-based encoders versus OncoBERT Set Transformer",
        fontsize=22, fontweight="semibold", y=0.985,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPRODUCED.mkdir(parents=True, exist_ok=True)
    stem = "rank_vs_set_retrieval_and_chord_os_composite"
    for extension in ("pdf", "png", "svg", "tiff"):
        destination = OUTPUT / f"{stem}.{extension}"
        figure.savefig(destination, dpi=300, facecolor="white")
        figure.savefig(REPRODUCED / f"{stem}.{extension}", dpi=300, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()

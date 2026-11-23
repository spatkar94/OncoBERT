"""Plot original three-state and binary mutation macro AUPRC by embedding dimension."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get(
    "ONCOBERT_EXPERIMENT_ROOT",
    Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments",
))
OUTPUT = ROOT / "experiments/three_state_set_dim_ablation_shared200_v20/results"
DIMENSIONS = (32, 64, 128, 256)
COLORS = ("#4E79A7", "#F28E2B", "#59A14F", "#B07AA1")


def main():
    table = pd.read_csv(OUTPUT / "dimension_ablation_metrics.csv").sort_values("embedding_dim")
    concise = table[[
        "embedding_dim", "three_state_macro_auprc", "binary_mutation_macro_auprc",
        "three_state_patient_bootstrap_95ci_lower",
        "three_state_patient_bootstrap_95ci_upper",
        "binary_patient_bootstrap_95ci_lower",
        "binary_patient_bootstrap_95ci_upper",
        "reconstruction_ce", "best_epoch",
    ]].rename(columns={
        "three_state_macro_auprc": "original_global_three_state_macro_auprc",
        "binary_mutation_macro_auprc": "per_gene_binary_mutation_macro_auprc",
    })
    concise.to_csv(OUTPUT / "original_state_and_binary_macro_auprc.csv", index=False)

    centers = np.arange(2, dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(DIMENSIONS))
    fig, axis = plt.subplots(figsize=(12.4, 7.2))
    for dimension, offset, color in zip(DIMENSIONS, offsets, COLORS):
        row = concise.loc[concise.embedding_dim == dimension].iloc[0]
        values = np.array([
            row.original_global_three_state_macro_auprc,
            row.per_gene_binary_mutation_macro_auprc,
        ])
        lower = np.array([
            row.three_state_patient_bootstrap_95ci_lower,
            row.binary_patient_bootstrap_95ci_lower,
        ])
        upper = np.array([
            row.three_state_patient_bootstrap_95ci_upper,
            row.binary_patient_bootstrap_95ci_upper,
        ])
        bars = axis.bar(
            centers + offset, values, width=0.16, color=color,
            edgecolor="white", linewidth=0.7,
            yerr=np.vstack([values - lower, upper - values]),
            error_kw={"ecolor": "#222222", "elinewidth": 1.25,
                      "capsize": 4, "capthick": 1.25},
            label=f"{dimension}D", zorder=3,
        )
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values],
                       padding=8, fontsize=10, rotation=90)

    axis.set_xticks(centers, [
        "Three state\n(WT / mutated-not-LOF / LOF)",
        "Two state\n(WT / mutated)",
    ])
    axis.set_xlim(-0.52, 1.52)
    axis.set_ylim(0.0, 0.54)

    three_state_random = 1.0 / 3.0
    prevalence = pd.read_csv(OUTPUT / "per_gene_binary_mutation_auprc.csv")
    binary_random = float(
        prevalence.loc[prevalence.embedding_dim == DIMENSIONS[0], "prevalence"].mean()
    )
    axis.hlines(three_state_random, -0.40, 0.40, color="black",
                linestyle=(0, (5, 3)), linewidth=1.8, zorder=5)
    axis.hlines(binary_random, 0.60, 1.40, color="black",
                linestyle=(0, (5, 3)), linewidth=1.8, zorder=5)
    axis.set_xlabel("Mutation-state prediction task", fontsize=14)
    axis.set_ylabel("Macro AUPRC", fontsize=14)
    axis.tick_params(axis="both", labelsize=12)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=1.8,
                          linestyle=(0, (5, 3))))
    labels.append("Random guessing")
    fig.legend(
        handles, labels,
        title="Embedding dimension", frameon=False, ncol=1,
        loc="upper left", bbox_to_anchor=(0.80, 0.77),
        fontsize=11.5, title_fontsize=11.5,
    )
    axis.grid(axis="y", alpha=.22)
    axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Set Transformer embedding-dimension ablation", fontsize=18, y=.98)
    fig.subplots_adjust(bottom=.14, top=.84, left=.10, right=.78)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"original_state_and_binary_macro_auprc.{extension}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(concise.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

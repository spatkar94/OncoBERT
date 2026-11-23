#!/usr/bin/env python3
"""Export the normalized-Laplacian eigengap panel from the Cluster 2 K sweep."""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
INPUT = (
    ROOT
    / "artifacts/cluster2_attention_modules/spectral_k_sweep"
    / "spectral_k_sweep_summary.tsv"
)
OUTPUT_DIR = INPUT.parent
REPRODUCED_DIR = ROOT / "reproduced_figures"
SELECTED_K = 6


def main() -> None:
    summary = pd.read_csv(INPUT, sep="\t").sort_values("k")

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.plot(
        summary["k"],
        summary["eigengap_after_k"],
        color="#e07a5f",
        marker="o",
        markersize=6,
        linewidth=2.2,
    )
    ax.axvline(
        SELECTED_K,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Selected K = {SELECTED_K}",
    )
    ax.set_title("Normalized-Laplacian eigengap", fontsize=18, pad=12)
    ax.set_xlabel("Number of spectral modules (K)", fontsize=14)
    ax.set_ylabel(r"$\lambda_{K+1}-\lambda_K$", fontsize=16)
    ax.set_xticks(range(2, 31, 2))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=12, loc="upper right")
    fig.tight_layout()

    stem = "cluster2_normalized_laplacian_eigengap"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPRODUCED_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{extension}",
            facecolor="white",
            bbox_inches="tight",
            **kwargs,
        )
        fig.savefig(
            REPRODUCED_DIR / f"{stem}.{extension}",
            facecolor="white",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()

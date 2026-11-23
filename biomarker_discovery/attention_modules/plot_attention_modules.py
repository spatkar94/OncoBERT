#!/usr/bin/env python3
"""Plot the full Cluster-2 K=6 attention matrix with six module zooms."""

from __future__ import annotations

from pathlib import Path
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "artifacts/cluster2_attention_modules/spectral_k6"
ATTENTION = BASE / "spectral_ordered_attention.tsv.gz"
MEMBERSHIP = BASE / "spectral_module_membership.tsv"
OUTPUT = BASE / "full_with_module_insets"
REPRODUCED = ROOT / "reproduced_figures"
STEM = "cluster2_k6_attention_full_with_module_insets"

MODULE_COLORS = ["#4E79A7", "#59A14F", "#9C755F", "#76B7B2", "#B07AA1", "#F28E2B"]
MODULE_THEMES = {
    1: "RAS–MAPK and receptor-kinase signaling",
    2: "Cellular senescence and G1/S checkpoint control",
    3: "Transcriptional/epigenetic and TGF-β control",
    4: "Double-strand-break repair and lineage specification",
    5: "Chromatin remodeling and genome maintenance",
    6: "Receptor tyrosine kinase–PI3K/AKT signaling",
}


def maximum_safe_label_size(axis: plt.Axes, n_genes: int) -> float:
    """Estimate the largest tick-label font that avoids adjacent collisions."""
    # The shorter physical axis controls the available center-to-center spacing.
    bbox = axis.get_window_extent().transformed(axis.figure.dpi_scale_trans.inverted())
    spacing_points = min(bbox.width, bbox.height) * 72.0 / max(n_genes, 1)
    # Rotated x labels and horizontal y labels can safely occupy most—but not
    # all—of the tick-to-tick distance.  The retained 14% gap prevents glyphs
    # from touching even for wide capitals and PDF/SVG font substitution.
    return float(np.clip(spacing_points * 0.86, 8.0, 16.0))


def module_heatmap(
    axis: plt.Axes,
    block: np.ndarray,
    genes: np.ndarray,
    module: int,
    vlim: float,
    panel_letter: str,
) -> None:
    """Draw an enlarged within-module matrix with maximized gene-label text."""
    axis.imshow(
        block, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        interpolation="nearest", aspect="equal", rasterized=True,
    )
    ticks = np.arange(len(genes))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    # Calculate after the axes have been laid out and the renderer is available.
    label_size = maximum_safe_label_size(axis, len(genes))
    axis.set_xticklabels(genes, rotation=90, ha="center", va="top",
                         fontsize=label_size, fontweight="normal")
    axis.set_yticklabels(genes, fontsize=label_size, fontweight="normal")
    axis.tick_params(axis="both", length=0, pad=1.0)
    axis.set_title(
        f"{panel_letter}  M{module} (n={len(genes)})\n{MODULE_THEMES[module]}",
        fontsize=17.5, fontweight="semibold",
        color=MODULE_COLORS[module - 1], pad=10, linespacing=1.12,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#444444")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPRODUCED.mkdir(parents=True, exist_ok=True)

    attention = pd.read_csv(ATTENTION, sep="\t", index_col=0)
    membership = pd.read_csv(MEMBERSHIP, sep="\t").sort_values("display_order")
    genes = attention.index.astype(str).to_numpy()
    if not np.array_equal(genes, membership.gene.astype(str).to_numpy()):
        raise RuntimeError("K=6 attention and membership orders differ")
    modules = membership.spectral_module.to_numpy(int)
    module_ids = sorted(np.unique(modules))
    matrix = attention.to_numpy(float)
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    vlim = float(np.quantile(np.abs(off_diagonal), 0.995))

    # The right-hand region follows the earlier 4-over-2 layout.  Its enlarged
    # canvas makes the gene symbols substantially larger without collisions.
    fig = plt.figure(figsize=(48, 32), facecolor="white")
    outer = fig.add_gridspec(
        1, 2, width_ratios=[0.72, 2.28],
        left=0.018, right=0.995, bottom=0.040, top=0.950, wspace=0.075,
    )

    full_axis = fig.add_subplot(outer[0, 0])
    divider = make_axes_locatable(full_axis)
    module_strip = divider.append_axes("top", size="2.8%", pad=0.08,
                                      sharex=full_axis)
    color_axis = divider.append_axes("bottom", size="3.5%", pad=0.70)
    full_image = full_axis.imshow(
        matrix, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
        interpolation="nearest", aspect="equal", rasterized=True,
    )
    module_strip.imshow(
        modules[None, :], aspect="auto", interpolation="nearest",
        cmap=ListedColormap(MODULE_COLORS), vmin=1, vmax=6,
        extent=(-0.5, len(modules) - 0.5, 0, 1), origin="lower",
    )
    module_strip.set_axis_off()
    for module in module_ids:
        selected = np.flatnonzero(modules == module)
        start = float(selected.min()) - 0.5
        size = float(len(selected))
        full_axis.add_patch(Rectangle(
            (start, start), size, size, fill=False,
            edgecolor=MODULE_COLORS[module - 1], linewidth=3.0, zorder=5,
        ))
        module_strip.text(
            selected.mean(), 1.20, f"M{module}", ha="center", va="bottom",
            fontsize=15, fontweight="semibold", clip_on=False,
        )
    full_axis.set_xticks([])
    full_axis.set_yticks([])
    full_axis.set_xlabel("Key gene", fontsize=18, labelpad=12)
    full_axis.set_ylabel("Query gene", fontsize=18, labelpad=12)
    full_axis.set_title(
        "A  Full module-ordered attention matrix", loc="left",
        fontsize=21, fontweight="semibold", pad=50,
    )
    colorbar = fig.colorbar(full_image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        "Bidirectionally averaged attention: log$_2$(observed / expected)",
        fontsize=15, labelpad=7,
    )
    colorbar.ax.tick_params(labelsize=13)

    right = outer[0, 1].subgridspec(
        2, 1, height_ratios=[0.39, 0.61], hspace=0.075,
    )
    top = right[0].subgridspec(1, 4, wspace=0.28)
    bottom = right[1].subgridspec(1, 2, wspace=0.19)
    axes = {
        1: fig.add_subplot(top[0, 0]),
        3: fig.add_subplot(top[0, 1]),
        4: fig.add_subplot(top[0, 2]),
        6: fig.add_subplot(top[0, 3]),
        2: fig.add_subplot(bottom[0, 0]),
        5: fig.add_subplot(bottom[0, 1]),
    }
    # Ensure physical axis sizes are known before calculating label fonts.
    fig.canvas.draw()
    letters = {1: "B", 3: "C", 4: "D", 6: "E", 2: "F", 5: "G"}
    for module, axis in axes.items():
        selected = np.flatnonzero(modules == module)
        module_heatmap(
            axis,
            matrix[np.ix_(selected, selected)],
            genes[selected],
            module,
            vlim,
            letters[module],
        )

    for extension in ("pdf", "png", "svg"):
        output_path = OUTPUT / f"{STEM}.{extension}"
        reproduced_path = REPRODUCED / f"{STEM}.{extension}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        shutil.copy2(output_path, reproduced_path)
    plt.close(fig)


if __name__ == "__main__":
    main()

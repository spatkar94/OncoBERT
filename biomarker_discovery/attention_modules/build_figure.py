#!/usr/bin/env python3
"""Generate the K=6 version of the Cluster-2 attention composite."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import common_plotting as common  # noqa: E402

BASE = ROOT / "artifacts/cluster2_attention_modules/spectral_k6"
ATTENTION = BASE / "spectral_ordered_attention.tsv.gz"
MEMBERSHIP = BASE / "spectral_module_membership.tsv"
OUTPUT = BASE / "composite"
MUTATION_OUTPUT = BASE / "mutation_patterns"
IG_OUTPUT = BASE / "integrated_gradients"
REPRODUCED = ROOT / "reproduced_figures"
STEM = "cluster2_k6_attention_interpretability_composite"
RAW_IG = ROOT / "artifacts/slide11_integrated_gradients/patient_gene_integrated_gradients.npz"

MODULE_COLORS = ["#4E79A7", "#59A14F", "#9C755F", "#76B7B2", "#B07AA1", "#F28E2B"]
MODULE_THEMES = {
    1: "RAS–MAPK and receptor-kinase signaling",
    2: "Cellular senescence and G1/S checkpoint control",
    3: "Transcriptional/epigenetic control\nand TGF-β tumor-suppressor signaling",
    4: "Double-strand-break repair\nand lineage specification",
    5: "Chromatin remodeling and\ngenome-maintenance programs",
    6: "Receptor tyrosine kinase–PI3K/AKT signaling",
}
BOOTSTRAPS = 2000
SEED = 42


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Patient-resampling percentile interval for a mean, in small batches."""
    means = []
    batch_size = 100
    for start in range(0, BOOTSTRAPS, batch_size):
        count = min(batch_size, BOOTSTRAPS - start)
        draws = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[draws].mean(axis=1))
    return tuple(np.quantile(np.concatenate(means), [0.025, 0.975]))


def load_patient_states() -> tuple[np.ndarray, np.ndarray]:
    """Rebuild patient-level three-state profiles in the original row order."""
    with np.load(common.SOURCE_RUN / "profiles.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str)
        sample_patients = archive["patients"].astype(str)
    sample_states = np.load(common.SOURCE_RUN / "functional_states.npy", mmap_mode="r")
    order = np.argsort(sample_patients, kind="stable")
    sorted_patients = sample_patients[order]
    boundaries = np.r_[
        0, np.flatnonzero(sorted_patients[1:] != sorted_patients[:-1]) + 1
    ]
    states = np.maximum.reduceat(sample_states[order], boundaries, axis=0)
    return genes, states


def calculate_summaries(
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame],
           dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """Calculate K=6 mutation, attribution, and co-mutation summaries."""
    MUTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    IG_OUTPUT.mkdir(parents=True, exist_ok=True)
    genes, patient_states = load_patient_states()
    gene_index = pd.Index(genes)
    local_membership = membership.copy()
    local_membership["gene_index"] = gene_index.get_indexer(
        local_membership.gene.astype(str)
    )
    if (local_membership.gene_index < 0).any():
        raise RuntimeError("K=6 module genes do not align with three-state genes")
    module_ids = sorted(local_membership.spectral_module.unique())
    indices = {
        module: local_membership.loc[
            local_membership.spectral_module.eq(module), "gene_index"
        ].to_numpy(int)
        for module in module_ids
    }

    manifest = pd.read_csv(common.MATCHED_MANIFEST, sep="\t")
    rng = np.random.default_rng(SEED)
    frequency_records: list[dict[str, object]] = []
    matrices: dict[str, pd.DataFrame] = {}
    for group, flag in (("Cluster 2", True), ("Matched controls", False)):
        local = manifest.loc[manifest.cluster2.eq(flag)].sort_values("pair_id")
        states = np.asarray(patient_states[local.state_row.to_numpy(int)], np.uint8)
        for state_name, state_code, short_name in (
            ("Mutated-not-LOF", 1, "nonlof"), ("LOF", 2, "lof")
        ):
            densities = []
            for module in module_ids:
                selected = states[:, indices[module]]
                patient_fraction = np.mean(selected == state_code, axis=1)
                densities.append(patient_fraction)
                low, high = bootstrap_mean_ci(patient_fraction, rng)
                frequency_records.append({
                    "group": group,
                    "module": int(module),
                    "module_size": len(indices[module]),
                    "state": state_name,
                    "aggregate_frequency": float(patient_fraction.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "patients": len(states),
                })
            density_matrix = np.column_stack(densities)
            matrix = density_matrix.T @ density_matrix / len(density_matrix)
            labels = [f"M{x}" for x in module_ids]
            frame = pd.DataFrame(matrix, index=labels, columns=labels)
            matrices[f"{short_name}_{group}"] = frame
            frame.to_csv(
                MUTATION_OUTPUT /
                f"{group.lower().replace(' ', '_')}_module_size_normalized_{short_name}_cooccurrence.tsv",
                sep="\t",
            )
    frequencies = pd.DataFrame(frequency_records)
    frequencies.to_csv(MUTATION_OUTPUT / "module_state_aggregate_frequencies.tsv",
                       sep="\t", index=False)

    payload = np.load(RAW_IG, allow_pickle=True)
    ig_genes = payload["genes"].astype(str)
    attributions = payload["attributions"].astype(float)
    cluster2 = payload["cluster2"].astype(bool)
    ig_index = {gene: index for index, gene in enumerate(ig_genes)}
    ig_rng = np.random.default_rng(SEED + 1)
    ig_records: list[dict[str, object]] = []
    for module in module_ids:
        columns = np.asarray([ig_index[gene] for gene in
                              local_membership.loc[
                                  local_membership.spectral_module.eq(module), "gene"
                              ].astype(str)])
        patient_mean = attributions[:, columns].mean(axis=1)
        for group, mask in (("Cluster 2", cluster2),
                            ("Matched controls", ~cluster2)):
            values = patient_mean[mask]
            low, high = bootstrap_mean_ci(values, ig_rng)
            ig_records.append({
                "module": int(module),
                "module_size": len(columns),
                "group": group,
                "patients": int(mask.sum()),
                "mean_attribution_per_gene": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
            })
    ig_summary = pd.DataFrame(ig_records)
    ig_summary.to_csv(IG_OUTPUT / "module_integrated_gradient_summary.tsv",
                      sep="\t", index=False)

    # The generic helper uses the same matched patients and computes paired
    # patient-bootstrap intervals for any number of modules.
    pair_intervals = common.pairwise_bootstrap_intervals(local_membership)
    return frequencies, ig_summary, matrices, pair_intervals


def module_heatmap(
    axis: plt.Axes, matrix: np.ndarray, genes: np.ndarray,
    module: int, vlim: float, letter: str,
) -> None:
    axis.imshow(matrix, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                interpolation="nearest", aspect="equal", rasterized=True)
    ticks = np.arange(len(genes))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    bbox = axis.get_window_extent().transformed(
        axis.figure.dpi_scale_trans.inverted()
    )
    spacing_points = min(bbox.width, bbox.height) * 72.0 / max(len(genes), 1)
    label_size = float(np.clip(spacing_points * 0.86, 8.0, 16.0))
    axis.set_xticklabels(genes, rotation=90, ha="center", va="top",
                         fontsize=label_size, fontweight="normal")
    axis.set_yticklabels(genes, fontsize=label_size, fontweight="normal")
    axis.tick_params(axis="both", length=0, pad=1.0)
    axis.set_title(
        f"{letter}  M{module}  (n={len(genes)})\n{MODULE_THEMES[module]}",
        fontsize=17.5, fontweight="semibold", color=MODULE_COLORS[module - 1],
        pad=10, linespacing=1.08,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPRODUCED.mkdir(parents=True, exist_ok=True)
    attention = pd.read_csv(ATTENTION, sep="\t", index_col=0)
    membership = pd.read_csv(MEMBERSHIP, sep="\t").sort_values("display_order")
    frequencies, ig_summary, matrices, pair_intervals = calculate_summaries(membership)

    genes = attention.index.astype(str).to_numpy()
    if not np.array_equal(genes, membership.gene.astype(str).to_numpy()):
        raise RuntimeError("K=6 attention and membership orders differ")
    modules = membership.spectral_module.to_numpy(int)
    module_ids = sorted(np.unique(modules))
    matrix = attention.to_numpy(float)
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    vlim = float(np.quantile(np.abs(off_diagonal), 0.995))

    # The upper region reproduces the finalized full-matrix + 4-over-2 inset
    # layout.  The lower region uses two balanced rows for the summary panels,
    # preventing the bar charts from becoming vertically elongated.
    fig = plt.figure(figsize=(48, 43), facecolor="white")
    master = fig.add_gridspec(
        2, 1, height_ratios=[1.62, 1.0],
        left=0.018, right=0.993, bottom=0.035, top=0.975,
        hspace=0.10,
    )
    heatmap_grid = master[0].subgridspec(
        1, 2, width_ratios=[0.72, 2.28], wspace=0.075,
    )

    full_axis = fig.add_subplot(heatmap_grid[0, 0])
    divider = make_axes_locatable(full_axis)
    strip = divider.append_axes("top", size="2.8%", pad=0.08, sharex=full_axis)
    color_axis = divider.append_axes("bottom", size="4.0%", pad=0.56)
    image = full_axis.imshow(matrix, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                             interpolation="nearest", aspect="equal",
                             rasterized=True)
    strip.imshow(modules[None, :], aspect="auto", interpolation="nearest",
                 cmap=ListedColormap(MODULE_COLORS), vmin=1, vmax=6,
                 extent=(-0.5, len(modules) - 0.5, 0, 1), origin="lower")
    strip.set_axis_off()
    for module in module_ids:
        positions = np.flatnonzero(modules == module)
        start = float(positions.min()) - 0.5
        size = float(len(positions))
        full_axis.add_patch(Rectangle(
            (start, start), size, size, fill=False,
            edgecolor=MODULE_COLORS[module - 1], linewidth=2.4, zorder=5,
        ))
        strip.text(positions.mean(), 1.18, f"M{module}", ha="center",
                   va="bottom", fontsize=15, fontweight="semibold",
                   clip_on=False)
    full_axis.set_xticks([])
    full_axis.set_yticks([])
    full_axis.set_xlabel("Key gene", fontsize=18, labelpad=12)
    full_axis.set_ylabel("Query gene", fontsize=18, labelpad=12)
    full_axis.set_title("A  Full module-ordered attention matrix", loc="left",
                        fontsize=21, fontweight="semibold", pad=50)
    colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        "Bidirectionally averaged attention: log$_2$(observed / expected)",
        fontsize=15, labelpad=7,
    )
    colorbar.ax.tick_params(labelsize=13, pad=2)

    inset_grid = heatmap_grid[0, 1].subgridspec(
        2, 1, height_ratios=[0.39, 0.61], hspace=0.075,
    )
    inset_top = inset_grid[0].subgridspec(1, 4, wspace=0.28)
    inset_bottom = inset_grid[1].subgridspec(1, 2, wspace=0.19)
    module_axes = {
        1: fig.add_subplot(inset_top[0, 0]),
        3: fig.add_subplot(inset_top[0, 1]),
        4: fig.add_subplot(inset_top[0, 2]),
        6: fig.add_subplot(inset_top[0, 3]),
        2: fig.add_subplot(inset_bottom[0, 0]),
        5: fig.add_subplot(inset_bottom[0, 1]),
    }
    letters = {1: "B", 3: "C", 4: "D", 6: "E", 2: "F", 5: "G"}
    fig.canvas.draw()
    for module, axis in module_axes.items():
        selected = np.flatnonzero(modules == module)
        module_heatmap(axis, matrix[np.ix_(selected, selected)], genes[selected],
                       module, vlim, letters[module])

    positions = np.arange(len(module_ids), dtype=float)
    offsets = {"Cluster 2": -0.19, "Matched controls": 0.19}
    handles = [Patch(facecolor=common.GROUP_COLORS[group], label=group)
               for group in offsets]

    summary_grid = master[1].subgridspec(
        2, 24, height_ratios=[1.0, 1.0], wspace=0.95, hspace=0.46,
    )

    ig_axis = fig.add_subplot(summary_grid[0, 0:10])
    for group, offset in offsets.items():
        local = ig_summary.loc[ig_summary.group.eq(group)].set_index("module").loc[module_ids]
        mean = local.mean_attribution_per_gene.to_numpy()
        low = local.ci_low.to_numpy()
        high = local.ci_high.to_numpy()
        ig_axis.bar(positions + offset, mean, width=0.38,
                    color=common.GROUP_COLORS[group], label=group,
                    yerr=np.vstack([mean - low, high - mean]), capsize=3,
                    edgecolor="white", linewidth=0.5,
                    error_kw={"elinewidth": 1.1, "ecolor": "#333333"})
    ig_axis.axhline(0, color="#333333", linewidth=1)
    ig_axis.set_xticks(positions, [f"M{x}" for x in module_ids])
    ig_axis.set_xlabel("Attention module", fontsize=15)
    ig_axis.set_ylabel("Mean integrated-gradient attribution per gene", fontsize=15)
    ig_axis.set_title("H  Integrated-gradient attribution across K=6 attention modules",
                      loc="left", fontsize=18, fontweight="semibold", pad=12)
    common.style_bar_axis(ig_axis)
    ig_axis.tick_params(labelsize=13.5)
    ig_axis.legend(handles=handles, frameon=False, loc="upper right",
                   fontsize=13, ncol=2)

    nonlof_axis = fig.add_subplot(summary_grid[0, 10:17])
    lof_axis = fig.add_subplot(summary_grid[0, 17:24])
    common.grouped_frequency_bars(
        nonlof_axis, frequencies, "Mutated-not-LOF",
        "Module-level non-LOF mutation frequency", "I",
    )
    common.grouped_frequency_bars(
        lof_axis, frequencies, "LOF", "Module-level LOF mutation frequency", "J",
    )
    for axis in (nonlof_axis, lof_axis):
        axis.tick_params(labelsize=13.5)
        axis.xaxis.label.set_size(15)
        axis.yaxis.label.set_size(15)
        axis.title.set_size(18)
        axis.title.set_fontweight("semibold")
    nonlof_axis.legend(
        handles=handles, frameon=False, loc="lower right",
        bbox_to_anchor=(1.0, 1.015), fontsize=12.5, ncol=2,
        borderaxespad=0.0, columnspacing=1.2, handletextpad=0.5,
    )
    lof_axis.legend(handles=handles, frameon=False, loc="upper right", fontsize=12.5)

    nonlof_pair_axis = fig.add_subplot(summary_grid[1, 0:12])
    lof_pair_axis = fig.add_subplot(summary_grid[1, 12:24])
    common.pairwise_comutation_bars(
        nonlof_pair_axis,
        matrices["nonlof_Cluster 2"], matrices["nonlof_Matched controls"],
        "Pairwise module non-LOF co-mutation frequency", "K",
        pair_intervals["nonlof"],
    )
    common.pairwise_comutation_bars(
        lof_pair_axis,
        matrices["lof_Cluster 2"], matrices["lof_Matched controls"],
        "Pairwise module LOF co-mutation frequency", "L",
        pair_intervals["lof"],
    )
    for axis in (nonlof_pair_axis, lof_pair_axis):
        axis.tick_params(labelsize=12.5)
        axis.xaxis.label.set_size(15)
        axis.yaxis.label.set_size(15)
        axis.title.set_size(18)
        axis.title.set_fontweight("semibold")
    nonlof_pair_axis.legend(
        handles=handles, frameon=False, loc="lower right",
        bbox_to_anchor=(1.0, 1.015), fontsize=13, ncol=2,
        borderaxespad=0.0, columnspacing=1.2, handletextpad=0.5,
    )
    lof_pair_axis.legend(handles=handles, frameon=False, loc="upper right",
                         fontsize=13, ncol=2)

    for extension in ("pdf", "png", "svg"):
        output_path = OUTPUT / f"{STEM}.{extension}"
        reproduced_path = REPRODUCED / f"{STEM}.{extension}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        shutil.copy2(output_path, reproduced_path)
    plt.close(fig)


if __name__ == "__main__":
    main()

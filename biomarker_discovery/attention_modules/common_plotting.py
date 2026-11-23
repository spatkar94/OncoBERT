#!/usr/bin/env python3
"""Assemble the Cluster-2 K=5 attention-module interpretability composite."""

from __future__ import annotations

from pathlib import Path
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
BASE = ROOT / "artifacts/cluster2_attention_modules/spectral_k5"
ATTENTION = BASE / "spectral_ordered_attention.tsv.gz"
MEMBERSHIP = BASE / "spectral_module_membership.tsv"
FREQUENCIES = BASE / "mutation_patterns/module_state_aggregate_frequencies.tsv"
IG = BASE / "integrated_gradients/module_integrated_gradient_summary.tsv"
MUTATION_DIR = BASE / "mutation_patterns"
MATCHED_MANIFEST = MUTATION_DIR / "matched_patient_manifest.tsv.gz"
SOURCE_RUN = (Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments")) /
              "experiments/candidate_three_state_logofunc_shared200_v20")
OUTPUT = BASE / "composite"
REPRODUCED = ROOT / "reproduced_figures"
STEM = "cluster2_k5_attention_interpretability_composite"

MODULE_COLORS = ["#4E79A7", "#59A14F", "#9C755F", "#76B7B2", "#B07AA1"]
MODULE_THEMES = {
    1: "Receptor tyrosine kinase–PI3K/AKT signaling",
    2: "Chromatin remodeling and\ngenome-maintenance programs",
    3: "Lineage transcription and WNT/TGF-β\ntumor-suppressor control",
    4: "Cellular senescence, G1/S checkpoint,\nand TGF-β signaling",
    5: "RAS–RAF–MAPK and receptor signaling",
}
GROUP_COLORS = {"Cluster 2": "#D55E00", "Matched controls": "#7F7F7F"}
PAIR_BOOTSTRAPS = 2000
PAIR_BOOTSTRAP_SEED = 4207


def style_bar_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", alpha=0.20, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=10)


def grouped_frequency_bars(
    axis: plt.Axes,
    data: pd.DataFrame,
    state: str,
    title: str,
    letter: str,
) -> None:
    modules = sorted(data.module.unique())
    positions = np.arange(len(modules), dtype=float)
    offsets = {"Cluster 2": -0.19, "Matched controls": 0.19}
    for group, offset in offsets.items():
        local = (
            data.loc[data.state.eq(state) & data.group.eq(group)]
            .set_index("module").loc[modules]
        )
        mean = 100 * local.aggregate_frequency.to_numpy()
        low = 100 * local.ci_low.to_numpy()
        high = 100 * local.ci_high.to_numpy()
        axis.bar(
            positions + offset,
            mean,
            width=0.38,
            color=GROUP_COLORS[group],
            label=group,
            yerr=np.vstack([mean - low, high - mean]),
            capsize=2.8,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
        )
    axis.set_xticks(positions, [f"M{x}" for x in modules])
    axis.set_xlabel("Attention module", fontsize=11)
    axis.set_ylabel(f"{state.replace('Mutated-not-LOF', 'Non-LOF')} mutation frequency (%)",
                    fontsize=11)
    axis.set_title(f"{letter}  {title}", loc="left", fontsize=14,
                   fontweight="semibold", pad=8)
    style_bar_axis(axis)


def pairwise_comutation_bars(
    axis: plt.Axes,
    cluster_matrix: pd.DataFrame,
    control_matrix: pd.DataFrame,
    title: str,
    letter: str,
    intervals: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Compare size-normalized co-mutation for each distinct module pair."""
    labels = list(cluster_matrix.index.astype(str))
    pairs = [(row, column) for row in range(len(labels))
             for column in range(row + 1, len(labels))]
    pair_labels = [f"{labels[row]}–{labels[column]}" for row, column in pairs]
    cluster_values = 100 * np.asarray([
        cluster_matrix.iloc[row, column] for row, column in pairs
    ], dtype=float)
    control_values = 100 * np.asarray([
        control_matrix.iloc[row, column] for row, column in pairs
    ], dtype=float)
    cluster_low, cluster_high = intervals["Cluster 2"]
    control_low, control_high = intervals["Matched controls"]
    cluster_low, cluster_high = 100 * cluster_low, 100 * cluster_high
    control_low, control_high = 100 * control_low, 100 * control_high
    positions = np.arange(len(pairs), dtype=float)
    axis.bar(
        positions - 0.19, cluster_values, width=0.38,
        color=GROUP_COLORS["Cluster 2"], label="Cluster 2",
        edgecolor="white", linewidth=0.5,
        yerr=np.vstack([cluster_values - cluster_low,
                        cluster_high - cluster_values]),
        capsize=2.8,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
    )
    axis.bar(
        positions + 0.19, control_values, width=0.38,
        color=GROUP_COLORS["Matched controls"], label="Matched controls",
        edgecolor="white", linewidth=0.5,
        yerr=np.vstack([control_values - control_low,
                        control_high - control_values]),
        capsize=2.8,
        error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
    )
    axis.set_xticks(positions, pair_labels, rotation=30, ha="right")
    axis.set_xlabel("Module pair", fontsize=11)
    axis.set_ylabel("Co-mutation frequency (%)", fontsize=11)
    axis.set_title(f"{letter}  {title}", loc="left", fontsize=14,
                   fontweight="semibold", pad=8)
    style_bar_axis(axis)


def pairwise_bootstrap_intervals(
    membership: pd.DataFrame,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Paired patient-bootstrap CIs for non-LOF and LOF module pairs."""
    with np.load(SOURCE_RUN / "profiles.npz", allow_pickle=False) as archive:
        state_genes = archive["genes"].astype(str)
        sample_patients = archive["patients"].astype(str)
    sample_states = np.load(SOURCE_RUN / "functional_states.npy", mmap_mode="r")
    order = np.argsort(sample_patients, kind="stable")
    sorted_patients = sample_patients[order]
    boundaries = np.r_[
        0, np.flatnonzero(sorted_patients[1:] != sorted_patients[:-1]) + 1
    ]
    patient_states = np.maximum.reduceat(sample_states[order], boundaries, axis=0)

    gene_index = pd.Index(state_genes)
    local_membership = membership.copy()
    local_membership["gene_index"] = gene_index.get_indexer(
        local_membership.gene.astype(str)
    )
    if (local_membership.gene_index < 0).any():
        raise RuntimeError("Module genes do not align with three-state input genes")
    module_ids = sorted(local_membership.spectral_module.unique())
    indices = [
        local_membership.loc[
            local_membership.spectral_module.eq(module), "gene_index"
        ].to_numpy(int)
        for module in module_ids
    ]
    sizes = np.asarray([len(value) for value in indices], dtype=float)
    pairs = [(row, column) for row in range(len(module_ids))
             for column in range(row + 1, len(module_ids))]

    manifest = pd.read_csv(MATCHED_MANIFEST, sep="\t")
    values: dict[tuple[str, str], np.ndarray] = {}
    pair_ids: dict[str, np.ndarray] = {}
    for group, flag in (("Cluster 2", True), ("Matched controls", False)):
        local = manifest.loc[manifest.cluster2.eq(flag)].sort_values("pair_id")
        pair_ids[group] = local.pair_id.to_numpy()
        states = np.asarray(patient_states[local.state_row.to_numpy(int)], np.uint8)
        for state_name, state_code in (("nonlof", 1), ("lof", 2)):
            counts = np.column_stack([
                np.sum(states[:, selected] == state_code, axis=1)
                for selected in indices
            ])
            densities = counts / sizes[None, :]
            values[(state_name, group)] = np.column_stack([
                densities[:, row] * densities[:, column]
                for row, column in pairs
            ])
    if not np.array_equal(pair_ids["Cluster 2"], pair_ids["Matched controls"]):
        raise RuntimeError("Matched Cluster 2 and control pair IDs are misaligned")

    rng = np.random.default_rng(PAIR_BOOTSTRAP_SEED)
    n_patients = len(pair_ids["Cluster 2"])
    replicate_means: dict[tuple[str, str], list[np.ndarray]] = {
        key: [] for key in values
    }
    batch_size = 50
    for start in range(0, PAIR_BOOTSTRAPS, batch_size):
        count = min(batch_size, PAIR_BOOTSTRAPS - start)
        # Resample matched-pair positions so both groups use identical draws.
        draws = rng.integers(0, n_patients, size=(count, n_patients))
        for key, patient_values in values.items():
            replicate_means[key].append(patient_values[draws].mean(axis=1))

    result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "nonlof": {}, "lof": {}
    }
    for (state_name, group), chunks in replicate_means.items():
        replicates = np.concatenate(chunks, axis=0)
        low, high = np.quantile(replicates, [0.025, 0.975], axis=0)
        result[state_name][group] = (low, high)
    return result


def module_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    genes: np.ndarray,
    module: int,
    vlim: float,
    letter: str,
) -> None:
    n_genes = len(genes)
    axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    ticks = np.arange(n_genes)
    # A common point size is used on every inset so that visual prominence is
    # not determined by module size. The two larger panels receive more space.
    label_size = 6.8
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(genes, rotation=90, ha="center", va="top",
                         fontsize=label_size, fontweight="normal")
    axis.set_yticklabels(genes, fontsize=label_size, fontweight="normal")
    axis.tick_params(axis="both", length=0, pad=1.0)
    axis.set_title(
        f"{letter}  M{module}  (n={n_genes})\n{MODULE_THEMES[module]}",
        fontsize=13.5,
        fontweight="semibold",
        color=MODULE_COLORS[module - 1],
        pad=8,
        linespacing=1.15,
    )
    for spine in axis.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPRODUCED.mkdir(parents=True, exist_ok=True)

    attention = pd.read_csv(ATTENTION, sep="\t", index_col=0)
    membership = pd.read_csv(MEMBERSHIP, sep="\t").sort_values("display_order")
    frequencies = pd.read_csv(FREQUENCIES, sep="\t")
    integrated_gradients = pd.read_csv(IG, sep="\t")
    nonlof_cluster = pd.read_csv(
        MUTATION_DIR / "cluster_2_module_size_normalized_nonlof_cooccurrence.tsv",
        sep="\t", index_col=0,
    )
    nonlof_control = pd.read_csv(
        MUTATION_DIR / "matched_controls_module_size_normalized_nonlof_cooccurrence.tsv",
        sep="\t", index_col=0,
    )
    lof_cluster = pd.read_csv(
        MUTATION_DIR / "cluster_2_module_size_normalized_lof_cooccurrence.tsv",
        sep="\t", index_col=0,
    )
    lof_control = pd.read_csv(
        MUTATION_DIR / "matched_controls_module_size_normalized_lof_cooccurrence.tsv",
        sep="\t", index_col=0,
    )
    pair_intervals = pairwise_bootstrap_intervals(membership)

    genes = attention.index.astype(str).to_numpy()
    expected = membership.gene.astype(str).to_numpy()
    if not np.array_equal(genes, expected):
        raise RuntimeError("Attention matrix and module membership orders differ")
    modules = membership.spectral_module.to_numpy(int)
    module_ids = sorted(np.unique(modules))
    matrix = attention.to_numpy(float)
    off_diagonal = matrix[~np.eye(len(matrix), dtype=bool)]
    vlim = float(np.quantile(np.abs(off_diagonal), 0.995))

    fig = plt.figure(figsize=(30, 27), facecolor="white")
    grid = fig.add_gridspec(
        4,
        20,
        height_ratios=[1.00, 1.22, 0.70, 0.63],
        left=0.025,
        right=0.985,
        bottom=0.055,
        top=0.965,
        wspace=1.55,
        hspace=0.40,
    )

    # Panel A: full module-ordered 200 x 200 attention matrix. It spans both
    # inset rows and is vertically centered within the available region.
    full_axis = fig.add_subplot(grid[0:2, 0:5])
    divider = make_axes_locatable(full_axis)
    module_strip = divider.append_axes("top", size="2.8%", pad=0.08,
                                      sharex=full_axis)
    color_axis = divider.append_axes("bottom", size="4.0%", pad=0.56)
    full_image = full_axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    module_strip.imshow(
        modules[None, :],
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(MODULE_COLORS),
        vmin=1,
        vmax=5,
        extent=(-0.5, len(modules) - 0.5, 0, 1),
        origin="lower",
    )
    module_strip.set_axis_off()
    for module in module_ids:
        positions = np.flatnonzero(modules == module)
        start = float(positions.min()) - 0.5
        size = float(len(positions))
        full_axis.add_patch(Rectangle(
            (start, start), size, size, fill=False,
            edgecolor=MODULE_COLORS[module - 1], linewidth=2.4, zorder=5,
        ))
        module_strip.text(
            positions.mean(), 1.18, f"M{module}", ha="center", va="bottom",
            fontsize=11, fontweight="semibold", clip_on=False,
        )
    full_axis.set_xticks([])
    full_axis.set_yticks([])
    full_axis.set_xlabel("Key gene", fontsize=13, labelpad=10)
    full_axis.set_ylabel("Query gene", fontsize=13, labelpad=10)
    full_axis.set_title(
        "A  Full module-ordered attention matrix",
        loc="left", fontsize=15, fontweight="semibold", pad=42,
    )
    colorbar = fig.colorbar(full_image, cax=color_axis, orientation="horizontal")
    colorbar.set_label(
        "Bidirectionally averaged attention: log$_2$(observed / expected)",
        fontsize=10.5, labelpad=5,
    )
    colorbar.ax.tick_params(labelsize=9, pad=2)

    # Panels B-F: within-module zooms in the requested 3-over-2 layout.
    module_axes = {
        1: fig.add_subplot(grid[0, 5:10]),
        3: fig.add_subplot(grid[0, 10:14]),
        5: fig.add_subplot(grid[0, 14:20]),
        2: fig.add_subplot(grid[1, 5:12]),
        4: fig.add_subplot(grid[1, 12:20]),
    }
    panel_letters = {1: "B", 3: "C", 5: "D", 2: "E", 4: "F"}
    for module, axis in module_axes.items():
        indices = np.flatnonzero(modules == module)
        module_heatmap(
            axis,
            matrix[np.ix_(indices, indices)],
            genes[indices],
            module,
            vlim,
            panel_letters[module],
        )

    # Panel G: matched-patient integrated-gradient attribution by module.
    ig_axis = fig.add_subplot(grid[2, 0:9])
    positions = np.arange(len(module_ids), dtype=float)
    offsets = {"Cluster 2": -0.19, "Matched controls": 0.19}
    for group, offset in offsets.items():
        local = (integrated_gradients.loc[integrated_gradients.group.eq(group)]
                 .set_index("module").loc[module_ids])
        mean = local.mean_attribution_per_gene.to_numpy()
        low = local.ci_low.to_numpy()
        high = local.ci_high.to_numpy()
        ig_axis.bar(
            positions + offset,
            mean,
            width=0.38,
            color=GROUP_COLORS[group],
            label=group,
            yerr=np.vstack([mean - low, high - mean]),
            capsize=3,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"elinewidth": 1.1, "ecolor": "#333333"},
        )
    ig_axis.axhline(0, color="#333333", linewidth=1)
    ig_axis.set_xticks(positions, [f"M{x}" for x in module_ids])
    ig_axis.set_xlabel("Attention module", fontsize=11)
    ig_axis.set_ylabel("Mean integrated-gradient attribution per gene", fontsize=11)
    ig_axis.set_title(
        "G  Integrated-gradient attribution across K=5 attention modules",
        loc="left", fontsize=14, fontweight="semibold", pad=8,
    )
    style_bar_axis(ig_axis)

    # Panels H-I: state-specific, module-size-normalized mutation frequencies.
    nonlof_axis = fig.add_subplot(grid[2, 9:14])
    lof_axis = fig.add_subplot(grid[2, 14:20])
    grouped_frequency_bars(
        nonlof_axis, frequencies, "Mutated-not-LOF",
        "Module-level non-LOF mutation frequency", "H",
    )
    grouped_frequency_bars(
        lof_axis, frequencies, "LOF",
        "Module-level LOF mutation frequency", "I",
    )

    legend_handles = [
        Patch(facecolor=GROUP_COLORS[group], label=group)
        for group in ("Cluster 2", "Matched controls")
    ]
    ig_axis.legend(handles=legend_handles, frameon=False, loc="upper right",
                   fontsize=10, ncol=2)
    nonlof_axis.legend(handles=legend_handles, frameon=False, loc="upper right",
                       fontsize=9)
    lof_axis.legend(handles=legend_handles, frameon=False, loc="upper right",
                    fontsize=9)

    # Panels J-K: all ten distinct module pairs. The matrices are normalized
    # by module size, so a value is the probability that one randomly selected
    # gene from each module is co-mutated in the same patient. Diagonal
    # within-module entries are intentionally omitted.
    nonlof_pair_axis = fig.add_subplot(grid[3, 0:10])
    lof_pair_axis = fig.add_subplot(grid[3, 10:20])
    pairwise_comutation_bars(
        nonlof_pair_axis,
        nonlof_cluster,
        nonlof_control,
        "Pairwise module non-LOF co-mutation frequency",
        "J",
        pair_intervals["nonlof"],
    )
    pairwise_comutation_bars(
        lof_pair_axis,
        lof_cluster,
        lof_control,
        "Pairwise module LOF co-mutation frequency",
        "K",
        pair_intervals["lof"],
    )
    nonlof_pair_axis.legend(handles=legend_handles, frameon=False,
                            loc="upper right", fontsize=10, ncol=2)
    lof_pair_axis.legend(handles=legend_handles, frameon=False,
                         loc="upper right", fontsize=10, ncol=2)

    for extension in ("pdf", "png", "svg"):
        output_path = OUTPUT / f"{STEM}.{extension}"
        reproduced_path = REPRODUCED / f"{STEM}.{extension}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        shutil.copy2(output_path, reproduced_path)
    plt.close(fig)


if __name__ == "__main__":
    main()

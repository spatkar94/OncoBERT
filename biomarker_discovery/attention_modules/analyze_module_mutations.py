#!/usr/bin/env python3
"""Mutation-state prevalence and co-occurrence for K=5 attention modules."""

from __future__ import annotations

import sys
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Wedge
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from mutation_pattern_utils import (  # noqa: E402
    coarsened_exact_match,
    load_patient_data,
)

MEMBERSHIP = (ROOT / "artifacts/cluster2_attention_modules/spectral_k5" /
              "spectral_module_membership.tsv")
OUTPUT = ROOT / "artifacts/cluster2_attention_modules/spectral_k5/mutation_patterns"
REPRODUCED = ROOT / "reproduced_figures"
STEM = "cluster2_k5_attention_modules_mutation_patterns"
SEED = 42
BOOTSTRAPS = 2000


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Patient-resampling 95% interval for a patient-level module fraction."""
    n = len(values)
    draws = rng.integers(0, n, size=(BOOTSTRAPS, n))
    means = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def cooccurrence(module_altered: np.ndarray) -> np.ndarray:
    """P(any alteration in module i AND any alteration in module j)."""
    binary = module_altered.astype(np.float64)
    return binary.T @ binary / len(binary)


def size_normalized_cooccurrence(
    altered_counts: np.ndarray,
    module_sizes: np.ndarray,
) -> np.ndarray:
    """Probability that a random cross-module gene pair is co-mutated.

    Off-diagonal entries average (k_i / m_i) * (k_j / m_j) over patients,
    where k is the number of altered genes and m is module size. Diagonal
    entries use distinct within-module gene pairs, k(k-1) / [m(m-1)].
    """
    densities = altered_counts / module_sizes[None, :]
    matrix = densities.T @ densities / len(densities)
    for column, size in enumerate(module_sizes):
        counts = altered_counts[:, column]
        matrix[column, column] = np.mean(
            counts * (counts - 1) / (size * (size - 1))
        )
    return matrix


def state_partitioned_cooccurrence(
    nonlof_counts: np.ndarray,
    lof_counts: np.ndarray,
    module_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition general co-mutation into non-LOF and LOF endpoints.

    Every co-mutated gene pair contributes two mutated endpoints. A pair with
    one LOF and one mutated-not-LOF endpoint contributes one half to each
    component. Consequently, the two component matrices sum exactly to the
    general co-mutation-frequency matrix.
    """
    mutated = nonlof_counts + lof_counts
    modules = mutated.shape[1]
    lof_component = np.zeros((modules, modules), dtype=float)
    nonlof_component = np.zeros((modules, modules), dtype=float)
    for i in range(modules):
        for j in range(modules):
            if i == j:
                denominator = module_sizes[i] * (module_sizes[i] - 1)
                lof_component[i, i] = np.mean(
                    lof_counts[:, i] * (mutated[:, i] - 1) / denominator
                )
                nonlof_component[i, i] = np.mean(
                    nonlof_counts[:, i] * (mutated[:, i] - 1) / denominator
                )
            else:
                denominator = 2 * module_sizes[i] * module_sizes[j]
                lof_component[i, j] = np.mean(
                    (lof_counts[:, i] * mutated[:, j] +
                     mutated[:, i] * lof_counts[:, j]) / denominator
                )
                nonlof_component[i, j] = np.mean(
                    (nonlof_counts[:, i] * mutated[:, j] +
                     mutated[:, i] * nonlof_counts[:, j]) / denominator
                )
    return nonlof_component + lof_component, nonlof_component, lof_component


def module_alteration_cooccurrence(
    nonlof_counts: np.ndarray,
    lof_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Module-level alteration and co-alteration frequencies by state.

    Diagonal entries are P(module is mutated). Off-diagonal entries are
    P(module i and module j are both mutated). Each altered patient contributes
    one unit, partitioned between mutation states according to the fraction of
    mutated genes in the implicated module(s) that are LOF versus non-LOF.
    The state components therefore sum exactly to the total frequency.
    """
    mutated = nonlof_counts + lof_counts
    altered = mutated > 0
    modules = mutated.shape[1]
    total = np.zeros((modules, modules), dtype=float)
    nonlof_component = np.zeros_like(total)
    lof_component = np.zeros_like(total)
    for i in range(modules):
        for j in range(modules):
            if i == j:
                eligible = altered[:, i]
                denominator = mutated[:, i]
                lof_share = np.divide(
                    lof_counts[:, i], denominator,
                    out=np.zeros(len(mutated), dtype=float),
                    where=denominator > 0,
                )
            else:
                eligible = altered[:, i] & altered[:, j]
                denominator = mutated[:, i] + mutated[:, j]
                lof_share = np.divide(
                    lof_counts[:, i] + lof_counts[:, j], denominator,
                    out=np.zeros(len(mutated), dtype=float),
                    where=denominator > 0,
                )
            total[i, j] = np.mean(eligible)
            lof_component[i, j] = np.mean(eligible * lof_share)
            nonlof_component[i, j] = np.mean(eligible * (1.0 - lof_share))
    return total, nonlof_component, lof_component


def pie_matrix(
    axis: plt.Axes,
    total: np.ndarray,
    nonlof: np.ndarray,
    lof: np.ndarray,
    module_labels: list[str],
    size_max: float,
    title: str,
    *,
    edgecolors: np.ndarray | None = None,
) -> None:
    """Draw pies whose area represents module mutation/co-mutation frequency."""
    modules = len(module_labels)
    axis.set_xlim(-0.5, modules - 0.5)
    axis.set_ylim(modules - 0.5, -0.5)
    axis.set_aspect("equal")
    for row in range(modules):
        for column in range(modules):
            value = float(abs(total[row, column]))
            if value <= 0 or size_max <= 0:
                continue
            radius = 0.44 * np.sqrt(value / size_max)
            components = np.asarray([max(nonlof[row, column], 0),
                                     max(lof[row, column], 0)], float)
            if components.sum() == 0:
                components[:] = (1, 0)
            fractions = components / components.sum()
            start = 90.0
            for fraction, color in zip(fractions, ("#E69F00", "#D55E00")):
                end = start + 360.0 * fraction
                axis.add_patch(Wedge((column, row), radius, start, end,
                                     facecolor=color, edgecolor="white",
                                     linewidth=0.45))
                start = end
            outline = (edgecolors[row, column] if edgecolors is not None
                       else "#404040")
            axis.add_patch(Circle((column, row), radius, facecolor="none",
                                  edgecolor=outline, linewidth=0.8))
    axis.set_xticks(range(modules), module_labels)
    axis.set_yticks(range(modules), module_labels)
    axis.set_xlabel("Attention module")
    axis.set_ylabel("Attention module")
    axis.set_title(title, loc="left", fontweight="semibold", fontsize=12)
    axis.set_xticks(np.arange(-0.5, modules, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, modules, 1), minor=True)
    axis.grid(which="minor", color="#D9D9D9", linewidth=0.7)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.spines[:].set_visible(False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPRODUCED.mkdir(parents=True, exist_ok=True)

    genes, patient_states, frame = load_patient_data()
    matched = coarsened_exact_match(frame)
    membership = pd.read_csv(MEMBERSHIP, sep="\t")
    gene_index = pd.Index(genes.astype(str))
    membership["gene_index"] = gene_index.get_indexer(membership.gene.astype(str))
    if (membership.gene_index < 0).any() or len(membership) != len(genes):
        raise RuntimeError("Spectral membership and three-state genes do not align")

    module_ids = sorted(membership.spectral_module.unique())
    module_indices = {
        module: membership.loc[membership.spectral_module.eq(module),
                               "gene_index"].to_numpy(int)
        for module in module_ids
    }
    module_sizes = {module: len(module_indices[module]) for module in module_ids}

    groups = {
        "Cluster 2": matched.loc[matched.cluster2],
        "Matched controls": matched.loc[~matched.cluster2],
    }
    rng = np.random.default_rng(SEED)
    frequency_records: list[dict[str, object]] = []
    altered_by_group: dict[str, np.ndarray] = {}
    mutation_counts_by_group: dict[str, np.ndarray] = {}
    nonlof_counts_by_group: dict[str, np.ndarray] = {}
    lof_counts_by_group: dict[str, np.ndarray] = {}
    state_burden_by_group: dict[str, dict[int, np.ndarray]] = {}

    for group_name, local in groups.items():
        states = np.asarray(patient_states[local.state_row.to_numpy()], np.uint8)
        altered = np.zeros((len(states), len(module_ids)), dtype=bool)
        mutation_counts = np.zeros((len(states), len(module_ids)), dtype=np.int16)
        nonlof_counts = np.zeros((len(states), len(module_ids)), dtype=np.int16)
        lof_counts = np.zeros((len(states), len(module_ids)), dtype=np.int16)
        state_burden_by_group[group_name] = {}
        for column, module in enumerate(module_ids):
            selected = states[:, module_indices[module]]
            altered[:, column] = np.any(selected > 0, axis=1)
            # General mutation status collapses mutated-not-LOF and LOF.
            mutation_counts[:, column] = np.sum(selected > 0, axis=1)
            nonlof_counts[:, column] = np.sum(selected == 1, axis=1)
            lof_counts[:, column] = np.sum(selected == 2, axis=1)
            for state, state_name in ((1, "Mutated-not-LOF"), (2, "LOF")):
                patient_fraction = np.mean(selected == state, axis=1)
                low, high = bootstrap_mean_ci(patient_fraction, rng)
                frequency_records.append({
                    "group": group_name,
                    "module": module,
                    "module_size": module_sizes[module],
                    "state": state_name,
                    "aggregate_frequency": patient_fraction.mean(),
                    "ci_low": low,
                    "ci_high": high,
                    "patients": len(states),
                    "state_calls": int(np.sum(selected == state)),
                    "possible_patient_gene_entries": int(selected.size),
                })
        altered_by_group[group_name] = altered
        mutation_counts_by_group[group_name] = mutation_counts
        nonlof_counts_by_group[group_name] = nonlof_counts
        lof_counts_by_group[group_name] = lof_counts

    frequencies = pd.DataFrame(frequency_records)
    frequencies.to_csv(OUTPUT / "module_state_aggregate_frequencies.tsv",
                       sep="\t", index=False)
    matched[["pair_id", "patient_id", "cluster2", "cancer_type",
             "institution", "assay", "log1p_tmb", "state_row"]].to_csv(
        OUTPUT / "matched_patient_manifest.tsv.gz", sep="\t", index=False,
        compression="gzip")

    matrices: dict[str, np.ndarray] = {}
    for group_name, altered in altered_by_group.items():
        matrix = cooccurrence(altered)
        matrices[group_name] = matrix
        pd.DataFrame(matrix, index=[f"M{x}" for x in module_ids],
                     columns=[f"M{x}" for x in module_ids]).to_csv(
            OUTPUT / f"{group_name.lower().replace(' ', '_')}_module_cooccurrence.tsv",
            sep="\t")
    difference = matrices["Cluster 2"] - matrices["Matched controls"]
    pd.DataFrame(difference, index=[f"M{x}" for x in module_ids],
                 columns=[f"M{x}" for x in module_ids]).to_csv(
        OUTPUT / "cluster2_minus_control_module_cooccurrence.tsv", sep="\t")

    # Module-size-normalized LOF co-mutation. A diagonal entry is the
    # probability that two distinct genes sampled from the same module are
    # both LOF in a patient. An off-diagonal entry is the probability that
    # one gene sampled from each of two modules is LOF in the same patient.
    size_vector = np.asarray([module_sizes[m] for m in module_ids], float)
    lof_matrices: dict[str, np.ndarray] = {}
    for group_name, counts in lof_counts_by_group.items():
        matrix = size_normalized_cooccurrence(counts, size_vector)
        lof_matrices[group_name] = matrix
        pd.DataFrame(matrix, index=[f"M{x}" for x in module_ids],
                     columns=[f"M{x}" for x in module_ids]).to_csv(
            OUTPUT / (group_name.lower().replace(" ", "_") +
                      "_module_size_normalized_lof_cooccurrence.tsv"), sep="\t")

    nonlof_matrices: dict[str, np.ndarray] = {}
    for group_name, counts in nonlof_counts_by_group.items():
        matrix = size_normalized_cooccurrence(counts, size_vector)
        nonlof_matrices[group_name] = matrix
        pd.DataFrame(matrix, index=[f"M{x}" for x in module_ids],
                     columns=[f"M{x}" for x in module_ids]).to_csv(
            OUTPUT / (group_name.lower().replace(" ", "_") +
                      "_module_size_normalized_nonlof_cooccurrence.tsv"), sep="\t")

    # Three-panel LOF-only summary.
    fig = plt.figure(figsize=(15.2, 5.7), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12],
                            left=0.06, right=0.97, bottom=0.16, top=0.82,
                            wspace=0.40)
    bar_ax = fig.add_subplot(grid[0, 0])
    c2_ax = fig.add_subplot(grid[0, 1])
    control_ax = fig.add_subplot(grid[0, 2])

    x = np.arange(len(module_ids), dtype=float)
    offsets = {"Cluster 2": -0.19, "Matched controls": 0.19}
    colors = {"Cluster 2": "#D55E00", "Matched controls": "#7F7F7F"}
    for group_name, offset in offsets.items():
        local = (frequencies.loc[
            frequencies.group.eq(group_name) & frequencies.state.eq("LOF")
        ].set_index("module").loc[module_ids])
        mean = 100 * local.aggregate_frequency.to_numpy()
        low = 100 * local.ci_low.to_numpy()
        high = 100 * local.ci_high.to_numpy()
        bar_ax.bar(x + offset, mean, width=0.38, color=colors[group_name],
                   label=group_name, edgecolor="white", linewidth=0.5,
                   yerr=np.vstack([mean-low, high-mean]), capsize=2.5,
                   error_kw={"elinewidth": 1.0, "ecolor": "#333333"})
    bar_ax.set_xticks(x, [f"M{m}" for m in module_ids])
    bar_ax.set_xlabel("Module")
    bar_ax.set_ylabel("LOF mutation frequency (%)")
    bar_ax.set_title("A  Module-level LOF mutation frequency",
                     loc="left", fontweight="semibold")
    bar_ax.legend(frameon=False, loc="upper right")
    bar_ax.grid(axis="y", alpha=0.22)
    bar_ax.spines[["top", "right"]].set_visible(False)
    # Keep the bar panel visually comparable to the square heatmaps rather
    # than allowing it to stretch across the full available height.
    bar_ax.set_box_aspect(1)

    cluster2_matrix = lof_matrices["Cluster 2"]
    control_matrix = lof_matrices["Matched controls"]
    module_labels = [f"M{x}" for x in module_ids]
    shared_max = max(cluster2_matrix.max(), control_matrix.max())
    heatmap_kwargs = dict(
        cmap="YlOrRd", vmin=0, vmax=shared_max, square=True,
        annot=True, fmt=".2%", annot_kws={"fontsize": 9},
        linewidths=0.8, linecolor="white",
        xticklabels=module_labels, yticklabels=module_labels,
    )
    sns.heatmap(cluster2_matrix, ax=c2_ax, cbar=False, **heatmap_kwargs)
    sns.heatmap(
        control_matrix, ax=control_ax, cbar=True,
        cbar_kws={"label": "LOF co-mutation frequency", "shrink": 0.78},
        **heatmap_kwargs,
    )
    for axis, title in ((c2_ax, "B  Cluster 2"),
                        (control_ax, "C  Matched controls")):
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("Module")
        axis.set_ylabel("Module")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "LOF mutation patterns across K=5 spectral attention modules",
        fontsize=18, fontweight="semibold", y=0.99)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{STEM}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(REPRODUCED / f"{STEM}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Separate differential heatmap. A half-minimum-positive pseudocount keeps
    # zero-frequency cells finite while remaining below every observed value.
    lof_positive = np.concatenate([
        lof_matrices["Cluster 2"][lof_matrices["Cluster 2"] > 0],
        lof_matrices["Matched controls"][lof_matrices["Matched controls"] > 0],
    ])
    lof_pseudocount = 0.5 * float(lof_positive.min())
    lof_log2fc = np.log2(
        (lof_matrices["Cluster 2"] + lof_pseudocount) /
        (lof_matrices["Matched controls"] + lof_pseudocount)
    )
    pd.DataFrame(lof_log2fc, index=module_labels, columns=module_labels).to_csv(
        OUTPUT / "cluster2_vs_control_module_lof_cooccurrence_log2fc.tsv",
        sep="\t",
    )
    pd.DataFrame({
        "quantity": ["pseudocount", "log_base"],
        "value": [lof_pseudocount, 2],
    }).to_csv(
        OUTPUT / "cluster2_vs_control_module_lof_cooccurrence_log2fc_metadata.tsv",
        sep="\t", index=False,
    )

    differential_stem = "cluster2_k5_attention_modules_lof_cooccurrence_log2fc"
    symmetric_limit = float(np.max(np.abs(lof_log2fc)))
    fig, axis = plt.subplots(figsize=(7.4, 6.5), facecolor="white")
    sns.heatmap(
        lof_log2fc, ax=axis, cmap="RdBu_r", center=0,
        vmin=-symmetric_limit, vmax=symmetric_limit, square=True,
        annot=True, fmt=".2f", annot_kws={"fontsize": 10},
        linewidths=0.8, linecolor="white",
        xticklabels=module_labels, yticklabels=module_labels,
        cbar_kws={"label": "LOF co-mutation log2 fold change", "shrink": 0.82},
    )
    axis.set_xlabel("Module")
    axis.set_ylabel("Module")
    axis.set_title(
        "LOF co-mutation enrichment in Cluster 2 versus controls",
        fontsize=16, fontweight="semibold", pad=14,
    )
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{differential_stem}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(REPRODUCED / f"{differential_stem}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Directly comparable non-LOF version of the three-panel summary.
    nonlof_stem = "cluster2_k5_attention_modules_nonlof_mutation_patterns"
    fig = plt.figure(figsize=(15.2, 5.7), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12],
                            left=0.06, right=0.97, bottom=0.16, top=0.82,
                            wspace=0.40)
    bar_ax = fig.add_subplot(grid[0, 0])
    c2_ax = fig.add_subplot(grid[0, 1])
    control_ax = fig.add_subplot(grid[0, 2])

    x = np.arange(len(module_ids), dtype=float)
    offsets = {"Cluster 2": -0.19, "Matched controls": 0.19}
    colors = {"Cluster 2": "#E69F00", "Matched controls": "#7F7F7F"}
    for group_name, offset in offsets.items():
        local = (frequencies.loc[
            frequencies.group.eq(group_name) &
            frequencies.state.eq("Mutated-not-LOF")
        ].set_index("module").loc[module_ids])
        mean = 100 * local.aggregate_frequency.to_numpy()
        low = 100 * local.ci_low.to_numpy()
        high = 100 * local.ci_high.to_numpy()
        bar_ax.bar(x + offset, mean, width=0.38, color=colors[group_name],
                   label=group_name, edgecolor="white", linewidth=0.5,
                   yerr=np.vstack([mean-low, high-mean]), capsize=2.5,
                   error_kw={"elinewidth": 1.0, "ecolor": "#333333"})
    bar_ax.set_xticks(x, [f"M{m}" for m in module_ids])
    bar_ax.set_xlabel("Module")
    bar_ax.set_ylabel("Non-LOF mutation frequency (%)")
    bar_ax.set_title("A  Module-level non-LOF mutation frequency",
                     loc="left", fontweight="semibold")
    bar_ax.legend(frameon=False, loc="upper right")
    bar_ax.grid(axis="y", alpha=0.22)
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.set_box_aspect(1)

    cluster2_matrix = nonlof_matrices["Cluster 2"]
    control_matrix = nonlof_matrices["Matched controls"]
    module_labels = [f"M{x}" for x in module_ids]
    shared_max = max(cluster2_matrix.max(), control_matrix.max())
    heatmap_kwargs = dict(
        cmap="YlGnBu", vmin=0, vmax=shared_max, square=True,
        annot=True, fmt=".2%", annot_kws={"fontsize": 9},
        linewidths=0.8, linecolor="white",
        xticklabels=module_labels, yticklabels=module_labels,
    )
    sns.heatmap(cluster2_matrix, ax=c2_ax, cbar=False, **heatmap_kwargs)
    sns.heatmap(
        control_matrix, ax=control_ax, cbar=True,
        cbar_kws={"label": "Non-LOF co-mutation frequency", "shrink": 0.78},
        **heatmap_kwargs,
    )
    for axis, title in ((c2_ax, "B  Cluster 2"),
                        (control_ax, "C  Matched controls")):
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("Module")
        axis.set_ylabel("Module")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)

    fig.suptitle(
        "Non-LOF mutation patterns across K=5 spectral attention modules",
        fontsize=18, fontweight="semibold", y=0.99)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{nonlof_stem}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
        fig.savefig(REPRODUCED / f"{nonlof_stem}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    audit = pd.DataFrame({
        "metric": ["matched_pairs", "cluster2_patients", "control_patients",
                   "modules", "bootstrap_replicates"],
        "value": [matched.pair_id.nunique(), len(groups["Cluster 2"]),
                  len(groups["Matched controls"]), len(module_ids), BOOTSTRAPS],
    })
    audit.to_csv(OUTPUT / "analysis_audit.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a matched GENIE Cluster-2 mutation-state Oncoprint.

The full matched cohort is retained for downstream ablation, pairwise, and
itemset analyses.  The displayed patients and genes are deterministic subsets
chosen without using Cluster-2 differential effects.
"""

from __future__ import annotations

import json
from pathlib import Path
import os

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


ROOT = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments"))
RUN = ROOT / "experiments/candidate_three_state_logofunc_shared200_v20"
OUTPUT = ROOT / "experiments/cluster2_mutation_pattern_interpretability_20261103"
SEED = 42
TMB_BIN_WIDTH = 0.2
DISPLAY_PAIRS = 400
DISPLAY_GENES = 40

WT = "#f7f7f7"
MUTATED = "#f28e2b"
LOF = "#b2182b"
CLUSTER2 = "#d62728"
CONTROL = "#9c9c9c"


def load_patient_data():
    with np.load(RUN / "profiles.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str)
        sample_patients = archive["patients"].astype(str)
    # `profiles.npz["states"]` belongs to the older binary input encoding
    # (WT=1, mutated=2). The selected three-state checkpoint was trained on
    # this separate WT=0, mutated-not-LOF=1, LOF=2 target matrix.
    sample_states = np.load(RUN / "functional_states.npy", mmap_mode="r")

    # Patient state is the most severe observed state over all samples:
    # WT < mutated-not-LOF < LOF. This is used only for the descriptive
    # Oncoprint and downstream mutation-pattern analyses, not for re-encoding.
    order = np.argsort(sample_patients, kind="stable")
    sorted_patients = sample_patients[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_patients[1:] != sorted_patients[:-1]) + 1]
    patient_ids = sorted_patients[boundaries]
    patient_states = np.maximum.reduceat(sample_states[order], boundaries, axis=0)

    labels = ad.read_h5ad(
        RUN / "evaluation/representation/three_state_logofunc/patients.h5ad",
        backed="r",
    ).obs[["leiden_r0p5"]].reset_index()
    covariates = pd.read_csv(
        RUN / "evaluation/representation/patient_umap_covariates.csv.gz"
    )
    frame = covariates.merge(labels, on="patient_id", validate="one_to_one")
    state_index = pd.Series(np.arange(len(patient_ids)), index=patient_ids)
    frame["state_row"] = frame.patient_id.map(state_index)
    if frame.state_row.isna().any() or len(frame) != len(patient_ids):
        raise ValueError("Patient state/covariate alignment failed")
    frame["state_row"] = frame.state_row.astype(int)
    frame["cluster2"] = frame.leiden_r0p5.astype(str).eq("2")
    frame["tmb_bin"] = np.floor(frame.log1p_tmb / TMB_BIN_WIDTH).astype(int)
    return genes, patient_states, frame


def coarsened_exact_match(frame):
    rng = np.random.default_rng(SEED)
    keys = ["cancer_type", "institution", "assay", "tmb_bin"]
    pairs = []
    pair_id = 0
    for values, local in frame.groupby(keys, sort=True, dropna=False):
        cases = local.loc[local.cluster2].index.to_numpy()
        controls = local.loc[~local.cluster2].index.to_numpy()
        count = min(len(cases), len(controls))
        if not count:
            continue
        cases = rng.permutation(cases)[:count]
        controls = rng.permutation(controls)[:count]
        for case, control in zip(cases, controls):
            pairs.append((pair_id, case, True, *values))
            pairs.append((pair_id, control, False, *values))
            pair_id += 1
    columns = ["pair_id", "frame_row", "cluster2", *keys]
    matched = pd.DataFrame(pairs, columns=columns)
    matched = matched.merge(
        frame.reset_index(names="frame_row"),
        on=["frame_row", "cluster2", *keys],
        validate="one_to_one",
    )
    if matched.patient_id.duplicated().any():
        raise ValueError("Matching reused a patient")
    return matched


def standardized_mean_difference(case, control):
    denominator = np.sqrt((case.var(ddof=1) + control.var(ddof=1)) / 2)
    return float((case.mean() - control.mean()) / denominator) if denominator else 0.0


def select_display(matched, states, genes):
    rng = np.random.default_rng(SEED + 1)
    pair_ids = matched.pair_id.drop_duplicates().to_numpy()
    selected_pairs = rng.choice(pair_ids, size=min(DISPLAY_PAIRS, len(pair_ids)), replace=False)
    display = matched.loc[matched.pair_id.isin(selected_pairs)].copy()

    full_rows = matched.state_row.to_numpy()
    overall_frequency = (states[full_rows] > 0).mean(axis=0)
    gene_order = np.argsort(-overall_frequency, kind="stable")[:DISPLAY_GENES]

    # Preserve the Cluster-2/control boundary while hierarchically clustering
    # patients within each block. Separate indicators retain the distinction
    # between mutated-not-LOF and LOF in the sparse Jaccard distance.
    ordered_blocks = []
    for group in (True, False):
        block = display.loc[display.cluster2.eq(group)].copy()
        block_states = states[block.state_row.to_numpy()][:, gene_order]
        features = np.concatenate([block_states == 1, block_states == 2], axis=1)
        if len(block) > 2:
            distances = np.nan_to_num(pdist(features, metric="jaccard"), nan=0.0)
            tree = linkage(distances, method="average", optimal_ordering=True)
            block = block.iloc[leaves_list(tree)]
        ordered_blocks.append(block)
    display = pd.concat(ordered_blocks, ignore_index=True)
    return display, gene_order, overall_frequency


def plot_oncoprint(display, states, genes, gene_order, output):
    matrix = states[display.state_row.to_numpy()][:, gene_order].T
    n_cluster2 = int(display.cluster2.sum())
    cancer_counts = display.cancer_type.value_counts()
    major_cancers = list(cancer_counts.head(10).index)
    display["cancer_display"] = display.cancer_type.where(
        display.cancer_type.isin(major_cancers), "Other"
    )
    cancer_levels = major_cancers + (["Other"] if (display.cancer_display == "Other").any() else [])
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(1, len(cancer_levels))))
    cancer_colors = dict(zip(cancer_levels, palette))
    cancer_track = np.array([cancer_levels.index(value) for value in display.cancer_display])

    fig = plt.figure(figsize=(24, 15), constrained_layout=True)
    grid = fig.add_gridspec(5, 1, height_ratios=[.20, .20, .33, 8.8, 1.0], hspace=.03)
    ax_group = fig.add_subplot(grid[0])
    ax_cancer = fig.add_subplot(grid[1], sharex=ax_group)
    ax_tmb = fig.add_subplot(grid[2], sharex=ax_group)
    ax = fig.add_subplot(grid[3], sharex=ax_group)
    legend_axis = fig.add_subplot(grid[4])
    legend_axis.axis("off")

    group_track = display.cluster2.astype(int).to_numpy()[None, :]
    ax_group.imshow(group_track, aspect="auto", interpolation="nearest",
                    cmap=ListedColormap([CONTROL, CLUSTER2]), vmin=0, vmax=1)
    ax_cancer.imshow(cancer_track[None, :], aspect="auto", interpolation="nearest",
                     cmap=ListedColormap([cancer_colors[x] for x in cancer_levels]),
                     vmin=-.5, vmax=len(cancer_levels)-.5)
    tmb_image = ax_tmb.imshow(display.log1p_tmb.to_numpy()[None, :], aspect="auto",
                              interpolation="nearest", cmap="viridis")
    cmap = ListedColormap([WT, MUTATED, LOF])
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap,
              norm=BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N), rasterized=True)

    for track, label in [(ax_group, "Cluster"), (ax_cancer, "Cancer"),
                         (ax_tmb, "log(1+TMB)")]:
        track.set_yticks([0], [label], fontsize=11, fontweight="bold")
        track.tick_params(axis="x", bottom=False, labelbottom=False)
        for spine in track.spines.values():
            spine.set_visible(False)
    ax.set_yticks(np.arange(len(gene_order)), genes[gene_order], fontsize=10)
    ax.set_xticks([n_cluster2/2, n_cluster2 + (len(display)-n_cluster2)/2],
                  [f"Cluster 2\n(n={n_cluster2:,})",
                   f"Matched controls\n(n={len(display)-n_cluster2:,})"],
                  fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=10)
    for current in (ax_group, ax_cancer, ax_tmb, ax):
        current.axvline(n_cluster2-.5, color="black", linewidth=1.4)
    ax.set_title(
        "GENIE shared-200 Cluster 2 mutation-state Oncoprint",
        fontsize=22, fontweight="bold", pad=16,
    )
    ax.set_ylabel("Top genes by overall matched-cohort mutation frequency",
                  fontsize=13, labelpad=12)

    state_handles = [Patch(facecolor=WT, edgecolor=".75", label="Wild type"),
                     Patch(facecolor=MUTATED, label="Mutated, not LOF"),
                     Patch(facecolor=LOF, label="LOF")]
    cancer_handles = [Patch(facecolor=cancer_colors[x], label=x) for x in cancer_levels]
    first = legend_axis.legend(handles=state_handles, title="Gene state", frameon=False,
                               ncol=3, loc="upper center", bbox_to_anchor=(.25, 1.0))
    legend_axis.add_artist(first)
    legend_axis.legend(handles=cancer_handles, title="Cancer type", frameon=False,
                       ncol=6, loc="upper center", bbox_to_anchor=(.72, 1.0), fontsize=9)
    colorbar = fig.colorbar(tmb_image, ax=ax_tmb, orientation="vertical", fraction=.008,
                           pad=.005, aspect=5)
    colorbar.ax.set_ylabel("", rotation=0)
    legend_axis.text(
        .5, .02,
        "Controls matched exactly on cancer type, institution, assay, and 0.2-wide log(1+TMB) bins. "
        "Displayed patients are a fixed random subset of matched pairs; columns are hierarchically clustered within each group.",
        transform=legend_axis.transAxes, ha="center", va="bottom", fontsize=11, color=".25",
    )
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(output / f"cluster2_matched_oncoprint.{suffix}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    return matrix


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    genes, states, frame = load_patient_data()
    matched = coarsened_exact_match(frame)
    display, gene_order, overall_frequency = select_display(matched, states, genes)
    matrix = plot_oncoprint(display, states, genes, gene_order, OUTPUT)

    matched.to_csv(OUTPUT / "matched_patient_cohort.tsv.gz", sep="\t", index=False)
    display.to_csv(
        OUTPUT / "oncoprint_display_patients.tsv", sep="\t", index=False
    )
    pd.DataFrame({
        "gene": genes,
        "overall_matched_mutation_frequency": overall_frequency,
        "displayed": np.isin(np.arange(len(genes)), gene_order),
        "display_order": pd.Series(np.arange(len(gene_order)), index=gene_order).reindex(
            np.arange(len(genes))
        ).to_numpy(),
    }).to_csv(OUTPUT / "gene_selection.tsv", sep="\t", index=False)
    np.savez_compressed(
        OUTPUT / "oncoprint_display_matrix.npz",
        states=matrix, genes=genes[gene_order],
        patient_ids=display.patient_id.to_numpy(), cluster2=display.cluster2.to_numpy(),
    )

    case = matched.loc[matched.cluster2, "log1p_tmb"]
    control = matched.loc[~matched.cluster2, "log1p_tmb"]
    audit = {
        "source_patients": int(len(frame)),
        "source_cluster2_patients": int(frame.cluster2.sum()),
        "matched_pairs": int(matched.pair_id.nunique()),
        "matched_patients": int(len(matched)),
        "display_pairs": int(display.pair_id.nunique()),
        "display_genes": int(len(gene_order)),
        "matching": {
            "exact": ["cancer_type", "institution", "assay", "tmb_bin"],
            "tmb_definition": f"floor(log1p_tmb/{TMB_BIN_WIDTH})",
            "without_replacement": True,
            "seed": SEED,
        },
        "balance": {
            "cluster2_mean_log1p_tmb": float(case.mean()),
            "control_mean_log1p_tmb": float(control.mean()),
            "log1p_tmb_standardized_mean_difference": standardized_mean_difference(case, control),
        },
        "patient_state_aggregation": "maximum state across samples: WT < mutated-not-LOF < LOF",
        "gene_selection": "top overall mutation frequency in full matched cohort; cluster labels unused",
        "column_ordering": "average-linkage hierarchical clustering within each group using Jaccard distance over separate mutated-not-LOF and LOF indicators",
    }
    (OUTPUT / "matching_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

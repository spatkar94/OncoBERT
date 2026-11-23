"""Merged variance and PCA/UMAP covariate diagnostics for NMF, GAT, and OncoBERT."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, to_hex
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from cancer_type_metrics import valid_labels
from diagnostics_utils import eta_squared
from compute_metrics import continuous_variance


ROOT = Path(os.environ.get(
    "ONCOBERT_EXPERIMENT_ROOT",
    Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments",
))
FINAL = ROOT / "experiments/final_shared200_five_encoder_benchmark_20261029"
SOURCES = {
    "nmf": ROOT / "experiments/nmf_gat_set_merged_diagnostics_shared200_v20/nmf_projection/nmf_k13/patients.h5ad",
    "gat": ROOT / "experiments/candidate_mutation_projector_dim10_mean_shared200_v20/evaluation/representation_diagnostics/mutation_projector_gat10_three_state/patients.h5ad",
    "set": ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/evaluation/representation/three_state_logofunc/patients.h5ad",
}
TITLES = {"nmf": "NMF", "gat": "MutationProjector (GAT)",
          "set": "OncoBERT (Set Transformer)"}
COLORS = {"nmf": "#9c755f", "gat": "#59a14f", "set": "#4e79a7"}


def extent(values):
    low, high = np.quantile(values, [.005, .995], axis=0)
    padding = np.maximum((high - low) * .04, 1e-8)
    return low - padding, high + padding


def categories(frame: pd.DataFrame, field: str):
    values = frame[field].fillna("Unknown / conflicting").astype(str)
    if field == "cancer_type":
        invalid = values.str.lower().str.contains("unknown|conflict|multiple|nan", regex=True)
        common = values[~invalid].value_counts().head(12).index
        values = values.where(values.isin(common), "Other cancer types")
        values = values.mask(invalid, "Unknown / conflicting")
    levels = sorted(values.unique())
    palette = {value: to_hex(plt.colormaps["tab20"](index % 20))
               for index, value in enumerate(levels)}
    if "Other cancer types" in palette:
        palette["Other cancer types"] = "#c8c8c8"
    for value in levels:
        if "unknown" in value.lower() or "multiple" in value.lower():
            palette[value] = "#777777"
    return values, palette


def variance_table(objects, matched):
    valid = valid_labels(matched) & matched[["assay", "institution", "log1p_tmb"]].notna().all(axis=1)
    frame = matched.loc[valid].copy()
    test = frame.split.eq("test").to_numpy()
    rows = []
    for model, obj in objects.items():
        index = pd.Index(obj.obs_names).get_indexer(frame.index)
        if (index < 0).any():
            raise ValueError(f"Missing matched patients for {model}")
        held = np.asarray(obj.X)[index[test]].astype(np.float64)
        centered = held - held.mean(axis=0)
        rows.append({
            "model": model, "display_name": TITLES[model],
            "representation_dimension": held.shape[1],
            "heldout_patients": int(test.sum()),
            "tmb_variance_explained": continuous_variance(
                centered, frame.log1p_tmb.to_numpy()[test]),
            "panel_source_variance_explained": eta_squared(
                held, frame.assay.to_numpy()[test])[0],
            "institution_variance_explained": eta_squared(
                held, frame.institution.to_numpy()[test])[0],
        })
    return pd.DataFrame(rows)


def variance_figure(table, output):
    models = list(SOURCES)
    covariates = [
        ("institution_variance_explained", "Institution"),
        ("panel_source_variance_explained", "Panel source"),
        ("tmb_variance_explained", "log(1 + TMB)"),
    ]
    positions = np.arange(len(covariates))
    width = .24
    fig, axis = plt.subplots(figsize=(10.5, 6.5))
    indexed = table.set_index("model")
    for index, model in enumerate(models):
        values = 100 * indexed.loc[model, [key for key, _ in covariates]].to_numpy(float)
        bars = axis.bar(positions + (index - 1) * width, values, width,
                        color=COLORS[model], label=TITLES[model])
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values],
                       padding=4, fontsize=9)
    axis.set_xticks(positions, [label for _, label in covariates])
    axis.set_ylabel("Native embedding variance explained (%)")
    axis.set_ylim(0, max(5, 100 * table[[key for key, _ in covariates]].to_numpy().max() * 1.22))
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.grid(axis="y", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Shared-200 representation associations", fontsize=16, y=.97)
    fig.text(.5, .025,
             "Identical 11,188 held-out patients · all native embedding dimensions · unadjusted, non-additive associations\n"
             "TMB: trace OLS R² with log(1 + TMB) · categorical covariates: multivariate η²",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.11, right=.98, bottom=.16, top=.86)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"embedding_variance_explained.{extension}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)


def overlay_figure(objects, frame, field, label):
    models = list(SOURCES)
    draw = np.random.default_rng(42).permutation(len(frame))
    fig, axes = plt.subplots(2, len(models), figsize=(16.5, 9.1))
    if field == "log1p_tmb":
        raw = frame[field].to_numpy(float)
        maximum = float(np.nanquantile(raw, .995))
        norm = Normalize(0, maximum, clip=True)
        colors = plt.colormaps["magma"](norm(np.nan_to_num(raw)))
        colors[~np.isfinite(raw)] = [.6, .6, .6, 1]
        palette = None
    else:
        values, palette = categories(frame, field)
        colors = values.map(palette).to_numpy()
    for column, model in enumerate(models):
        obj = objects[model]
        for row, basis in enumerate(["X_umap", "X_pca"]):
            coordinates = np.asarray(obj.obsm[basis])[:, :2]
            axis = axes[row, column]
            axis.scatter(coordinates[draw, 0], coordinates[draw, 1],
                         c=colors[draw], s=.35, alpha=.63, linewidths=0,
                         rasterized=True)
            low, high = extent(coordinates)
            axis.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]))
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=7)
            if row == 0:
                axis.set_xlabel("UMAP 1", fontsize=8)
                axis.set_ylabel("UMAP 2", fontsize=8)
                axis.set_title(TITLES[model], fontsize=11, pad=9)
            else:
                variance = np.asarray(obj.uns["pca"]["variance_ratio"])
                axis.set_xlabel(f"PC1 ({100 * variance[0]:.1f}%)", fontsize=8)
                axis.set_ylabel(f"PC2 ({100 * variance[1]:.1f}%)", fontsize=8)
    right = .97
    if field == "log1p_tmb":
        # Reserve a dedicated strip to the right of the subplot grid.  Passing
        # all axes directly to fig.colorbar lets Matplotlib steal space from
        # the existing axes and previously placed the bar over the last model.
        right = .885
        colorbar_axis = fig.add_axes([.915, .235, .012, .54])
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="magma"),
            cax=colorbar_axis, extend="max")
        ticks = [value for value in [0, 1, 5, 20, 100, 500]
                 if np.log1p(value) <= maximum]
        colorbar.set_ticks(np.log1p(ticks), labels=[str(value) for value in ticks])
        colorbar.set_label("Exact GENIE TMB (mut/Mb)")
    else:
        handles = [Line2D([0], [0], marker="o", linestyle="", color=color,
                          label=value, markersize=5)
                   for value, color in palette.items()]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.845, .5),
                   frameon=False, fontsize=7.5)
        right = .84
    fig.suptitle(f"Shared-200 PCA and UMAP overlays · {label}", fontsize=16, y=.985)
    fig.text(.5, .012,
             "Patient-level representations · identical colors and drawing order across models · central 99% of each projection shown",
             ha="center", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=.055, right=right, top=.91, bottom=.075,
                        wspace=.28, hspace=.27)
    return fig


def main(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    objects = {model: ad.read_h5ad(path) for model, path in SOURCES.items()}
    patient_ids = pd.Index(objects["nmf"].obs_names, name="patient_id")
    for model in ["gat", "set"]:
        np.testing.assert_array_equal(objects[model].obs_names, patient_ids)
    all_covariates = pd.read_csv(
        FINAL / "canonical_assets/covariates/all_patient_covariates.csv.gz",
        index_col="patient_id").loc[patient_ids]
    matched = pd.read_csv(
        FINAL / "canonical_assets/covariates/matched_train_test_covariates.csv.gz",
        index_col="patient_id")
    table = variance_table(objects, matched)
    table.to_csv(output / "embedding_variance_explained.csv", index=False)
    variance_figure(table, output)

    overlay_specs = [
        ("log1p_tmb", "TMB", "tmb"),
        ("assay", "Panel source", "panel_source"),
        ("institution", "Institution", "institution"),
        ("cancer_type", "Cancer type", "cancer_type"),
    ]
    with PdfPages(output / "pca_umap_covariate_overlays.pdf") as document:
        for field, label, stem in overlay_specs:
            figure = overlay_figure(objects, all_covariates, field, label)
            document.savefig(figure, bbox_inches="tight")
            figure.savefig(output / f"pca_umap_overlay_{stem}.pdf", bbox_inches="tight")
            figure.savefig(output / f"pca_umap_overlay_{stem}.png", dpi=190, bbox_inches="tight")
            plt.close(figure)
    (output / "manifest.json").write_text(json.dumps({
        "complete": True, "models": list(SOURCES), "titles": TITLES,
        "sources": {model: str(path) for model, path in SOURCES.items()},
        "patients": len(patient_ids), "heldout_variance_patients": int(table.heldout_patients.iloc[0]),
        "variance_protocol": "native dimensions; trace OLS R2 for log1p TMB; multivariate eta2 for categorical labels; held-out patients only",
        "projection_protocol": "PCA top min(50, native_dim), then UMAP n_neighbors15 min_dist0.3 euclidean seed42",
    }, indent=2))
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)

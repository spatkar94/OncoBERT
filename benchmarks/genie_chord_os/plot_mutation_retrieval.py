"""Standalone binary mutation-retrieval macro-AUPRC comparison."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(os.environ.get(
    "ONCOBERT_EXPERIMENT_ROOT",
    Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments",
))
OUTPUT = ROOT / "experiments" / "mutation_retrieval_macro_auprc_shared200_v20"
PROFILES = ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/profiles.npz"
TEST_MASKS = ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/test_masks.npz"
PATIENT_BOOTSTRAPS = 2_000
SOURCES = {
    "NMF": ROOT / "experiments/nmf_consensus_k2_30_shared200_v20/masked_reconstruction/k_13/test_masked_predictions.npz",
    "MutationProjector\n(GAT)": ROOT / "experiments/candidate_mutation_projector_dim10_mean_shared200_v20/runs/mutation_projector_dim10_mean_vectorized_seed42/test_predictions.npz",
    "OncoBERT\n(Set Transformer)": ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/runs/three_state_seed42/test_predictions.npz",
}


def load_binary(path: Path, nmf: bool):
    saved = np.load(path, allow_pickle=False)
    labels = (saved["labels"] > 0).astype(np.uint8)
    gene_indices = saved["gene_indices"].astype(np.int16)
    scores = (saved["raw_reconstruction_scores"] if nmf
              else 1.0 - saved["probabilities"][:, 0])
    return labels, scores.astype(np.float64), gene_indices


def patient_bootstrap_gene_ap(labels, scores, gene_indices,
                              entry_patient_indices, bootstrap_counts,
                              genes=200, chunk_size=100):
    """Exact weighted AP after clustered patient resampling.

    Predictions are sorted once per gene. For each bootstrap replicate, patient
    multiplicities become observation weights. Equal-score groups are collapsed,
    matching sklearn's average_precision_score treatment of ties exactly.
    """
    replicates = len(bootstrap_counts)
    result = np.empty((replicates, genes), dtype=np.float64)
    for gene in range(genes):
        selected = gene_indices == gene
        gene_labels = labels[selected].astype(np.float64)
        gene_scores = scores[selected]
        gene_patients = entry_patient_indices[selected]
        order = np.argsort(-gene_scores, kind="stable")
        gene_labels = gene_labels[order]
        gene_scores = gene_scores[order]
        gene_patients = gene_patients[order]
        group_starts = np.r_[0, np.flatnonzero(gene_scores[1:] != gene_scores[:-1]) + 1]
        for start in range(0, replicates, chunk_size):
            stop = min(start + chunk_size, replicates)
            weights = bootstrap_counts[start:stop, gene_patients].astype(np.float64)
            positive = np.add.reduceat(weights * gene_labels, group_starts, axis=1)
            total = np.add.reduceat(weights, group_starts, axis=1)
            cumulative_positive = np.cumsum(positive, axis=1)
            cumulative_total = np.cumsum(total, axis=1)
            precision = np.divide(
                cumulative_positive, cumulative_total,
                out=np.zeros_like(cumulative_positive), where=cumulative_total > 0,
            )
            positive_total = cumulative_positive[:, -1]
            result[start:stop, gene] = np.divide(
                np.sum(precision * positive, axis=1), positive_total,
                out=np.zeros_like(positive_total), where=positive_total > 0,
            )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    reference = None
    prediction_data = {}
    for model, path in SOURCES.items():
        labels, scores, gene_indices = load_binary(path, model.startswith("NMF"))
        clean_model = model.replace("\n", " ")
        prediction_data[clean_model] = (labels, scores, gene_indices)
        if reference is None:
            reference = (labels.copy(), gene_indices.copy())
        else:
            np.testing.assert_array_equal(labels, reference[0])
            np.testing.assert_array_equal(gene_indices, reference[1])
        for gene in range(200):
            selected = gene_indices == gene
            y = labels[selected]
            if not (0 < y.sum() < len(y)):
                raise ValueError(f"Gene {gene} lacks both binary classes")
            rows.append({
                "model": clean_model,
                "gene_index": gene,
                "masked_entries": int(selected.sum()),
                "mutated_entries": int(y.sum()),
                "prevalence": float(y.mean()),
                "average_precision": float(average_precision_score(y, scores[selected])),
            })
    per_gene = pd.DataFrame(rows)
    per_gene.to_csv(OUTPUT / "per_gene_binary_mutation_auprc.csv", index=False)

    # Recover the patient attached to every flattened masked prediction. Each
    # held-out sample contributes exactly 30 contiguous entries; patients with
    # multiple samples are kept together in every clustered bootstrap draw.
    profiles = np.load(PROFILES, allow_pickle=False)
    test_rows = np.flatnonzero(profiles["split"] == "test")
    test_patients = profiles["patients"][test_rows]
    patient_ids, sample_patient_indices = np.unique(
        test_patients, return_inverse=True,
    )
    saved_masks = np.load(TEST_MASKS, allow_pickle=False)
    np.testing.assert_array_equal(saved_masks["sample_rows"], test_rows)
    test_masks = saved_masks["masks"].astype(bool)
    if test_masks.shape != (len(test_rows), 200) or not np.all(test_masks.sum(1) == 30):
        raise ValueError("Expected exactly 30 fixed masked genes per held-out sample")
    np.testing.assert_array_equal(np.nonzero(test_masks)[1], reference[1])
    entry_patient_indices = np.repeat(sample_patient_indices, 30)
    if len(entry_patient_indices) != len(reference[0]):
        raise ValueError("Patient-to-prediction alignment failed")

    bootstrap_rng = np.random.default_rng(42)
    bootstrap_counts = np.empty(
        (PATIENT_BOOTSTRAPS, len(patient_ids)), dtype=np.int16,
    )
    patient_probabilities = np.full(len(patient_ids), 1 / len(patient_ids))
    for start in range(0, PATIENT_BOOTSTRAPS, 100):
        stop = min(start + 100, PATIENT_BOOTSTRAPS)
        bootstrap_counts[start:stop] = bootstrap_rng.multinomial(
            len(patient_ids), patient_probabilities, size=stop - start,
        ).astype(np.int16)
    bootstrap_gene_ap = {
        model: patient_bootstrap_gene_ap(
            labels, scores, gene_indices, entry_patient_indices,
            bootstrap_counts,
        )
        for model, (labels, scores, gene_indices) in prediction_data.items()
    }

    summary = []
    for display_name in SOURCES:
        model = display_name.replace("\n", " ")
        values = per_gene.loc[per_gene.model == model, "average_precision"].to_numpy()
        bootstrap = bootstrap_gene_ap[model].mean(axis=1)
        summary.append({
            "model": model,
            "macro_auprc": float(values.mean()),
            "patient_bootstrap_95ci_lower": float(np.quantile(bootstrap, .025)),
            "patient_bootstrap_95ci_upper": float(np.quantile(bootstrap, .975)),
            "genes": len(values),
            "masked_entries": int(len(reference[0])),
            "mutation_prevalence": float(reference[0].mean()),
        })
    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(OUTPUT / "binary_mutation_macro_auprc.csv", index=False)

    values = summary_table.macro_auprc.to_numpy()
    lower = values - summary_table.patient_bootstrap_95ci_lower.to_numpy()
    upper = summary_table.patient_bootstrap_95ci_upper.to_numpy() - values
    colors = ["#9c755f", "#59a14f", "#4e79a7"]
    fig, axis = plt.subplots(figsize=(8.8, 6.8))
    positions = np.arange(len(summary_table))
    bars = axis.bar(positions, values, color=colors, width=.68,
                    yerr=np.vstack([lower, upper]), capsize=5,
                    error_kw={"elinewidth": 1.4, "capthick": 1.4})
    axis.set_xticks(positions, list(SOURCES.keys()), fontsize=10.5)
    axis.set_ylabel("Gene-macro AUPRC", fontsize=12)
    upper_limit = max(summary_table.patient_bootstrap_95ci_upper) + .038
    axis.set_ylim(0, upper_limit)
    axis.tick_params(axis="y", labelsize=10.5)
    axis.grid(axis="y", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value, ci_upper in zip(
        bars, values, summary_table.patient_bootstrap_95ci_upper.to_numpy()
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2, ci_upper + .006,
            f"{value:.3f}", ha="center", va="bottom", fontsize=11,
            fontweight="semibold",
        )
    fig.suptitle("Masked mutation-state retrieval", fontsize=16, y=.965)
    axis.set_title("Shared-200 held-out cohort", fontsize=11, color="#555555", pad=13)
    fig.text(
        .5, .065,
        "Binary target: wild type vs mutated (mutated-not-LOF or LOF)\n"
        "Identical fixed 15% test masks · macro average across 200 genes\n"
        f"Error bars: 95% clustered bootstrap interval over {len(patient_ids):,} patients "
        f"({PATIENT_BOOTSTRAPS:,} replicates) · NMF rank selected by maximum macro AUPRC",
        ha="center", va="center", fontsize=8.8, color="#444444", linespacing=1.35,
    )
    fig.subplots_adjust(left=.13, right=.98, bottom=.25, top=.84)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"binary_mutation_macro_auprc.{extension}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Bin genes into the three requested held-out mutation-frequency ranges.
    frequency_edges = [0, 5, 10, np.inf]
    frequency_labels = ["<5%", "5–10%", ">10%"]
    descending_labels = list(reversed(frequency_labels))
    per_gene["frequency_percent"] = 100 * per_gene.prevalence
    per_gene["frequency_bin"] = pd.cut(
        per_gene.frequency_percent,
        bins=frequency_edges,
        labels=frequency_labels,
        right=False,
        include_lowest=True,
    )
    binned = (per_gene.groupby(["model", "frequency_bin"], observed=False)
              .agg(mean_auprc=("average_precision", "mean"),
                   genes=("gene_index", "size"),
                   mean_mutation_frequency=("frequency_percent", "mean"))
              .reset_index())
    binned["frequency_bin"] = binned.frequency_bin.astype(str)
    # Clustered 95% bootstrap intervals over patients. For every patient draw,
    # recompute each gene's AP and then average across the fixed genes in a bin.
    intervals = []
    for (model, frequency_bin), local in per_gene.groupby(
        ["model", "frequency_bin"], observed=False
    ):
        genes_in_bin = local.gene_index.to_numpy(int)
        if not len(genes_in_bin):
            continue
        bootstrap = bootstrap_gene_ap[model][:, genes_in_bin].mean(axis=1)
        intervals.append({
            "model": model,
            "frequency_bin": str(frequency_bin),
            "bootstrap_95ci_lower": float(np.quantile(bootstrap, .025)),
            "bootstrap_95ci_upper": float(np.quantile(bootstrap, .975)),
        })
    binned = binned.merge(pd.DataFrame(intervals),
                          on=["model", "frequency_bin"], how="left")
    binned.to_csv(OUTPUT / "mutation_frequency_bin_auprc.csv", index=False)

    fig, axis = plt.subplots(figsize=(11.3, 7.2))
    positions = np.arange(len(descending_labels))
    width = .21
    model_offset = .245
    model_names = [name.replace("\n", " ") for name in SOURCES]
    for index, (model, color) in enumerate(zip(model_names, colors)):
        local = binned[binned.model.eq(model)].set_index("frequency_bin").loc[descending_labels]
        values = local.mean_auprc.to_numpy(float)
        lower = values - local.bootstrap_95ci_lower.to_numpy(float)
        upper = local.bootstrap_95ci_upper.to_numpy(float) - values
        bars = axis.bar(
            positions + (index - 1) * model_offset,
            values,
            width=width,
            color=color,
            yerr=np.vstack([
                values - local.bootstrap_95ci_lower.to_numpy(float),
                local.bootstrap_95ci_upper.to_numpy(float) - values,
            ]),
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
            label=model,
        )
        for bar, value, ci_upper in zip(
            bars, values, local.bootstrap_95ci_upper.to_numpy(float),
        ):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                ci_upper + .009,
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold",
            )
    reference_counts = (binned[binned.model.eq(model_names[0])]
                        .set_index("frequency_bin").loc[descending_labels, "genes"])
    tick_labels = [f"{label}\n(n={int(count)})"
                   for label, count in zip(descending_labels, reference_counts)]
    axis.set_xticks(positions, tick_labels)
    axis.set_xlabel("Gene mutation frequency", fontsize=12)
    axis.set_ylabel("Gene mutation state prediction AUPRC", fontsize=12)
    axis.set_title("Mutation frequency versus masked-state retrieval", fontsize=15, pad=16)
    axis.grid(axis="y", alpha=.18)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    axis.set_ylim(0, min(1.0, binned.bootstrap_95ci_upper.max() + .10))
    fig.text(
        .5, .035,
        f"Bars show mean AUPRC; error bars are clustered patient-bootstrap 95% CIs "
        f"({PATIENT_BOOTSTRAPS:,} replicates).\n"
        "Patients are resampled with replacement; all samples from a selected patient remain grouped.",
        ha="center", va="center", fontsize=9, color="#444444",
    )
    fig.subplots_adjust(left=.12, right=.98, bottom=.18, top=.88)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"mutation_frequency_vs_retrieval_auprc.{extension}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "complete": True,
        "target": "wild type (state 0) versus any mutation (states 1 or 2)",
        "masking": "identical canonical fixed uniform-15% test masks; exactly 30/200 genes per sample",
        "samples": 12579,
        "masked_entries": int(len(reference[0])),
        "genes": 200,
        "macro_definition": "unweighted mean of gene-specific average precision",
        "confidence_intervals": {
            "method": "clustered nonparametric patient bootstrap",
            "replicates": PATIENT_BOOTSTRAPS,
            "unique_test_patients": int(len(patient_ids)),
            "test_samples": int(len(test_rows)),
            "clustering": "all samples and masked predictions from a patient resampled together",
        },
        "nmf_selection": "K=13, maximum NMF held-out macro AUPRC over K=2..30",
        "sources": {name.replace("\n", " "): str(path) for name, path in SOURCES.items()},
        "metrics": summary,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary_table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

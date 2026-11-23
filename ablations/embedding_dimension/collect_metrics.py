"""Collect three-state set-Transformer embedding-dimension ablation metrics."""
from __future__ import annotations

import csv
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
EXPERIMENT = ROOT / "experiments/three_state_set_dim_ablation_shared200_v20"
SOURCES = {
    32: EXPERIMENT / "runs/dim32_seed42",
    64: EXPERIMENT / "runs/dim64_seed42",
    128: ROOT / "experiments/candidate_three_state_logofunc_shared200_v20/runs/three_state_seed42",
    256: EXPERIMENT / "runs/dim256_seed42",
}


def main() -> None:
    output = EXPERIMENT / "results"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    summary = []
    gene_rows = []
    gene_state_rows = []
    reference = None
    for dimension, source in SOURCES.items():
        metrics = json.loads((source / "test_metrics.json").read_text())
        predictions = np.load(source / "test_predictions.npz", allow_pickle=False)
        state_labels = predictions["labels"].astype(np.uint8)
        labels = (state_labels > 0).astype(np.uint8)
        gene_indices = predictions["gene_indices"].astype(np.int16)
        mutation_scores = 1.0 - predictions["probabilities"][:, 0]
        if reference is None:
            reference = (labels.copy(), gene_indices.copy())
        else:
            np.testing.assert_array_equal(labels, reference[0])
            np.testing.assert_array_equal(gene_indices, reference[1])
        per_gene = []
        per_gene_state_macro = []
        for gene in range(200):
            selected = gene_indices == gene
            y = labels[selected]
            if not (0 < y.sum() < len(y)):
                raise ValueError(f"Gene {gene} lacks both binary classes")
            value = float(average_precision_score(y, mutation_scores[selected]))
            per_gene.append(value)
            gene_rows.append({
                "embedding_dim": dimension, "gene_index": gene,
                "masked_entries": int(selected.sum()), "mutated_entries": int(y.sum()),
                "prevalence": float(y.mean()), "binary_mutation_auprc": value,
            })
            supported_state_ap = []
            for state_id, state_name in enumerate(["WILD_TYPE", "MUTATED_NOT_LOF", "LOF"]):
                state_target = state_labels[selected] == state_id
                positives = int(state_target.sum())
                state_ap = (float(average_precision_score(
                    state_target, predictions["probabilities"][selected, state_id]
                )) if 0 < positives < int(selected.sum()) else None)
                gene_state_rows.append({
                    "embedding_dim": dimension, "gene_index": gene,
                    "state_id": state_id, "state": state_name,
                    "masked_entries": int(selected.sum()), "positive_entries": positives,
                    "prevalence": float(state_target.mean()),
                    "average_precision": state_ap,
                })
                if state_ap is not None:
                    supported_state_ap.append(state_ap)
            if not supported_state_ap:
                raise ValueError(f"Gene {gene} has no supported state AUPRC")
            per_gene_state_macro.append(float(np.mean(supported_state_ap)))
        per_gene = np.asarray(per_gene)
        per_gene_state_macro = np.asarray(per_gene_state_macro)
        indices = rng.integers(0, 200, size=(20_000, 200))
        bootstrap = per_gene[indices].mean(axis=1)
        state_bootstrap = per_gene_state_macro[indices].mean(axis=1)
        summary.append({
            "embedding_dim": dimension,
            "per_gene_three_state_macro_auprc": float(per_gene_state_macro.mean()),
            "per_gene_three_state_bootstrap_95ci_lower": float(np.quantile(state_bootstrap, .025)),
            "per_gene_three_state_bootstrap_95ci_upper": float(np.quantile(state_bootstrap, .975)),
            "binary_mutation_macro_auprc": float(per_gene.mean()),
            "gene_bootstrap_95ci_lower": float(np.quantile(bootstrap, .025)),
            "gene_bootstrap_95ci_upper": float(np.quantile(bootstrap, .975)),
            "three_state_macro_auprc": float(metrics["state_macro_auprc"]),
            "mutation_micro_auprc": float(metrics["mutation_micro_auprc"]),
            "reconstruction_ce": float(metrics["reconstruction_ce"]),
            "best_epoch": int(metrics["best_epoch"]),
            "source": str(source),
        })
    table = pd.DataFrame(summary).sort_values("embedding_dim")
    table.to_csv(output / "dimension_ablation_metrics.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(output / "per_gene_binary_mutation_auprc.csv", index=False)
    pd.DataFrame(gene_state_rows).to_csv(output / "per_gene_state_auprc.csv", index=False)

    x = np.arange(len(table))
    colors = ["#a0cbe8", "#4e79a7", "#f28e2b", "#e15759"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    panels = [
        ("per_gene_three_state_macro_auprc", "per_gene_three_state_bootstrap_95ci_lower",
         "per_gene_three_state_bootstrap_95ci_upper", "A  Three-state reconstruction",
         "Per-gene three-state macro AUPRC"),
        ("binary_mutation_macro_auprc", "gene_bootstrap_95ci_lower",
         "gene_bootstrap_95ci_upper", "B  Collapsed binary mutation retrieval",
         "WT-versus-mutated macro AUPRC"),
    ]
    for axis, (metric, low, high, title, ylabel) in zip(axes, panels):
        values = table[metric].to_numpy()
        lower = values - table[low].to_numpy()
        upper = table[high].to_numpy() - values
        bars = axis.bar(x, values, color=colors, width=.68,
                        yerr=np.vstack([lower, upper]), capsize=5)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=5, fontsize=10)
        axis.set_xticks(x, [str(value) for value in table.embedding_dim])
        axis.set_xlabel("Token / patient embedding dimension")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0, table[high].max() * 1.25)
        axis.set_title(title, loc="left")
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Three-state set Transformer · embedding-dimension ablation", fontsize=15, y=.97)
    fig.text(.5, .025,
             "Three-state metric averages supported WT / mutated-not-LOF / LOF AP within each gene, then across 200 genes.\n"
             "Binary target collapses both mutation states. Identical fixed 15% masks; error bars are 95% gene-bootstrap intervals.",
             ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.19, top=.87)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"embedding_dimension_macro_auprc.{extension}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)

    best = table.loc[table.per_gene_three_state_macro_auprc.idxmax()]
    manifest = {
        "complete": True,
        "dimensions": table.embedding_dim.tolist(),
        "best_dimension_by_per_gene_three_state_macro_auprc": int(best.embedding_dim),
        "best_per_gene_three_state_macro_auprc": float(best.per_gene_three_state_macro_auprc),
        "primary_metric": "mean across 200 genes of each gene's mean supported state-specific AP (WT, mutated-not-LOF, LOF)",
        "secondary_metric": "unweighted mean of 200 gene-specific AP values for WT versus any mutation",
        "global_three_state_metric": "three_state_macro_auprc retains the original trainer metric: AP per state pooled across genes, then mean across three states",
        "architecture_control": "8 layers; 8 heads; FFN width 2048; dropout 0.1; only embedding dimension varies",
        "training_control": "seed42; uniform15% masks; fixed validation/test masks; AdamW; lr1e-4; weight_decay1e-4; max100 epochs; patience20",
        "reused_128d_run": True,
        "results": summary,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

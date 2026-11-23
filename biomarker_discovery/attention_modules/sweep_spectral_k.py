#!/usr/bin/env python3
"""Empirically select K for Cluster-2 attention spectral modules."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
INPUT = (ROOT / "artifacts/cluster2_attention_modules/final_layer" /
         "cluster2_final_layer_directed_gene_attention.tsv.gz")
OUTPUT = ROOT / "artifacts/cluster2_attention_modules/spectral_k_sweep"
REPRODUCED = ROOT / "reproduced_figures"
K_VALUES = range(2, 31)
SEEDS = range(20)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPRODUCED.mkdir(parents=True, exist_ok=True)

    directed = pd.read_csv(INPUT, sep="\t", index_col=0)
    matrix = directed.to_numpy(np.float64)
    np.fill_diagonal(matrix, 0.0)
    symmetric = 0.5 * (matrix + matrix.T)
    strength = symmetric.sum(axis=1)
    expected = np.outer(strength, strength) / strength.sum()
    eps = np.finfo(np.float64).eps
    adjusted = np.log2((symmetric + eps) / (expected + eps))
    np.fill_diagonal(adjusted, 0.0)

    correlation = np.nan_to_num(np.corrcoef(adjusted), nan=0.0)
    affinity = np.clip((correlation + 1.0) / 2.0, 0.0, 1.0) ** 2
    np.fill_diagonal(affinity, 1.0)
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)

    # Normalized-Laplacian eigengaps are deterministic for this affinity.
    eigenvalues = np.linalg.eigvalsh(laplacian(affinity, normed=True))
    pd.DataFrame({"index": np.arange(len(eigenvalues)),
                  "eigenvalue": eigenvalues}).to_csv(
        OUTPUT / "normalized_laplacian_eigenvalues.tsv", sep="\t", index=False)

    replicate_rows: list[dict[str, float | int]] = []
    labels_by_k: dict[int, list[np.ndarray]] = {}
    for k in K_VALUES:
        labels_by_k[k] = []
        for seed in SEEDS:
            labels = SpectralClustering(
                n_clusters=k, affinity="precomputed",
                assign_labels="kmeans", n_init=50,
                random_state=seed,
            ).fit_predict(affinity)
            labels_by_k[k].append(labels)
            counts = np.bincount(labels)
            replicate_rows.append({
                "k": k,
                "seed": seed,
                "silhouette": silhouette_score(distance, labels,
                                                metric="precomputed"),
                "minimum_module_size": int(counts.min()),
                "maximum_module_size": int(counts.max()),
                "singleton_modules": int(np.sum(counts == 1)),
                "modules_smaller_than_5": int(np.sum(counts < 5)),
            })
    replicates = pd.DataFrame(replicate_rows)
    replicates.to_csv(OUTPUT / "spectral_k_seed_metrics.tsv",
                      sep="\t", index=False)

    summary_rows: list[dict[str, float | int]] = []
    for k in K_VALUES:
        subset = replicates[replicates.k == k]
        aris = [adjusted_rand_score(labels_by_k[k][i], labels_by_k[k][j])
                for i, j in combinations(range(len(SEEDS)), 2)]
        summary_rows.append({
            "k": k,
            "mean_silhouette": subset.silhouette.mean(),
            "sd_silhouette": subset.silhouette.std(ddof=1),
            "mean_pairwise_ari": np.mean(aris),
            "minimum_pairwise_ari": np.min(aris),
            "median_minimum_module_size": subset.minimum_module_size.median(),
            "minimum_module_size_across_seeds": subset.minimum_module_size.min(),
            "mean_singleton_modules": subset.singleton_modules.mean(),
            "mean_modules_smaller_than_5": subset.modules_smaller_than_5.mean(),
            "eigengap_after_k": eigenvalues[k] - eigenvalues[k - 1],
        })
    summary = pd.DataFrame(summary_rows)

    # Selection rule: avoid tiny modules and unstable labelings, then maximize
    # held-profile silhouette. These constraints were declared before viewing
    # the sweep: median smallest module >=5 and mean seed ARI >=0.90.
    eligible = summary[(summary.median_minimum_module_size >= 5) &
                       (summary.mean_pairwise_ari >= 0.90)]
    if eligible.empty:
        eligible = summary[summary.median_minimum_module_size >= 5]
    selected_k = int(eligible.loc[eligible.mean_silhouette.idxmax(), "k"])
    raw_k = int(summary.loc[summary.mean_silhouette.idxmax(), "k"])
    summary["selected_stable_nondegenerate"] = summary.k.eq(selected_k)
    summary["raw_silhouette_optimum"] = summary.k.eq(raw_k)
    summary.to_csv(OUTPUT / "spectral_k_sweep_summary.tsv",
                   sep="\t", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    x = summary.k.to_numpy()
    ax = axes[0, 0]
    ax.errorbar(x, summary.mean_silhouette, yerr=summary.sd_silhouette,
                color="#2b6ca3", marker="o", ms=4, capsize=2, lw=1.5)
    ax.axvline(selected_k, color="#d62728", ls="--", lw=1.8)
    ax.set(title="A  Attention-profile separation",
           xlabel="Number of spectral modules (K)", ylabel="Mean silhouette")

    ax = axes[0, 1]
    ax.plot(x, summary.mean_pairwise_ari, color="#4c956c", marker="o", ms=4)
    ax.axhline(0.90, color="black", ls=":", lw=1.2)
    ax.axvline(selected_k, color="#d62728", ls="--", lw=1.8)
    ax.set(title="B  Stability across 20 seeds",
           xlabel="Number of spectral modules (K)",
           ylabel="Mean pairwise adjusted Rand index", ylim=(0, 1.04))

    ax = axes[1, 0]
    ax.plot(x, summary.median_minimum_module_size, color="#8b5e3c",
            marker="o", ms=4, label="Median smallest module")
    ax.plot(x, summary.mean_modules_smaller_than_5, color="#9467bd",
            marker="s", ms=4, label="Mean modules with <5 genes")
    ax.axhline(5, color="black", ls=":", lw=1.2)
    ax.axvline(selected_k, color="#d62728", ls="--", lw=1.8)
    ax.set(title="C  Module-size degeneracy",
           xlabel="Number of spectral modules (K)", ylabel="Genes or modules")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(x, summary.eigengap_after_k, color="#e07a5f", marker="o", ms=4)
    ax.axvline(selected_k, color="#d62728", ls="--", lw=1.8)
    ax.set(title="D  Normalized-Laplacian eigengap",
           xlabel="Number of spectral modules (K)",
           ylabel=r"$\lambda_{K+1}-\lambda_K$")

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Empirical selection of spectral attention modules: selected K={selected_k}",
        fontsize=17, fontweight="semibold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"cluster2_attention_spectral_k_sweep.{extension}",
                    dpi=300, facecolor="white")
        fig.savefig(REPRODUCED / f"cluster2_attention_spectral_k_sweep.{extension}",
                    dpi=300, facecolor="white")
    plt.close(fig)

    print(f"raw_silhouette_optimum={raw_k}")
    print(f"selected_stable_nondegenerate={selected_k}")


if __name__ == "__main__":
    main()

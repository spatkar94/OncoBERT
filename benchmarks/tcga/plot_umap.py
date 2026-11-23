"""Publication-style labeled TCGA UMAP for OncoBERT set-Transformer embeddings."""
from __future__ import annotations

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


ROOT = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
# Preserve the original patient-level UMAP visualization used in the revision
# slides.  Retrieval benchmarks remain sample-level; this file is used only for
# the descriptive embedding-space panel in the TCGA composite.
SOURCE = ROOT / "artifacts/slide03_tcga_umap/patient_umap_coordinates.csv.gz"
OUTPUT = SOURCE.parent

FULL_NAMES = {
    "BLCA": "Bladder", "BRCA": "Breast", "HNSC": "Head-neck",
    "LGG": "Lower-grade glioma", "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous", "OV": "Ovarian", "PRAD": "Prostate",
    "SKCM": "Melanoma", "STAD": "Stomach", "THCA": "Thyroid",
    "UCEC": "Endometrial",
}

# Offsets are in points and place labels outside dense regions. Arrows retain
# the connection to each cancer group's robust UMAP center.
LABEL_OFFSETS = {
    "BLCA": (-92, -42), "BRCA": (28, 24), "HNSC": (38, 50),
    "LGG": (0, 28), "LUAD": (6, -64), "LUSC": (-52, 58),
    "OV": (-8, 28), "PRAD": (0, 28), "SKCM": (-28, -28),
    "STAD": (30, -30), "THCA": (-30, 28), "UCEC": (30, -30),
}


def main() -> None:
    frame = pd.read_csv(SOURCE)
    known = frame[frame.cancer_type.ne("Unknown")]
    major = known.cancer_type.value_counts().head(12).index.tolist()
    frame["display_group"] = frame.cancer_type.where(
        frame.cancer_type.isin(major), "Other"
    )

    # A high-contrast qualitative palette, consistent across points, labels,
    # and the compact legend.
    cmap = plt.colormaps["tab20"]
    colors = {group: cmap(index) for index, group in enumerate(major)}
    colors["Other"] = (0.72, 0.72, 0.72, 0.42)

    # Anchor each direct label at the neighborhood most enriched for that
    # cancer type, rather than at a global centroid that can fall in a mixed
    # region. This mirrors the visual logic of labeled cancer islands.
    xy = frame[["UMAP1", "UMAP2"]].to_numpy()
    labels = frame.cancer_type.to_numpy()
    neighbors = NearestNeighbors(n_neighbors=101).fit(xy).kneighbors(
        return_distance=False
    )[:, 1:]
    anchors = {}
    for group in major:
        candidates = np.flatnonzero(labels == group)
        purity = (labels[neighbors[candidates]] == group).mean(axis=1)
        anchors[group] = xy[candidates[np.argmax(purity)]]

    figure, axis = plt.subplots(figsize=(10.2, 9.2))
    random = np.random.default_rng(42)
    # Draw neutral points first and highlighted cancer types afterward.
    for group in ["Other"] + major:
        local = frame[frame.display_group.eq(group)]
        order = random.permutation(len(local))
        axis.scatter(
            local.UMAP1.to_numpy()[order], local.UMAP2.to_numpy()[order],
            s=7 if group != "Other" else 4,
            alpha=.76 if group != "Other" else .22,
            color=colors[group], linewidths=0, rasterized=True, zorder=1,
        )

    for group in major:
        anchor = anchors[group]
        offset = LABEL_OFFSETS.get(group, (0, 25))
        axis.annotate(
            f"{FULL_NAMES.get(group, group)}\n({group})",
            xy=anchor, xycoords="data", xytext=offset,
            textcoords="offset points", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#202020",
            arrowprops={"arrowstyle": "-", "color": colors[group],
                        "linewidth": 1.1, "shrinkA": 2, "shrinkB": 3},
            bbox={"boxstyle": "round,pad=.16", "facecolor": "white",
                  "edgecolor": "none", "alpha": .82},
            zorder=3,
        )

    low = frame[["UMAP1", "UMAP2"]].quantile(.0025).to_numpy()
    high = frame[["UMAP1", "UMAP2"]].quantile(.9975).to_numpy()
    padding = (high - low) * .06
    axis.set_xlim(low[0] - padding[0], high[0] + padding[0])
    axis.set_ylim(low[1] - padding[1], high[1] + padding[1])
    axis.set_xlabel("UMAP 1", fontsize=14.5, labelpad=8)
    axis.set_ylabel("UMAP 2", fontsize=14.5)
    axis.tick_params(axis="both", labelsize=13)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.set_title("TCGA cancer types in OncoBERT embedding space", fontsize=20, pad=15)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=colors[group], markeredgewidth=0,
               label=group)
        for group in major
    ]
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                          markerfacecolor="#b8b8b8", markeredgewidth=0,
                          label="Other"))
    figure.legend(
        handles=handles, title="Cancer types", frameon=False,
        loc="lower center", bbox_to_anchor=(.5, .055), ncol=7,
        fontsize=11, title_fontsize=12, handletextpad=.35,
        columnspacing=1.0,
    )
    figure.text(
        .5, .012,
        f"{len(frame):,} TCGA-CDR matched patients · "
        "12 most frequent cancer types highlighted; remaining types grouped as Other · "
        "same PCA-to-UMAP coordinates as the retrieval analysis",
        ha="center", fontsize=8.5, color="#4a4a4a",
    )
    # Reserve separate vertical bands for the x-axis title, legend, and footer
    # so none of the text collides near the bottom of the panel.
    figure.subplots_adjust(left=.09, right=.98, top=.92, bottom=.205)
    for extension in ["pdf", "png", "svg"]:
        figure.savefig(OUTPUT / f"tcga_cancer_type_umap_labeled.{extension}",
                       dpi=240, bbox_inches="tight")
    plt.close(figure)

    counts = frame.display_group.value_counts().rename_axis("cancer_group").reset_index(name="samples")
    counts.to_csv(OUTPUT / "tcga_cancer_type_umap_labeled_counts.csv", index=False)


if __name__ == "__main__":
    main()

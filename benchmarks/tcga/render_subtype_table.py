"""Render the TCGA molecular-subtype table from saved held-out metrics."""
from __future__ import annotations

from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT = REVISION / "artifacts/slide03_04_tcga_retrieval"
SOURCE = OUTPUT / "molecular_subtype_metrics_by_model_cancer.csv"
MODELS = {
    "nmf": ("NMF", "#9c755f"),
    "gat": ("MutationProjector\n(GAT)", "#59a14f"),
    "set_transformer": ("OncoBERT\n(Set Transformer)", "#4e79a7"),
}


def main() -> None:
    table = pd.read_csv(SOURCE)
    cohort_order = (table[table.model.eq("set_transformer")]
                    .sort_values("balanced_accuracy", ascending=False)
                    .cohort.tolist())
    wide = (table.pivot(index="cohort", columns="model", values="balanced_accuracy")
            .loc[cohort_order, list(MODELS)] * 100)
    subtype_counts = (table[table.model.eq("set_transformer")]
                      .set_index("cohort")["classes"].astype(int))
    random_baseline = (100 / subtype_counts.loc[cohort_order]).rename("random_guessing")
    display_values = wide.copy()
    display_values["random_guessing"] = random_baseline
    export = wide.reset_index().rename(columns={
        "cohort": "Cancer cohort",
        "nmf": "NMF balanced accuracy (%)",
        "gat": "MutationProjector (GAT) balanced accuracy (%)",
        "set_transformer": "OncoBERT (Set Transformer) balanced accuracy (%)",
    })
    export["Random guessing balanced accuracy (%)"] = random_baseline.to_numpy()
    export["Cancer cohort"] = export["Cancer cohort"].map(
        lambda cohort: f"{cohort} ({subtype_counts.loc[cohort]} subtypes)"
    )
    average_row = pd.DataFrame([{
        "Cancer cohort": "Average across cancer types",
        "NMF balanced accuracy (%)": float(wide["nmf"].mean()),
        "MutationProjector (GAT) balanced accuracy (%)": float(wide["gat"].mean()),
        "OncoBERT (Set Transformer) balanced accuracy (%)": float(wide["set_transformer"].mean()),
        "Random guessing balanced accuracy (%)": float(random_baseline.mean()),
    }])
    export = pd.concat([export, average_row], ignore_index=True)
    export.to_csv(OUTPUT / "tcga_molecular_subtype_per_cancer_supplement.csv", index=False)

    # Use a portrait canvas so the dense table remains readable when used as a
    # full-height panel in the composite figure.  The compact title band leaves
    # nearly the entire canvas available for table rows.
    figure, axis = plt.subplots(figsize=(12.4, 16.0))
    axis.axis("off")
    headers = (["Cancer cohort"] + [label for label, _ in MODELS.values()]
               + ["Random\nguessing"])
    cell_text = [
        [f"{cohort} ({subtype_counts.loc[cohort]} subtypes)"]
        + [f"{display_values.loc[cohort, model]:.1f}%" for model in MODELS]
        + [f"{display_values.loc[cohort, 'random_guessing']:.1f}%"]
        for cohort in cohort_order
    ]
    cell_text.append([
        "Average across cancer types",
        *[f"{wide[model].mean():.1f}%" for model in MODELS],
        f"{random_baseline.mean():.1f}%",
    ])
    rendered = axis.table(
        cellText=cell_text, colLabels=headers, cellLoc="center", colLoc="center",
        colWidths=[.25, .14, .22, .25, .14], bbox=[0, .005, 1, .968],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(11.7)
    rendered.scale(1, 1.45)
    header_colors = (["#4c4c4c"] + [color for _, color in MODELS.values()]
                     + ["#777777"])
    for column, color in enumerate(header_colors):
        cell = rendered[(0, column)]
        cell.set_facecolor(color)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")
    for row_index, cohort in enumerate(cohort_order, start=1):
        shade = "#f3f3f3" if row_index % 2 == 0 else "white"
        for column in range(5):
            cell = rendered[(row_index, column)]
            cell.set_facecolor(shade)
            cell.set_edgecolor("#d6d6d6")
            cell.set_linewidth(.6)
        rendered[(row_index, 0)].set_text_props(ha="left")
        values = wide.loc[cohort, list(MODELS)].to_numpy(float)
        for winner in np.flatnonzero(np.isclose(values, values.max(), atol=1e-12)):
            rendered[(row_index, int(winner)+1)].set_text_props(fontweight="bold")
    average_index = len(cohort_order) + 1
    for column in range(5):
        rendered[(average_index, column)].set_facecolor("#e4e4e4")
        rendered[(average_index, column)].set_text_props(fontweight="normal")
        rendered[(average_index, column)].set_edgecolor("#bdbdbd")
    rendered[(average_index, 0)].set_text_props(ha="left", fontweight="bold")
    average_values = np.array([wide[model].mean() for model in MODELS])
    for winner in np.flatnonzero(np.isclose(
        average_values, average_values.max(), atol=1e-12
    )):
        rendered[(average_index, int(winner)+1)].set_text_props(fontweight="bold")
    figure.suptitle("Cancer-specific molecular subtype classification", fontsize=19, y=.995)
    figure.subplots_adjust(left=.025, right=.975, top=.982, bottom=.006)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(OUTPUT / f"tcga_molecular_subtype_per_cancer_supplement.{extension}",
                       dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()

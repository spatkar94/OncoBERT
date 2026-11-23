"""Collect K=2..30 masked NMF reconstruction and compare with neural baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--neural-models", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for rank in range(2, 31):
        record = json.loads((args.input / f"k_{rank:02d}" / "masked_reconstruction_metrics.json").read_text())
        rows.append({"rank": rank, **record["test"]})
    table = pd.DataFrame(rows)
    table.to_csv(args.input / "masked_reconstruction_by_k.csv", index=False)

    mutation_projector = json.loads(
        (args.neural_models / "mutation_projector_gat" / "test_metrics.json").read_text()
    )
    set_transformer = json.loads(
        (args.neural_models / "four_state_set" / "test_metrics.json").read_text()
    )
    best_mse = table.loc[table.masked_reconstruction_mse.idxmin()]
    best_macro = table.loc[table.mutation_macro_auprc.idxmax()]
    best_micro = table.loc[table.mutation_micro_auprc.idxmax()]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    axes[0].plot(table["rank"], table["masked_reconstruction_mse"], "o-", color="#3572b0")
    axes[0].set(title="A  Masked reconstruction error", ylabel="Masked-entry MSE")
    axes[1].plot(table["rank"], table["mutation_macro_auprc"], "o-", color="#55a868", label="NMF")
    axes[1].axhline(mutation_projector["mutation_macro_auprc"], color="#c44e52", linestyle="--",
                    label="MutationProjector (GAT)")
    axes[1].set(title="B  Per-gene mutation retrieval", ylabel="Macro AUPRC")
    axes[2].plot(table["rank"], table["mutation_micro_auprc"], "o-", color="#8172b3", label="NMF")
    axes[2].axhline(mutation_projector["mutation_micro_auprc"], color="#c44e52", linestyle="--",
                    label="MutationProjector (GAT)")
    axes[2].axhline(set_transformer["mutation_micro_auprc"], color="#dd8452", linestyle=":",
                    label="Four-state set Transformer")
    axes[2].set(title="C  Overall mutation retrieval", ylabel="Micro AUPRC")
    for axis in axes:
        axis.set_xlabel("NMF rank K")
        axis.set_xticks(np.arange(2, 31, 4))
        axis.grid(alpha=.2)
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle("Shared-200 · uniform-15% masked-state reconstruction", fontsize=15)
    fig.text(.5, .015,
             "NMF coefficients inferred from the 170 visible genes only; fixed neural validation/test masks reused. "
             "NMF MSE is not directly comparable with neural cross-entropy.", ha="center", fontsize=9)
    fig.subplots_adjust(bottom=.17, top=.86, wspace=.28)
    for extension in ("pdf", "png"):
        fig.savefig(args.input / f"nmf_masked_reconstruction.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "complete": True,
        "minimum_masked_mse_rank": int(best_mse["rank"]),
        "minimum_masked_mse": float(best_mse["masked_reconstruction_mse"]),
        "maximum_macro_auprc_rank": int(best_macro["rank"]),
        "maximum_macro_auprc": float(best_macro["mutation_macro_auprc"]),
        "maximum_micro_auprc_rank": int(best_micro["rank"]),
        "maximum_micro_auprc": float(best_micro["mutation_micro_auprc"]),
        "mutation_projector_macro_auprc": mutation_projector["mutation_macro_auprc"],
        "mutation_projector_micro_auprc": mutation_projector["mutation_micro_auprc"],
        "set_transformer_micro_auprc": set_transformer["mutation_micro_auprc"],
        "comparability_note": "AUPRC uses the same binary masked targets and masks. NMF MSE and neural CE are different loss scales.",
    }
    (args.input / "masked_reconstruction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

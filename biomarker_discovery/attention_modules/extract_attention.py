#!/usr/bin/env python3
"""Extract patient-weighted final-layer gene attention for Cluster 2."""

from __future__ import annotations

import json
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import torch


ORIGINAL = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments"))
REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
RUN = ORIGINAL / "experiments/candidate_three_state_logofunc_shared200_v20"
INTERPRET = ORIGINAL / "experiments/cluster2_mutation_pattern_interpretability_20261103"
CHECKPOINT = RUN / "runs/three_state_seed42/best.pth"
MATCHED = INTERPRET / "matched_patient_cohort.tsv.gz"
OUTPUT = REVISION / "artifacts/cluster2_attention_modules/final_layer"

sys.path.insert(0, str(ORIGINAL))
from oncobert.clean_masked_set import CleanMaskedSetModel  # noqa: E402


@torch.no_grad()
def final_layer_gene_attention(model, states: torch.Tensor) -> torch.Tensor:
    nodes = model.tokenize(states)
    nodes = torch.cat([model.cls_token.expand(len(states), -1, -1), nodes], dim=1)
    final_weights = None
    for block in model.blocks:
        output, weights = block.self_attn(
            nodes, nodes, nodes, need_weights=True, average_attn_weights=False,
        )
        nodes = block.norm1(nodes + block.dropout1(output))
        feed_forward = block.linear2(block.dropout(block.activation(block.linear1(nodes))))
        nodes = block.norm2(nodes + block.dropout2(feed_forward))
        final_weights = weights

    gene = final_weights[:, :, 1:, 1:].float()
    diagonal = torch.eye(states.shape[1], dtype=torch.bool, device=states.device)
    gene = gene.masked_fill(diagonal[None, None], 0)
    gene = gene / gene.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return gene.mean(dim=1)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(MATCHED, sep="\t")
    matched["patient_id"] = matched.patient_id.astype(str)
    cluster2_patients = matched.loc[matched.cluster2.astype(bool), "patient_id"].unique()

    with np.load(RUN / "profiles.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str)
        sample_patients = archive["patients"].astype(str)
    states = np.load(RUN / "functional_states.npy", mmap_mode="r")
    rows = np.flatnonzero(np.isin(sample_patients, cluster2_patients))
    local_patients = sample_patients[rows]
    patient_counts = pd.Series(local_patients).value_counts()
    sample_weights = pd.Series(local_patients).map(1.0 / patient_counts).to_numpy(np.float32)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = CleanMaskedSetModel(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval().requires_grad_(False)

    attention_sum = np.zeros((len(genes), len(genes)), dtype=np.float64)
    batch_size = 64
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch_rows = rows[start:stop]
        batch_states = torch.as_tensor(
            np.asarray(states[batch_rows], np.int64), device=device)
        attention = final_layer_gene_attention(model, batch_states).cpu().numpy()
        attention_sum += np.einsum(
            "b,bij->ij", sample_weights[start:stop], attention)
        if start == 0 or stop == len(rows) or stop % (batch_size * 10) == 0:
            print(f"Final-layer attention: {stop:,}/{len(rows):,} samples", flush=True)

    mean_attention = attention_sum / len(cluster2_patients)
    pd.DataFrame(mean_attention, index=genes, columns=genes).to_csv(
        OUTPUT / "cluster2_final_layer_directed_gene_attention.tsv.gz",
        sep="\t", compression="gzip",
    )
    audit = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "patients": int(len(cluster2_patients)),
        "samples": int(len(rows)),
        "genes": int(len(genes)),
        "layer": "final encoder layer only",
        "heads": int(checkpoint["architecture"]["heads"]),
        "aggregation": "mean across heads; inverse sample-count weighting within patient; equal patient weighting",
        "self_edges": "removed, then each query renormalized across the other 199 genes",
    }
    (OUTPUT / "cluster2_final_layer_attention_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

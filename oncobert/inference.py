#!/usr/bin/env python3
"""Export sample and patient embeddings from a trained OncoBERT Set Transformer."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from oncobert.model import OncoBERT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path, help="Samples x genes .npy matrix")
    parser.add_argument("--metadata", required=True, type=Path, help="CSV with sample_id and patient_id")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()

    states = np.load(args.states, allow_pickle=False)
    metadata = pd.read_csv(args.metadata, dtype=str)
    if not {"sample_id", "patient_id"}.issubset(metadata.columns):
        raise ValueError("Metadata must contain sample_id and patient_id")
    if states.ndim != 2 or states.shape[0] != len(metadata) or not np.isin(states, [0, 1, 2]).all():
        raise ValueError("Expected an aligned samples x genes matrix with values 0, 1, or 2")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    architecture = checkpoint["architecture"]
    if states.shape[1] != architecture["num_genes"]:
        raise ValueError("Input gene count does not match checkpoint")
    model = OncoBERT(**architecture).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    embeddings = np.empty((len(states), architecture["dim"]), dtype=np.float32)
    context = (
        lambda: torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and args.precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode():
        for start in range(0, len(states), args.batch_size):
            values = torch.from_numpy(states[start : start + args.batch_size].astype(np.int64)).to(device)
            with context():
                _, cls = model(values)
            embeddings[start : start + len(cls)] = cls.float().cpu().numpy()

    patients = metadata.patient_id.to_numpy()
    patient_ids, inverse, counts = np.unique(patients, return_inverse=True, return_counts=True)
    sums = np.zeros((len(patient_ids), architecture["dim"]), dtype=np.float64)
    np.add.at(sums, inverse, embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        string_type = h5py.string_dtype("utf-8")
        handle.create_dataset("sample_ids", data=metadata.sample_id.to_numpy(dtype=object), dtype=string_type)
        handle.create_dataset("sample_patients", data=patients.astype(object), dtype=string_type)
        handle.create_dataset("sample_embeddings", data=embeddings, compression="lzf")
        handle.create_dataset("patient_ids", data=patient_ids.astype(object), dtype=string_type)
        handle.create_dataset(
            "patient_embeddings", data=(sums / counts[:, None]).astype(np.float32), compression="lzf"
        )
        handle.create_dataset("patient_n_samples", data=counts)
        handle.create_dataset("genes", data=np.asarray(checkpoint["genes"], dtype=object), dtype=string_type)
        handle.attrs["sample_pooling"] = "unmasked CLS"
        handle.attrs["patient_pooling"] = "mean of sample-level CLS embeddings"
    print(f"Wrote {len(states):,} sample and {len(patient_ids):,} patient embeddings to {args.output}")


if __name__ == "__main__":
    main()

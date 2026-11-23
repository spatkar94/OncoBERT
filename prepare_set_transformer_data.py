#!/usr/bin/env python3
"""Prepare patient-disjoint tensors for three-state Set Transformer pretraining.

The input state matrix must use the manuscript vocabulary:
0 = wild type, 1 = mutated-not-LOF, 2 = LOF. Functional annotation of raw
variants (for example with LoGoFunc) is intentionally upstream of this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_genes(path: Path) -> np.ndarray:
    genes = [line.strip().split(",")[0] for line in path.read_text().splitlines()]
    genes = [gene for gene in genes if gene and gene.lower() not in {"gene", "genes"}]
    if len(genes) != len(set(genes)):
        raise ValueError("Gene vocabulary contains duplicates")
    return np.asarray(genes, dtype=str)


def split_patients(
    patients: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> np.ndarray:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training + validation fractions must be below one")
    unique = np.unique(patients)
    shuffled = np.random.default_rng(seed).permutation(unique)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    patient_split = {
        **{patient: "train" for patient in shuffled[:train_end]},
        **{patient: "validation" for patient in shuffled[train_end:validation_end]},
        **{patient: "test" for patient in shuffled[validation_end:]},
    }
    return np.asarray([patient_split[patient] for patient in patients], dtype=str)


def fixed_masks(rows: np.ndarray, num_genes: int, fraction: float, seed: int):
    count = max(1, int(num_genes * fraction))
    scores = np.random.default_rng(seed).random((len(rows), num_genes))
    masks = np.argsort(np.argsort(scores, axis=1), axis=1) < count
    return {"sample_rows": rows, "masks": masks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, type=Path, help="Samples x genes .npy matrix")
    parser.add_argument("--metadata", required=True, type=Path, help="CSV with sample_id and patient_id")
    parser.add_argument("--genes", required=True, type=Path, help="One ordered gene symbol per line")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    args = parser.parse_args()

    states = np.load(args.states, allow_pickle=False)
    metadata = pd.read_csv(args.metadata, dtype=str)
    genes = read_genes(args.genes)
    required = {"sample_id", "patient_id"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must contain {sorted(required)}")
    if states.ndim != 2 or states.shape != (len(metadata), len(genes)):
        raise ValueError("State matrix must match metadata rows and ordered genes")
    if not np.isin(states, [0, 1, 2]).all():
        raise ValueError("State matrix must contain only 0=WT, 1=mutated-not-LOF, 2=LOF")
    if metadata.sample_id.duplicated().any() or metadata.patient_id.isna().any():
        raise ValueError("Sample IDs must be unique and patient IDs must be present")

    samples = np.asarray(metadata.sample_id, dtype=str)
    patients = np.asarray(metadata.patient_id, dtype=str)
    split = split_patients(
        patients, args.seed, args.train_fraction, args.validation_fraction
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "functional_states.npy", states.astype(np.uint8))
    np.savez_compressed(
        args.output / "profiles.npz",
        samples=samples,
        patients=patients,
        genes=genes,
        split=split,
        mutations=(states != 0),
    )
    for name, offset in (("validation", 100_000), ("test", 200_000)):
        rows = np.flatnonzero(split == name)
        np.savez_compressed(
            args.output / f"{name}_masks.npz",
            **fixed_masks(rows, len(genes), args.mask_fraction, args.seed + offset),
        )
    summary = {
        "samples": len(samples),
        "patients": int(len(np.unique(patients))),
        "genes": len(genes),
        "state_names": ["WILD_TYPE", "MUTATED_NOT_LOF", "LOF"],
        "sample_counts": {name: int((split == name).sum()) for name in ("train", "validation", "test")},
        "patient_counts": {
            name: int(len(np.unique(patients[split == name])))
            for name in ("train", "validation", "test")
        },
        "mask_fraction": args.mask_fraction,
        "split_seed": args.seed,
    }
    (args.output / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

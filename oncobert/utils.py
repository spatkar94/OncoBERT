"""Shared utilities for OncoBERT preparation, training, and inference."""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be 'bf16' or 'fp32'")
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def atomic_json(value, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch_save(value, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(".tmp.pth")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_state_matrix(states: np.ndarray, num_genes: int | None = None) -> None:
    if states.ndim != 2:
        raise ValueError("Expected a samples x genes state matrix")
    if num_genes is not None and states.shape[1] != num_genes:
        raise ValueError(f"Expected {num_genes} genes, received {states.shape[1]}")
    if not np.isin(states, [0, 1, 2]).all():
        raise ValueError("States must be 0=WT, 1=mutated-not-LOF, or 2=LOF")


def load_prepared(root: str | Path):
    """Load the core prepared-data contract and verify patient-disjoint splits."""
    root = Path(root)
    profiles = dict(np.load(root / "profiles.npz", allow_pickle=False))
    states = np.load(root / "functional_states.npy", allow_pickle=False)
    validate_state_matrix(states, len(profiles["genes"]))
    if states.shape != profiles["mutations"].shape:
        raise ValueError("State and profile matrices do not align")
    parts = {
        split: np.flatnonzero(profiles["split"] == split)
        for split in ("train", "validation", "test")
    }
    patient_sets = {
        split: set(profiles["patients"][rows]) for split, rows in parts.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if patient_sets[left] & patient_sets[right]:
            raise ValueError("Prepared splits are not patient-disjoint")
    return profiles, torch.from_numpy(states.astype(np.int64)), parts


def load_fixed_mask(root: str | Path, split: str, rows: np.ndarray) -> torch.Tensor:
    saved = np.load(Path(root) / f"{split}_masks.npz", allow_pickle=False)
    np.testing.assert_array_equal(saved["sample_rows"], rows)
    return torch.from_numpy(saved["masks"].astype(bool))


def patient_mean_embeddings(sample_embeddings: np.ndarray, patients: np.ndarray):
    """Average all sample-level embeddings belonging to each patient."""
    patient_ids, inverse, counts = np.unique(
        patients, return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(patient_ids), sample_embeddings.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, sample_embeddings)
    return patient_ids, (sums / counts[:, None]).astype(np.float32), counts


def write_embedding_h5(
    path: str | Path,
    sample_embeddings: np.ndarray,
    sample_ids: np.ndarray,
    patients: np.ndarray,
    genes: np.ndarray,
    sample_split: np.ndarray | None = None,
) -> None:
    """Write sample CLS embeddings and mean patient representations."""
    patient_ids, patient_embeddings, counts = patient_mean_embeddings(
        sample_embeddings, patients
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        string_type = h5py.string_dtype("utf-8")
        for name, value in (
            ("sample_ids", sample_ids),
            ("sample_patients", patients),
            ("genes", genes),
            ("patient_ids", patient_ids),
        ):
            handle.create_dataset(
                name, data=np.asarray(value, dtype=object), dtype=string_type
            )
        if sample_split is not None:
            handle.create_dataset(
                "sample_split",
                data=np.asarray(sample_split, dtype=object),
                dtype=string_type,
            )
        handle.create_dataset(
            "sample_embeddings", data=sample_embeddings, compression="lzf"
        )
        handle.create_dataset(
            "patient_embeddings", data=patient_embeddings, compression="lzf"
        )
        handle.create_dataset("patient_n_samples", data=counts)
        handle.attrs["sample_pooling"] = "unmasked CLS"
        handle.attrs["patient_pooling"] = "mean sample-level CLS"

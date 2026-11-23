"""Export unmasked sample-level CLS embeddings for the three rank encoders.

The canonical benchmark retained patient means and the final checkpoints.  This
script replays the documented unmasked inference path on the canonical 127,061
sample rank matrices, preserving the exact sample order in ``profiles.npz``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os

import h5py
import numpy as np
import torch

import sys

EXPERIMENTS = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments"))
sys.path.insert(0, str(EXPERIMENTS))
from oncobert.clean_masked_set import CleanMaskedSetModel  # noqa: E402


REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
CANONICAL = EXPERIMENTS / "experiments/final_shared200_five_encoder_benchmark_20261029/canonical_assets"
OUTPUT = REVISION / "artifacts/standardized_sample_level_retrieval/rank_sample_embeddings"
VARIANTS = ("esm2", "genept", "string")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def export_variant(variant: str, device: torch.device, batch_size: int = 512) -> dict:
    destination = OUTPUT / f"rank_{variant}_sample_embeddings.h5"
    checkpoint_path = CANONICAL / f"models/rank_{variant}/best.pth"
    ranks_path = CANONICAL / f"training_data/{variant}_ranks.npy"
    profiles_path = CANONICAL / "training_data/profiles.npz"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CleanMaskedSetModel(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    profiles = np.load(profiles_path, allow_pickle=False)
    ranks = np.load(ranks_path, mmap_mode="r")
    if ranks.shape != (len(profiles["samples"]), 200):
        raise ValueError(f"Unexpected {variant} rank shape: {ranks.shape}")
    vectors = np.empty((len(ranks), checkpoint["architecture"]["dim"]), np.float32)
    for start in range(0, len(ranks), batch_size):
        stop = min(start + batch_size, len(ranks))
        values = torch.from_numpy(np.asarray(ranks[start:stop], dtype=np.int64) - 1).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            _, cls = model(values)
        vectors[start:stop] = cls.float().cpu().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite {variant} sample embeddings")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.h5")
    strings = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.create_dataset("sample_ids", data=profiles["samples"].astype(object), dtype=strings)
        handle.create_dataset("sample_patients", data=profiles["patients"].astype(object), dtype=strings)
        handle.create_dataset("sample_split", data=profiles["split"].astype(object), dtype=strings)
        handle.create_dataset("sample_embeddings", data=vectors, compression="lzf")
        handle.attrs.update(
            variant=variant,
            checkpoint_epoch=int(checkpoint["epoch"]),
            checkpoint_sha256=sha256(checkpoint_path),
            ranks_sha256=sha256(ranks_path),
            profiles_sha256=sha256(profiles_path),
            representation="unmasked sample CLS",
        )
    temporary.replace(destination)
    return {
        "variant": variant,
        "path": str(destination),
        "samples": len(vectors),
        "dimension": vectors.shape[1],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records = [export_variant(variant, device) for variant in VARIANTS]
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "complete": True,
        "device": str(device),
        "records": records,
    }, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

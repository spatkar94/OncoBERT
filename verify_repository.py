#!/usr/bin/env python3
"""Static integrity checks for the public OncoBERT repository."""

from __future__ import annotations

import compileall
import json
from pathlib import Path

import torch

from oncobert.model import OncoBERT


ROOT = Path(__file__).resolve().parent

REQUIRED_PATHS = [
    "oncobert/model.py",
    "oncobert/set_transformer.py",
    "oncobert/training.py",
    "oncobert/inference.py",
    "oncobert/rank_encoder.py",
    "oncobert/train_rank_encoder.py",
    "prepare_set_transformer_data.py",
    "train.py",
    "inference.py",
    "tests/test_set_transformer.py",
    "saved_checkpoints/oncobert_set_transformer_shared200_dim128.pth",
    "data/shared200_genes.txt",
    "baselines/mutation_projector/model.py",
    "baselines/mutation_projector/train_three_state.py",
    "baselines/nmf/train.py",
    "benchmarks/genie_chord_os/build_figure.py",
    "benchmarks/tcga/build_figure.py",
    "benchmarks/nsclc_ici_os/benchmark.py",
    "benchmarks/representation_diagnostics/plot_overlays.py",
    "ablations/rank_vs_set/build_figure.py",
    "ablations/embedding_dimension/build_figure.py",
    "biomarker_discovery/treatment_cluster_interaction/build_figure.py",
    "biomarker_discovery/cross_cancer/build_figure.py",
    "biomarker_discovery/leiden_robustness/build_figure.py",
    "biomarker_discovery/attention_modules/build_figure.py",
    "biomarker_discovery/normalized_laplacian/build_figure.py",
    "data/manifests/figure_workflows.json",
    "data/reference_networks/mutation_projector_network_edges.pt",
]


def main() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

    with (ROOT / "data/manifests/figure_workflows.json").open() as handle:
        workflows = json.load(handle)
    for name, workflow in workflows.items():
        for key in ("script", "reference_output"):
            target = ROOT / workflow[key]
            if not target.is_file():
                raise SystemExit(f"{name}: missing {key}: {target}")

    source_roots = [
        ROOT / "oncobert",
        ROOT / "tests",
        ROOT / "baselines",
        ROOT / "benchmarks",
        ROOT / "ablations",
        ROOT / "biomarker_discovery",
        ROOT / "data",
    ]
    if not all(compileall.compile_dir(path, quiet=1) for path in source_roots):
        raise SystemExit("Python compilation failed")

    for script in (
        "prepare_set_transformer_data.py",
        "train.py",
        "inference.py",
    ):
        if not compileall.compile_file(ROOT / script, quiet=1):
            raise SystemExit(f"Python compilation failed: {script}")

    checkpoint = torch.load(
        ROOT / "saved_checkpoints/oncobert_set_transformer_shared200_dim128.pth",
        map_location="cpu",
        weights_only=True,
    )
    model = OncoBERT(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    genes = (ROOT / "data/shared200_genes.txt").read_text().splitlines()
    if genes != list(checkpoint["genes"]):
        raise SystemExit("Set Transformer checkpoint and shared-200 vocabulary differ")

    print(
        f"Verified the Set Transformer checkpoint, {len(workflows)} figure "
        "workflows, and Python compilation."
    )


if __name__ == "__main__":
    main()

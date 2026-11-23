#!/usr/bin/env python3
"""Pretrain a full-gene diffusion-rank encoder.

Ranks must be precomputed from the complete mutation profile with the desired
weighted ESM2, GenePT, or STRING diffusion network. Supply either one-based
ranks (1..G) or zero-based ranks (0..G-1); the script records the source file
but does not recompute diffusion during pretraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .model import masked_cross_entropy, uniform_mask
from .rank_encoder import RankEncoder
from .utils import (
    atomic_json,
    atomic_torch_save,
    autocast_context,
    load_fixed_mask,
    set_seed,
    write_embedding_h5,
)


def load_inputs(prepared: Path, ranks_path: Path):
    profiles = dict(np.load(prepared / "profiles.npz", allow_pickle=False))
    ranks = np.load(ranks_path, allow_pickle=False)
    num_genes = len(profiles["genes"])
    if ranks.shape != profiles["mutations"].shape:
        raise ValueError("Rank and profile matrices do not align")
    if np.array_equal(
        np.sort(ranks, axis=1),
        np.broadcast_to(np.arange(1, num_genes + 1), ranks.shape),
    ):
        ranks = ranks - 1
    expected = np.broadcast_to(np.arange(num_genes), ranks.shape)
    if not np.array_equal(np.sort(ranks, axis=1), expected):
        raise ValueError("Every row must be a permutation of all gene ranks")
    parts = {
        split: np.flatnonzero(profiles["split"] == split)
        for split in ("train", "validation", "test")
    }
    patient_sets = {
        split: set(profiles["patients"][rows]) for split, rows in parts.items()
    }
    if any(
        patient_sets[left] & patient_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("Prepared splits are not patient-disjoint")
    return profiles, torch.from_numpy(ranks.astype(np.int64)), parts


@torch.inference_mode()
def evaluate(model, values, masks, batch_size, device, precision):
    model.eval()
    total = correct1 = correct5 = correct10 = 0
    absolute_error = squared_error = loss_sum = 0.0
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].to(device)
        mask = masks[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            logits, _ = model(batch, mask)
        selected = logits[mask].float()
        targets = batch[mask]
        prediction = selected.argmax(-1)
        top = selected.topk(min(10, selected.shape[1]), dim=1).indices
        count = len(targets)
        total += count
        loss_sum += torch.nn.functional.cross_entropy(
            selected, targets, reduction="sum"
        ).item()
        correct1 += int((prediction == targets).sum())
        correct5 += int((top[:, :5] == targets[:, None]).any(1).sum())
        correct10 += int((top == targets[:, None]).any(1).sum())
        difference = prediction - targets
        absolute_error += float(difference.abs().sum())
        squared_error += float(difference.square().sum())
    return {
        "reconstruction_ce": loss_sum / total,
        "masked_entries": total,
        "rank_top1": correct1 / total,
        "rank_top5": correct5 / total,
        "rank_top10": correct10 / total,
        "rank_mae": absolute_error / total,
        "rank_rmse": float(np.sqrt(squared_error / total)),
    }


@torch.inference_mode()
def export_embeddings(model, values, profiles, output, batch_size, device, precision):
    model.eval()
    embeddings = np.empty((len(values), model.dim), dtype=np.float32)
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            _, cls = model(batch)
        embeddings[start : start + len(cls)] = cls.float().cpu().numpy()
    write_embedding_h5(
        output / "embeddings.h5",
        embeddings,
        profiles["samples"],
        profiles["patients"],
        profiles["genes"],
        profiles["split"],
    )


def write_history(history, path):
    rows = []
    for record in history:
        rows.append(
            {**{key: value for key, value in record.items() if key != "validation"},
             **{f"validation_{key}": value for key, value in record["validation"].items()}}
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    prepared, output = args.prepared.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles, values, parts = load_inputs(prepared, args.ranks.resolve())
    num_genes = values.shape[1]
    device = torch.device(args.device)
    architecture = {
        "num_genes": num_genes,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }
    config = {
        "architecture": architecture,
        "rank_source": str(args.ranks.resolve()),
        "rank_network": args.network,
        "rank_timing": "precomputed from the complete mutation profile",
        "objective": "unweighted masked-rank cross-entropy",
        "mask_fraction": args.mask_fraction,
        "seed": args.seed,
        "sample_counts": {split: len(rows) for split, rows in parts.items()},
    }
    atomic_json(config, output / "config.json")
    model = RankEncoder(**architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    fixed_masks = {
        split: load_fixed_mask(prepared, split, parts[split])
        for split in ("validation", "test")
    }
    mask_generator = torch.Generator().manual_seed(args.seed + 2)
    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    best, stale, start_epoch, history = float("inf"), 0, 0, []
    if args.resume and (output / "latest.pth").exists():
        checkpoint = torch.load(
            output / "latest.pth", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best = checkpoint["best_validation_ce"]
        stale = checkpoint["stale_epochs"]
        start_epoch = checkpoint["epoch"]
        history = checkpoint["history"]
        mask_generator.set_state(checkpoint["mask_generator_state"].cpu())
        batch_generator.set_state(checkpoint["batch_generator_state"].cpu())

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        began = time.monotonic()
        model.train()
        loss_sum = masked_total = train_correct = 0
        order = parts["train"][
            torch.randperm(len(parts["train"]), generator=batch_generator).numpy()
        ]
        for offset in range(0, len(order), args.batch_size):
            batch = values[order[offset : offset + args.batch_size]]
            mask = uniform_mask(batch, args.mask_fraction, mask_generator).to(device)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                logits, _ = model(batch, mask)
                loss = masked_cross_entropy(logits, batch, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(mask.sum())
            loss_sum += loss.item() * count
            masked_total += count
            train_correct += int((logits[mask].argmax(-1) == batch[mask]).sum())
        validation = evaluate(
            model,
            values[parts["validation"]],
            fixed_masks["validation"],
            args.batch_size,
            device,
            args.precision,
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "architecture": architecture,
            "config": config,
            "genes": profiles["genes"].tolist(),
            "validation": validation,
        }
        if validation["reconstruction_ce"] < best:
            best, stale = validation["reconstruction_ce"], 0
            atomic_torch_save(checkpoint, output / "best.pth")
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train_reconstruction_ce": loss_sum / masked_total,
            "train_top1": train_correct / masked_total,
            "validation": validation,
            "epochs_without_improvement": stale,
            "seconds": time.monotonic() - began,
        }
        history.append(record)
        atomic_torch_save(
            {
                **checkpoint,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_ce": best,
                "stale_epochs": stale,
                "history": history,
                "mask_generator_state": mask_generator.get_state(),
                "batch_generator_state": batch_generator.get_state(),
            },
            output / "latest.pth",
        )
        atomic_json(history, output / "history.json")
        write_history(history, output / "epoch_metrics.csv")
        print(json.dumps(record), flush=True)

    checkpoint = torch.load(
        output / "best.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test = evaluate(
        model,
        values[parts["test"]],
        fixed_masks["test"],
        args.batch_size,
        device,
        args.precision,
    )
    test["best_epoch"] = checkpoint["epoch"]
    atomic_json(test, output / "test_metrics.json")
    export_embeddings(
        model, values, profiles, output, args.batch_size, device, args.precision
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--ranks", required=True, type=Path)
    parser.add_argument("--network", choices=("esm2", "genept", "string"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())

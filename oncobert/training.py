#!/usr/bin/env python3
"""Pretrain the manuscript three-state, position-free OncoBERT Set Transformer."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from oncobert.model import OncoBERT, masked_cross_entropy, uniform_mask


STATE_NAMES = ["WILD_TYPE", "MUTATED_NOT_LOF", "LOF"]


def autocast(device: torch.device, precision: str):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )


def atomic_torch_save(value, path: Path) -> None:
    temporary = path.with_suffix(".tmp.pth")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(value, path: Path) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_prepared(root: Path):
    profiles = dict(np.load(root / "profiles.npz", allow_pickle=False))
    states = np.load(root / "functional_states.npy", allow_pickle=False)
    if states.shape != profiles["mutations"].shape or states.shape[1] != len(profiles["genes"]):
        raise ValueError("Prepared matrix and profile metadata do not align")
    if not np.isin(states, [0, 1, 2]).all():
        raise ValueError("Functional states must use 0=WT, 1=mutated-not-LOF, 2=LOF")
    parts = {
        name: np.flatnonzero(profiles["split"] == name)
        for name in ("train", "validation", "test")
    }
    patient_sets = {name: set(profiles["patients"][rows]) for name, rows in parts.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if patient_sets[left] & patient_sets[right]:
            raise ValueError("Prepared splits are not patient-disjoint")
    return profiles, torch.from_numpy(states.astype(np.int64)), parts


def load_fixed_mask(root: Path, split: str, rows: np.ndarray) -> torch.Tensor:
    saved = np.load(root / f"{split}_masks.npz", allow_pickle=False)
    np.testing.assert_array_equal(saved["sample_rows"], rows)
    return torch.from_numpy(saved["masks"].astype(bool))


@torch.inference_mode()
def evaluate(model, states, masks, batch_size, device, precision):
    model.eval()
    loss_sum = total = correct = 0
    labels, probabilities, gene_ids = [], [], []
    for start in range(0, len(states), batch_size):
        values = states[start : start + batch_size].to(device)
        mask = masks[start : start + batch_size].to(device)
        with autocast(device, precision):
            logits, _ = model(values, mask)
        selected = logits[mask].float()
        targets = values[mask]
        loss_sum += torch.nn.functional.cross_entropy(selected, targets, reduction="sum").item()
        total += len(targets)
        correct += int((selected.argmax(-1) == targets).sum())
        labels.append(targets.cpu().numpy())
        probabilities.append(selected.softmax(-1).cpu().numpy())
        gene_ids.append(torch.nonzero(mask, as_tuple=False)[:, 1].cpu().numpy())
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    gene_ids = np.concatenate(gene_ids)
    prediction = probabilities.argmax(1)
    state_ap = []
    for state_id in range(3):
        binary = labels == state_id
        if 0 < binary.sum() < len(binary):
            state_ap.append(float(average_precision_score(binary, probabilities[:, state_id])))
    mutated = labels != 0
    mutation_probability = 1.0 - probabilities[:, 0]
    per_gene_ap = []
    for gene_id in range(states.shape[1]):
        selected = gene_ids == gene_id
        target = mutated[selected]
        if 0 < target.sum() < len(target):
            per_gene_ap.append(float(average_precision_score(target, mutation_probability[selected])))
    return {
        "reconstruction_ce": loss_sum / total,
        "masked_entries": total,
        "state_top1": correct / total,
        "state_balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "state_macro_auprc": float(np.mean(state_ap)),
        "mutation_micro_auprc": float(average_precision_score(mutated, mutation_probability)),
        "mutation_per_gene_macro_auprc": float(np.mean(per_gene_ap)),
        "genes_with_binary_ap": len(per_gene_ap),
    }


@torch.inference_mode()
def export_embeddings(model, states, profiles, output, batch_size, device, precision):
    model.eval()
    sample_embeddings = np.empty((len(states), model.dim), dtype=np.float32)
    for start in range(0, len(states), batch_size):
        values = states[start : start + batch_size].to(device)
        with autocast(device, precision):
            _, embedding = model(values)
        sample_embeddings[start : start + len(embedding)] = embedding.float().cpu().numpy()
    patient_ids, inverse, counts = np.unique(
        profiles["patients"], return_inverse=True, return_counts=True
    )
    sums = np.zeros((len(patient_ids), model.dim), dtype=np.float64)
    np.add.at(sums, inverse, sample_embeddings)
    with h5py.File(output / "embeddings.h5", "w") as handle:
        string_type = h5py.string_dtype("utf-8")
        for name, value in (
            ("sample_ids", profiles["samples"]),
            ("sample_patients", profiles["patients"]),
            ("sample_split", profiles["split"]),
            ("genes", profiles["genes"]),
            ("patient_ids", patient_ids),
        ):
            handle.create_dataset(name, data=np.asarray(value, dtype=object), dtype=string_type)
        handle.create_dataset("sample_embeddings", data=sample_embeddings, compression="lzf")
        handle.create_dataset(
            "patient_embeddings", data=(sums / counts[:, None]).astype(np.float32), compression="lzf"
        )
        handle.create_dataset("patient_n_samples", data=counts)
        handle.attrs["sample_pooling"] = "unmasked CLS"
        handle.attrs["patient_pooling"] = "mean of sample-level CLS embeddings"
        handle.attrs["representation_dim"] = model.dim


def write_history(history, path: Path) -> None:
    if not history:
        return
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


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root, output = args.prepared.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles, states, parts = load_prepared(root)
    device = torch.device(args.device)
    architecture = {
        "num_classes": 3,
        "num_genes": states.shape[1],
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }
    config = {
        "architecture": architecture,
        "state_names": STATE_NAMES,
        "objective": "unweighted cross-entropy at masked states",
        "mask_fraction": args.mask_fraction,
        "seed": args.seed,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "sample_counts": {name: len(rows) for name, rows in parts.items()},
    }
    atomic_json(config, output / "config.json")
    model = OncoBERT(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    fixed_masks = {
        name: load_fixed_mask(root, name, parts[name]) for name in ("validation", "test")
    }
    mask_generator = torch.Generator().manual_seed(args.seed + 2)
    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    best, stale, start_epoch, history = float("inf"), 0, 0, []
    latest_path = output / "latest.pth"
    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best, stale = checkpoint["best_validation_ce"], checkpoint["stale_epochs"]
        start_epoch, history = checkpoint["epoch"], checkpoint["history"]
        mask_generator.set_state(checkpoint["mask_generator_state"].cpu())
        batch_generator.set_state(checkpoint["batch_generator_state"].cpu())

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if stale >= args.patience:
            break
        began = time.monotonic()
        model.train()
        train_loss = masked_total = train_correct = 0
        order = parts["train"][
            torch.randperm(len(parts["train"]), generator=batch_generator).numpy()
        ]
        for offset in range(0, len(order), args.batch_size):
            batch = states[order[offset : offset + args.batch_size]]
            mask = uniform_mask(batch, args.mask_fraction, mask_generator).to(device)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, args.precision):
                logits, _ = model(batch, mask)
                loss = masked_cross_entropy(logits, batch, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(mask.sum())
            train_loss += float(loss) * count
            masked_total += count
            train_correct += int((logits[mask].argmax(-1) == batch[mask]).sum())
        validation = evaluate(
            model, states[parts["validation"]], fixed_masks["validation"],
            args.batch_size, device, args.precision,
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
            "train_reconstruction_ce": train_loss / masked_total,
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
            latest_path,
        )
        atomic_json(history, output / "history.json")
        write_history(history, output / "epoch_metrics.csv")
        print(json.dumps(record), flush=True)

    checkpoint = torch.load(output / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(
        model, states[parts["test"]], fixed_masks["test"],
        args.batch_size, device, args.precision,
    )
    test["best_epoch"] = checkpoint["epoch"]
    atomic_json(test, output / "test_metrics.json")
    export_embeddings(model, states, profiles, output, args.batch_size, device, args.precision)
    atomic_json(
        {"complete": True, "best_epoch": checkpoint["epoch"], "test": test},
        output / "pretraining_summary.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
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

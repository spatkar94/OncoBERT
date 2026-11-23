"""Train the 10-D, mean-pooled MutationProjector GAT on three functional states."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .masking import masked_ce, uniform_mask
from .model import MutationProjectorBinaryGAT
from .training_utils import amp, atomic_json, checksum, save
from .three_state_utils import evaluate, write_rows


BASE = Path(__file__).resolve().parent
NETWORKS = [
    "GRN_expanded", "E3_expanded", "phosphorylation_expanded",
    "physical_ppi_expanded", "genetic_interaction_expanded", "DDRAM",
    "STRING", "PCNET",
]
STATE_NAMES = ["WILD_TYPE", "MUTATED_NOT_LOF", "LOF"]


def export_embeddings(model, values, data, output, args, device):
    """Mean 200 final 10-D GAT nodes per sample, then mean samples per patient."""
    model.eval()
    samples = np.empty((len(values), args.dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(values), args.batch_size):
            batch = values[start:start + args.batch_size].to(device)
            with amp(device, args.precision):
                _, pooled = model(batch)
            samples[start:start + len(pooled)] = pooled.float().cpu().numpy()
    assert np.isfinite(samples).all()
    patient_ids, inverse, counts = np.unique(
        data["patients"], return_inverse=True, return_counts=True)
    sums = np.zeros((len(patient_ids), args.dim), dtype=np.float64)
    np.add.at(sums, inverse, samples)
    temporary = output / "embeddings.tmp.h5"
    with h5py.File(temporary, "w") as handle:
        string_type = h5py.string_dtype("utf-8")
        for key, value in [
                ("sample_ids", data["samples"]),
                ("sample_patients", data["patients"]),
                ("sample_split", data["split"]),
                ("genes", data["genes"]), ("patient_ids", patient_ids)]:
            handle.create_dataset(key, data=np.asarray(value, dtype=object),
                                  dtype=string_type)
        handle.create_dataset("sample_embeddings", data=samples, compression="lzf")
        handle.create_dataset("patient_embeddings",
                              data=(sums / counts[:, None]).astype(np.float32),
                              compression="lzf")
        handle.create_dataset("patient_n_samples", data=counts)
        handle.attrs.update(
            checkpoint_sha256=checksum(output / "best.pth"),
            model_type="mutation_projector_eight_network_three_state_gat",
            state_names=json.dumps(STATE_NAMES),
            sample_pooling=f"mean_{model.num_genes}_final_GAT_gene_nodes",
            patient_pooling="mean_sample_embeddings",
            representation_dim=args.dim, masked_input=False,
        )
    temporary.replace(output / "embeddings.h5")


def train(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    root = Path(args.prepared).resolve(); output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(root / "profiles.npz", allow_pickle=False))
    states = np.load(root / "functional_states.npy")
    assert states.shape == data["mutations"].shape
    assert np.isin(states, np.arange(len(STATE_NAMES))).all()
    np.testing.assert_array_equal(states > 0, data["mutations"].astype(bool))
    values = torch.from_numpy(states.astype(np.int64))
    parts = {split: np.flatnonzero(data["split"] == split)
             for split in ["train", "validation", "test"]}
    edge_dict = torch.load(root / "network_edges.pt", map_location="cpu", weights_only=True)
    assert list(edge_dict) == NETWORKS
    edges = [edge_dict[name].long().contiguous() for name in NETWORKS]
    architecture = {"num_genes": len(data["genes"]), "dim": args.dim,
                    "blocks": 2, "heads": 1, "d_ff": 100, "dropout": .1,
                    "num_states": len(STATE_NAMES),
                    "vectorized_gene_layers": True,
                    "cache_batched_edges": True}
    config = {
        "prepared": str(root), "output": str(output), "architecture": architecture,
        "state_names": STATE_NAMES, "seed": args.seed, "epochs": args.epochs,
        "patience": args.patience, "batch_size": args.batch_size,
        "lr": 1e-4, "weight_decay": 1e-4,
        "masking": "Uniform15%, exactly 30/200; fixed canonical validation/test masks",
        "objective": "Unweighted three-state masked-token cross entropy",
        "sample_representation": "Mean of 200 final 10-D GAT gene nodes",
        "execution_optimization": "Gene-specific linear/FF layers are vectorized into batched einsum operations; disconnected batched edge indices are cached by batch size and device. Parameterization and graph semantics are unchanged.",
        "patient_representation": "Mean sample embeddings",
        "excluded_inputs": ["cancer type", "TMB", "mutation burden", "panel",
                            "institution", "CNA", "histopathology"],
        "network_order": NETWORKS,
        "input_sha256": {name: checksum(root / name) for name in [
            "profiles.npz", "functional_states.npy", "network_edges.pt",
            "network_audit.json", "validation_masks.npz", "test_masks.npz"]},
        "code_sha256": {name: checksum(BASE / name) for name in [
            "train_three_state.py", "model.py", "masking.py"]},
    }
    atomic_json(config, output / "config.json")
    model = MutationProjectorBinaryGAT(edges, **architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    fixed_masks = {}
    for split in ["validation", "test"]:
        saved = np.load(root / f"{split}_masks.npz")
        np.testing.assert_array_equal(saved["sample_rows"], parts[split])
        fixed_masks[split] = torch.from_numpy(saved["masks"])
        assert torch.all(fixed_masks[split].sum(1) == 30)
    mask_rng = torch.Generator().manual_seed(args.seed + 2)
    batch_rng = torch.Generator().manual_seed(args.seed + 1)
    best, stale, start, history = float("inf"), 0, 0, []
    if (output / "latest.pth").exists():
        checkpoint = torch.load(output / "latest.pth", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, stale, history = (checkpoint["epoch"], checkpoint["best"],
                                       checkpoint["stale"], checkpoint["history"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        mask_rng.set_state(checkpoint["mask_rng"].cpu())
        batch_rng.set_state(checkpoint["batch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    print(json.dumps({"architecture": architecture,
                      "parameters": sum(parameter.numel() for parameter in model.parameters()),
                      "resume_epoch": start}), flush=True)
    invocations = 0
    for epoch in range(start + 1, args.epochs + 1):
        if stale >= args.patience: break
        begun = time.monotonic(); model.train(); loss_sum = 0.; total = correct = 0
        order = parts["train"][torch.randperm(len(parts["train"]), generator=batch_rng).numpy()]
        for step, offset in enumerate(range(0, len(order), args.batch_size)):
            rows = order[offset:offset + args.batch_size]; batch = values[rows]
            mask = uniform_mask(batch, .15, mask_rng).to(device); batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp(device, args.precision):
                logits, _ = model(batch, mask); loss = masked_ce(logits, batch, mask)
            if not torch.isfinite(loss): raise ValueError("Nonfinite reconstruction loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            count = int(mask.sum()); loss_sum += float(loss) * count; total += count
            correct += int((logits[mask].argmax(-1) == batch[mask]).sum())
            if step == 0 or (step + 1) % 200 == 0:
                print(f"epoch{epoch} batch{step + 1} CE={float(loss):.6f}", flush=True)
        validation, per_state, _ = evaluate(
            model, values[parts["validation"]], fixed_masks["validation"],
            STATE_NAMES, args.batch_size, device, args.precision)
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "architecture": architecture, "config": config,
                      "genes": data["genes"].tolist(), "validation": validation}
        if validation["reconstruction_ce"] < best:
            best, stale = validation["reconstruction_ce"], 0
            save(checkpoint, output / "best.pth")
            write_rows(per_state, output / "validation_best_per_state.csv")
        else: stale += 1
        record = {"epoch": epoch, "train_reconstruction_ce": loss_sum / total,
                  "train_top1": correct / total, "validation": validation,
                  "epochs_without_improvement": stale,
                  "seconds": time.monotonic() - begun}
        history.append(record)
        save({**checkpoint, "optimizer": optimizer.state_dict(), "best": best,
              "stale": stale, "history": history,
              "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
              "mask_rng": mask_rng.get_state(), "batch_rng": batch_rng.get_state()},
             output / "latest.pth")
        atomic_json(history, output / "history.json")
        flat = [{key: value for key, value in row.items() if key != "validation"} |
                {"validation_" + key: value for key, value in row["validation"].items()}
                for row in history]
        with (output / "epoch_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        print(json.dumps(record), flush=True)
        invocations += 1
        if args.stop_after and invocations >= args.stop_after: return
    checkpoint = torch.load(output / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test, per_state, prediction = evaluate(
        model, values[parts["test"]], fixed_masks["test"], STATE_NAMES,
        args.batch_size, device, args.precision)
    test["best_epoch"] = int(checkpoint["epoch"])
    atomic_json(test, output / "test_metrics.json"); write_rows(per_state, output / "test_per_state.csv")
    np.savez_compressed(output / "test_predictions.npz", **prediction)
    export_embeddings(model, values, data, output, args, device)
    atomic_json({"complete": True, "epochs_run": len(history),
                 "best_epoch": int(checkpoint["epoch"]),
                 "validation": checkpoint["validation"], "test": test,
                 "early_stopped": stale >= args.patience}, output / "pretraining_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True); parser.add_argument("--output", required=True)
    for name, default in [("seed", 42), ("epochs", 100), ("patience", 20),
                          ("batch-size", 64), ("dim", 10), ("threads", 8),
                          ("stop-after", 0)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    train(parser.parse_args())

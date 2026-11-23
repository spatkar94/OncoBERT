"""Train the controlled shared-200 MutationProjector eight-network GAT model."""
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
from .training_utils import (amp, atomic_json, binary_metrics, checksum,
                             save, write_rows)


BASE = Path(__file__).resolve().parent
NETWORKS = [
    "GRN_expanded", "E3_expanded", "phosphorylation_expanded",
    "physical_ppi_expanded", "genetic_interaction_expanded", "DDRAM",
    "STRING", "PCNET",
]
CODE = ["train_binary.py", "model.py", "masking.py"]


def load_data(root):
    data = dict(np.load(root / "profiles.npz", allow_pickle=False))
    assert data["states"].shape[1] == len(data["genes"])
    assert np.isin(data["states"], [1, 2]).all()
    assert np.array_equal(data["states"] == 2, data["mutations"].astype(bool))
    parts = {split: np.flatnonzero(data["split"] == split)
             for split in ["train", "validation", "test"]}
    patient_sets = {split: set(data["patients"][rows])
                    for split, rows in parts.items()}
    assert all(not patient_sets[left] & patient_sets[right]
               for left, right in [("train", "validation"),
                                   ("train", "test"),
                                   ("validation", "test")])
    values = torch.from_numpy(data["states"].astype(np.int64) - 1)
    edge_dict = torch.load(root / "network_edges.pt", map_location="cpu",
                           weights_only=True)
    assert list(edge_dict) == NETWORKS
    edges = [edge_dict[name].long().contiguous() for name in NETWORKS]
    return data, parts, values, edges


@torch.inference_mode()
def evaluate(model, values, masks, genes, prior, batch_size, device, precision):
    model.eval()
    total, ce = 0, 0.0
    labels, probabilities, gene_indices = [], [], []
    for start in range(0, len(values), batch_size):
        batch = values[start:start + batch_size].to(device)
        mask = masks[start:start + batch_size].to(device)
        with amp(device, precision):
            logits, _ = model(batch, mask)
        selected = logits[mask].float()
        target = batch[mask]
        count = len(target)
        ce += torch.nn.functional.cross_entropy(
            selected, target, reduction="sum").item()
        total += count
        labels.append(target.cpu().numpy().astype(np.uint8))
        probabilities.append(selected.softmax(-1)[:, 1].cpu().numpy())
        gene_indices.append(torch.nonzero(mask, as_tuple=False)[:, 1].cpu().numpy())
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    gene_indices = np.concatenate(gene_indices)
    metrics, per_gene = binary_metrics(labels, probabilities, gene_indices,
                                       genes, prior)
    metrics.update(reconstruction_ce=ce / total, masked_entries=total)
    predictions = {"labels": labels, "mutation_probabilities": probabilities,
                   "gene_indices": gene_indices}
    return metrics, per_gene, predictions


@torch.inference_mode()
def export_embeddings(model, values, data, output, args, device):
    """Export unmasked mean-node sample embeddings and patient means."""
    model.eval()
    samples = np.empty((len(values), args.dim), dtype=np.float32)
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
                ("genes", data["genes"]),
                ("patient_ids", patient_ids)]:
            handle.create_dataset(key, data=np.asarray(value, dtype=object),
                                  dtype=string_type)
        handle.create_dataset("sample_embeddings", data=samples,
                              compression="lzf")
        handle.create_dataset(
            "patient_embeddings",
            data=(sums / counts[:, None]).astype(np.float32), compression="lzf")
        handle.create_dataset("patient_n_samples", data=counts)
        handle.attrs.update(
            checkpoint_sha256=checksum(output / "best.pth"),
            model_type="mutation_projector_eight_network_binary_gat",
            sample_pooling=f"mean_{model.num_genes}_final_GAT_gene_nodes",
            patient_pooling="mean_sample_embeddings",
            representation_dim=args.dim,
            masked_input=False,
        )
    temporary.replace(output / "embeddings.h5")


def train(args):
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    root = Path(args.prepared).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data, parts, values, network_edges = load_data(root)
    num_genes = len(data["genes"])
    masked_genes = max(1, int(num_genes * .15))
    architecture = {
        "num_genes": num_genes, "dim": args.dim, "blocks": 2, "heads": 1,
        "d_ff": 100, "dropout": .1,
    }
    config = {key: value for key, value in vars(args).items()
              if key != "stop_after"}
    inputs = ["profiles.npz", "network_edges.pt", "network_audit.json",
              "validation_masks.npz", "test_masks.npz", "manifest.json"]
    config.update(
        model_type="mutation_projector_eight_network_binary_gat",
        prepared=str(root), output=str(output), architecture=architecture,
        official_network_order=NETWORKS,
        gene_identity=f"Fresh trainable nn.Embedding({num_genes},{args.dim})",
        state_vocabulary={"wild_type": 0, "nonsynonymous": 1,
                          "input_mask_only": 2},
        masking=f"Uniform without replacement; exactly {masked_genes}/{num_genes} states; gene identity retained",
        objective="Standard unweighted binary CE on masked genes only",
        selection="Minimum fixed-mask validation reconstruction CE",
        patient_representation=f"Mean of {num_genes} final GAT node embeddings",
        excluded_inputs=["cancer type", "TMB", "mutation burden", "panel",
                         "institution", "CNA", "histopathology"],
        input_sha256={name: checksum(root / name) for name in inputs},
        code_sha256={name: checksum(BASE / name) for name in CODE},
        sample_counts={split: len(rows) for split, rows in parts.items()},
    )
    config_path = output / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    atomic_json(config, config_path)

    model = MutationProjectorBinaryGAT(network_edges, **architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=1e-4)
    prior = data["mutations"][parts["train"]].mean(0)
    validation_masks = uniform_mask(
        values[parts["validation"]], .15,
        torch.Generator().manual_seed(args.seed + 100000))
    test_masks = uniform_mask(
        values[parts["test"]], .15,
        torch.Generator().manual_seed(args.seed + 200000))
    for split, masks in [("validation", validation_masks), ("test", test_masks)]:
        fixed = np.load(root / f"{split}_masks.npz")
        np.testing.assert_array_equal(fixed["sample_rows"], parts[split])
        np.testing.assert_array_equal(fixed["masks"], masks.numpy())

    mask_rng = torch.Generator().manual_seed(args.seed + 2)
    batch_rng = torch.Generator().manual_seed(args.seed + 1)
    best, stale, start, history = float("inf"), 0, 0, []
    if (output / "latest.pth").exists():
        checkpoint = torch.load(output / "latest.pth", map_location=device,
                                weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, stale = (checkpoint["epoch"], checkpoint["best"],
                              checkpoint["stale"])
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        mask_rng.set_state(checkpoint["mask_rng"].cpu())
        batch_rng.set_state(checkpoint["batch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng"]])
    print(json.dumps({
        "architecture": architecture,
        "networks": {name: int(edge.shape[1])
                     for name, edge in zip(NETWORKS, network_edges)},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "resume_epoch": start,
    }), flush=True)

    invocations = 0
    for epoch in range(start, args.epochs):
        if stale >= args.patience:
            break
        begun = time.monotonic()
        model.train()
        loss_sum, total, correct = 0.0, 0, 0
        order = parts["train"][
            torch.randperm(len(parts["train"]), generator=batch_rng).numpy()]
        for step, offset in enumerate(range(0, len(order), args.batch_size)):
            rows = order[offset:offset + args.batch_size]
            batch = values[rows]
            mask = uniform_mask(batch, .15, mask_rng).to(device)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp(device, args.precision):
                logits, _ = model(batch, mask)
                loss = masked_ce(logits, batch, mask)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(mask.sum())
            loss_sum += loss.item() * count
            total += count
            correct += (logits[mask].argmax(-1) == batch[mask]).sum().item()
            if step == 0 or (step + 1) % 200 == 0:
                print(f"epoch{epoch + 1} batch{step + 1} CE={loss.item():.5f}",
                      flush=True)

        validation, per_gene, _ = evaluate(
            model, values[parts["validation"]], validation_masks,
            data["genes"], prior, args.batch_size, device, args.precision)
        base = {
            "epoch": epoch + 1, "model_state_dict": model.state_dict(),
            "architecture": architecture, "network_names": NETWORKS,
            "config": config, "genes": data["genes"].tolist(),
            "validation": validation,
        }
        if validation["reconstruction_ce"] < best:
            best, stale = validation["reconstruction_ce"], 0
            save(base, output / "best.pth")
            write_rows(per_gene, output / "validation_best_per_gene_ap.csv")
        else:
            stale += 1
        record = {
            "epoch": epoch + 1,
            "train_reconstruction_ce": loss_sum / total,
            "train_top1": correct / total,
            "train_masked_entries": total,
            "validation": validation,
            "epochs_without_improvement": stale,
            "seconds": time.monotonic() - begun,
        }
        history.append(record)
        save({
            **base, "optimizer": optimizer.state_dict(), "best": best,
            "stale": stale, "history": history,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if device.type == "cuda" else []),
            "mask_rng": mask_rng.get_state(), "batch_rng": batch_rng.get_state(),
        }, output / "latest.pth")
        atomic_json(history, output / "history.json")
        flat = [{key: value for key, value in row.items()
                 if key != "validation"} |
                {"validation_" + key: value
                 for key, value in row["validation"].items()}
                for row in history]
        with (output / "epoch_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        print(json.dumps(record), flush=True)
        invocations += 1
        if args.stop_after and invocations >= args.stop_after:
            return

    checkpoint = torch.load(output / "best.pth", map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test, per_gene, predictions = evaluate(
        model, values[parts["test"]], test_masks, data["genes"], prior,
        args.batch_size, device, args.precision)
    test["best_epoch"] = int(checkpoint["epoch"])
    atomic_json(test, output / "test_metrics.json")
    write_rows(per_gene, output / "test_per_gene_ap.csv")
    np.savez_compressed(output / "test_predictions.npz", **predictions)
    export_embeddings(model, values, data, output, args, device)
    atomic_json({
        "complete": True, "epochs_run": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "validation": checkpoint["validation"], "test": test,
        "early_stopped": stale >= args.patience,
    }, output / "pretraining_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output", required=True)
    for name, default in [("seed", 42), ("epochs", 100), ("patience", 20),
                          ("batch-size", 64), ("dim", 128), ("threads", 8),
                          ("stop-after", 0)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    train(parser.parse_args())

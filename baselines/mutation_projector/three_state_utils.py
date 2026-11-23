"""Functional-state masked pretraining with the clean set-Transformer architecture."""
import argparse, csv, json, random, time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix

from .masking import CleanMaskedSetModel, masked_ce, uniform_mask
from .training_utils import amp, atomic_json, checksum, export_embeddings, save


DEFAULT_STATE_NAMES = ["WILD_TYPE", "GOF", "LOF", "UNKNOWN"]


def write_rows(rows, path):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.inference_mode()
def evaluate(model, values, masks, state_names, batch_size, device, precision):
    model.eval(); total = 0; loss_sum = 0.0; labels = []; probabilities = []; genes = []
    for start in range(0, len(values), batch_size):
        batch_values = values[start:start + batch_size].to(device)
        batch_masks = masks[start:start + batch_size].to(device)
        with amp(device, precision):
            logits, _ = model(batch_values, batch_masks)
        selected = logits[batch_masks].float(); target = batch_values[batch_masks]
        loss_sum += torch.nn.functional.cross_entropy(selected, target, reduction="sum").item()
        total += len(target); labels.append(target.cpu().numpy()); probabilities.append(selected.softmax(-1).cpu().numpy())
        genes.append(torch.nonzero(batch_masks, as_tuple=False)[:, 1].cpu().numpy())
    labels = np.concatenate(labels); probabilities = np.concatenate(probabilities); gene_ids = np.concatenate(genes)
    prediction = probabilities.argmax(1)
    rows = []
    for state_id, state in enumerate(state_names):
        binary = labels == state_id; positives = int(binary.sum())
        ap = float(average_precision_score(binary, probabilities[:, state_id])) if 0 < positives < len(binary) else None
        rows.append({"state": state, "masked_targets": positives, "prevalence": float(binary.mean()), "average_precision": ap,
                     "top1_recall": float((prediction[binary] == state_id).mean()) if positives else None})
    supported = [row["average_precision"] for row in rows if row["average_precision"] is not None]
    mutated = labels != 0; mutation_probability = 1.0 - probabilities[:, 0]
    metrics = {
        "reconstruction_ce": loss_sum / total, "masked_entries": total,
        "state_top1": float((prediction == labels).mean()),
        "state_balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "state_macro_auprc": float(np.mean(supported)),
        "mutation_micro_auprc": float(average_precision_score(mutated, mutation_probability)),
        "mutation_prevalence": float(mutated.mean()),
    }
    matrix = confusion_matrix(labels, prediction, labels=np.arange(len(state_names)))
    return metrics, rows, {"labels": labels.astype(np.uint8), "probabilities": probabilities.astype(np.float32), "gene_indices": gene_ids.astype(np.uint16), "confusion": matrix}


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    root = Path(args.prepared).resolve(); output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(root / "profiles.npz", allow_pickle=False)); values_np = np.load(root / "functional_states.npy")
    assert values_np.shape == data["mutations"].shape and values_np.shape[1] == len(data["genes"])
    state_names = list(args.state_names)
    assert len(state_names) >= 2 and state_names[0] == "WILD_TYPE"
    assert len(state_names) == len(set(state_names))
    assert np.isin(values_np, np.arange(len(state_names))).all()
    values = torch.from_numpy(values_np.astype(np.int64)); parts = {split: np.flatnonzero(data["split"] == split) for split in ["train", "validation", "test"]}
    num_genes = len(data["genes"]); masked_genes = max(1, int(num_genes * .15))
    architecture = {"num_classes": len(state_names), "num_genes": num_genes, "dim": args.dim, "layers": 8, "heads": 8, "ff_dim": 2048, "dropout": 0.1}
    config = {"prepared": str(root), "output": str(output), "architecture": architecture, "state_names": state_names,
              "masking": f"Uniform15%, exactly {masked_genes}/{num_genes}; identical seeds/fixed masks as binary comparator", "objective": "Unweighted masked-state CE",
              "seed": args.seed, "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "lr": 1e-4, "weight_decay": 1e-4,
              "input_sha256": {name: checksum(root / name) for name in ["profiles.npz", "functional_states.npy", "validation_masks.npz", "test_masks.npz"]}}
    atomic_json(config, output / "config.json")
    device = torch.device(args.device); model = CleanMaskedSetModel(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    fixed_masks = {}
    for split in ["validation", "test"]:
        saved = np.load(root / f"{split}_masks.npz"); np.testing.assert_array_equal(saved["sample_rows"], parts[split])
        fixed_masks[split] = torch.from_numpy(saved["masks"])
    mask_rng = torch.Generator().manual_seed(args.seed + 2); batch_rng = torch.Generator().manual_seed(args.seed + 1)
    best = float("inf"); stale = start = 0; history = []
    if (output / "latest.pth").exists():
        checkpoint = torch.load(output / "latest.pth", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start = checkpoint["epoch"]
        best = checkpoint["best"]
        stale = checkpoint["stale"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        mask_rng.set_state(checkpoint["mask_rng"].cpu())
        batch_rng.set_state(checkpoint["batch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    print(json.dumps({"condition": "functional_state", "architecture": architecture,
                      "resume_epoch": start,
                      "gene_lookup_trainable": model.gene_embedding.weight.requires_grad}), flush=True)
    invocations = 0
    for epoch in range(start + 1, args.epochs + 1):
        if stale >= args.patience: break
        began = time.monotonic(); model.train(); loss_sum = 0.0; total = correct = 0
        order = parts["train"][torch.randperm(len(parts["train"]), generator=batch_rng).numpy()]
        for step, offset in enumerate(range(0, len(order), args.batch_size)):
            rows = order[offset:offset + args.batch_size]; batch = values[rows]; mask = uniform_mask(batch, 0.15, mask_rng).to(device); batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp(device, args.precision): logits, _ = model(batch, mask); loss = masked_ce(logits, batch, mask)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            n = int(mask.sum()); loss_sum += float(loss) * n; total += n; correct += int((logits[mask].argmax(-1) == batch[mask]).sum())
            if step == 0 or (step + 1) % 200 == 0: print(f"functional epoch{epoch} batch{step+1} CE={float(loss):.6f}", flush=True)
        validation, per_state, _ = evaluate(model, values[parts["validation"]], fixed_masks["validation"], state_names, args.batch_size, device, args.precision)
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(), "architecture": architecture, "config": config, "genes": data["genes"].tolist(), "validation": validation}
        if validation["reconstruction_ce"] < best:
            best = validation["reconstruction_ce"]; stale = 0; save(checkpoint, output / "best.pth"); write_rows(per_state, output / "validation_best_per_state.csv")
        else: stale += 1
        record = {"epoch": epoch, "train_reconstruction_ce": loss_sum / total, "train_top1": correct / total, "validation": validation,
                  "epochs_without_improvement": stale, "seconds": time.monotonic() - began}
        history.append(record)
        save({**checkpoint, "optimizer": optimizer.state_dict(), "best": best,
              "stale": stale, "history": history,
              "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
              "mask_rng": mask_rng.get_state(), "batch_rng": batch_rng.get_state()},
             output / "latest.pth")
        atomic_json(history, output / "history.json")
        flat = [{k: v for k, v in row.items() if k != "validation"} |
                {"validation_" + k: v for k, v in row["validation"].items()}
                for row in history]
        write_rows(flat, output / "epoch_metrics.csv")
        print(json.dumps(record), flush=True)
        invocations += 1
        if args.stop_after and invocations >= args.stop_after:
            return
    checkpoint = torch.load(output / "best.pth", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state_dict"])
    test, per_state, prediction = evaluate(model, values[parts["test"]], fixed_masks["test"], state_names, args.batch_size, device, args.precision)
    test["best_epoch"] = int(checkpoint["epoch"]); atomic_json(test, output / "test_metrics.json"); write_rows(per_state, output / "test_per_state.csv")
    np.savez_compressed(output / "test_predictions.npz", **prediction)
    args.condition = "functional_state"; export_embeddings(model, values, data, output, args, device)
    atomic_json({"complete": True, "epochs_run": len(history), "best_epoch": int(checkpoint["epoch"]), "validation": checkpoint["validation"], "test": test, "early_stopped": stale >= args.patience}, output / "pretraining_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True); parser.add_argument("--output", required=True)
    for name, default in [("seed", 42), ("epochs", 100), ("patience", 20), ("batch-size", 64), ("threads", 8), ("stop-after", 0)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--state-names", nargs="+", default=DEFAULT_STATE_NAMES)
    train(parser.parse_args())

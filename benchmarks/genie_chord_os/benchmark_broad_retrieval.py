"""Standardized sample-level GENIE broad and detailed cancer retrieval.

Rules shared by every panel:
  * one row and one target per biopsy sample;
  * any target represented by fewer than five eligible samples is mapped to
    ``Other`` rather than excluded;
  * the canonical patient-grouped train/test split is preserved;
  * probes are class-balanced linear Ridge classifiers with training-only
    standardization and no encoder fine-tuning;
  * uncertainty is a patient-cluster bootstrap over held-out samples.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler


REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
EXPERIMENTS = Path(os.environ.get("ONCOBERT_EXPERIMENT_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "OncoBERT_experiments")) / "experiments"
CANONICAL = EXPERIMENTS / "final_shared200_five_encoder_benchmark_20261029/canonical_assets"
PROFILES = CANONICAL / "training_data/profiles.npz"
CLINICAL = Path(os.environ.get("ONCOBERT_GENIE_CLINICAL", Path(__file__).resolve().parents[2] / "data" / "raw" / "genie" / "data_clinical_sample.txt"))
SET_EMBEDDINGS = EXPERIMENTS / "candidate_three_state_logofunc_shared200_v20/runs/three_state_seed42/embeddings.h5"
GAT_EMBEDDINGS = EXPERIMENTS / "candidate_mutation_projector_dim10_mean_shared200_v20/runs/mutation_projector_dim10_mean_vectorized_seed42/embeddings.h5"
NMF_MODEL = EXPERIMENTS / "nmf_consensus_k2_30_shared200_v20/masked_reconstruction/k_13/nmf.joblib"
RANK_ROOT = REVISION / "artifacts/standardized_sample_level_retrieval/rank_sample_embeddings"
OUTPUT = REVISION / "artifacts/standardized_sample_level_retrieval/genie"

THREE_MODELS = {
    "NMF": "#9c755f",
    "MutationProjector (GAT)": "#59a14f",
    "OncoBERT (Set Transformer)": "#4e79a7",
}
RANK_MODELS = {
    "ESM2 rank": "#4e79a7",
    "GenePT rank": "#f28e2b",
    "STRING rank": "#59a14f",
    "OncoBERT (Set Transformer)": "#b07aa1",
}
TASKS = {
    "broad": ("CANCER_TYPE", "Broad Cancer Type Retrieval"),
    "detailed": ("ONCOTREE_CODE", "Detailed Cancer Type Retrieval"),
}
TOP_K = (1, 3, 5, 10)
BOOTSTRAPS = 2_000
SEED = 42
INVALID = {"", "unknown", "not collected", "not applicable", "nan", "none"}


def load_h5_samples(path: Path, expected: np.ndarray) -> np.ndarray:
    with h5py.File(path) as handle:
        observed = handle["sample_ids"].asstr()[:]
        if not np.array_equal(observed, expected):
            raise ValueError(f"Sample order mismatch: {path}")
        return handle["sample_embeddings"][:].astype(np.float64)


def prepare_cohort() -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    profiles = np.load(PROFILES, allow_pickle=False)
    samples = profiles["samples"].astype(str)
    clinical = pd.read_csv(CLINICAL, sep="\t", comment="#", dtype=str, low_memory=False)
    clinical = clinical.drop_duplicates("SAMPLE_ID").set_index("SAMPLE_ID").reindex(samples)
    if clinical[["CANCER_TYPE", "ONCOTREE_CODE"]].isna().all(axis=1).any():
        raise ValueError("Some canonical samples have neither target annotation")
    frame = pd.DataFrame({
        "sample_id": samples,
        "patient_id": profiles["patients"].astype(str),
        "split": profiles["split"].astype(str),
        "CANCER_TYPE": clinical["CANCER_TYPE"].to_numpy(object),
        "ONCOTREE_CODE": clinical["ONCOTREE_CODE"].to_numpy(object),
    })
    representations = {
        "NMF": joblib.load(NMF_MODEL).transform(profiles["mutations"].astype(np.float32)).astype(np.float64),
        "MutationProjector (GAT)": load_h5_samples(GAT_EMBEDDINGS, samples),
        "OncoBERT (Set Transformer)": load_h5_samples(SET_EMBEDDINGS, samples),
        "ESM2 rank": load_h5_samples(RANK_ROOT / "rank_esm2_sample_embeddings.h5", samples),
        "GenePT rank": load_h5_samples(RANK_ROOT / "rank_genept_sample_embeddings.h5", samples),
        "STRING rank": load_h5_samples(RANK_ROOT / "rank_string_sample_embeddings.h5", samples),
    }
    return frame, samples, representations


def standardize_labels(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw = frame[column].astype("string").str.strip()
    eligible = raw.notna() & ~raw.str.lower().isin(INVALID)
    counts = raw[eligible].value_counts()
    rare = set(counts[counts < 5].index.astype(str))
    labels = raw[eligible].astype(str).where(~raw[eligible].astype(str).isin(rare), "Other").to_numpy()
    audit = pd.DataFrame({"original_label": counts.index.astype(str), "sample_count": counts.to_numpy()})
    audit["standardized_label"] = audit.original_label.where(audit.sample_count >= 5, "Other")
    return eligible.to_numpy(), labels, audit


def patient_bootstrap(correct: np.ndarray, patients: np.ndarray) -> tuple[float, float]:
    grouped = pd.DataFrame({"patient": patients, "correct": correct.astype(float)}).groupby("patient", sort=False).correct.agg(["sum", "count"])
    numerators, denominators = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(SEED)
    estimates = np.empty(BOOTSTRAPS)
    for start in range(0, BOOTSTRAPS, 100):
        stop = min(start + 100, BOOTSTRAPS)
        draws = rng.integers(0, len(grouped), (stop-start, len(grouped)), dtype=np.int32)
        estimates[start:stop] = numerators[draws].sum(1) / denominators[draws].sum(1)
    return tuple(np.quantile(estimates, [.025, .975]))


def evaluate_task(frame: pd.DataFrame, representations: dict[str, np.ndarray], task: str):
    column, title = TASKS[task]
    eligible, labels, audit = standardize_labels(frame, column)
    local = frame.loc[eligible].reset_index(names="source_row")
    local["label"] = labels
    train = local.split.eq("train").to_numpy()
    test = local.split.eq("test").to_numpy()
    classes = np.sort(np.unique(labels[train]))
    class_index = {value: index for index, value in enumerate(classes)}
    truth = np.array([class_index.get(value, -1) for value in labels[test]])
    rows, prediction_rows = [], []
    for model, full_matrix in representations.items():
        matrix = full_matrix[local.source_row.to_numpy(int)]
        scaler = StandardScaler().fit(matrix[train])
        probe = RidgeClassifier(alpha=1, class_weight="balanced", solver="cholesky").fit(scaler.transform(matrix[train]), labels[train])
        scores = probe.decision_function(scaler.transform(matrix[test]))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        ranking = np.argsort(scores, axis=1)[:, ::-1]
        correct = {k: (truth >= 0) & (ranking[:, :min(k, len(classes))] == truth[:, None]).any(1) for k in TOP_K}
        row = {
            "model": model, "dimension": matrix.shape[1],
            "train_samples": int(train.sum()), "heldout_samples": int(test.sum()),
            "train_patients": int(local.loc[train, "patient_id"].nunique()),
            "heldout_patients": int(local.loc[test, "patient_id"].nunique()),
            "standardized_class_count": int(local.label.nunique()),
            "training_class_count": int(len(classes)),
            "heldout_only_samples": int((truth < 0).sum()),
        }
        for k in TOP_K:
            lo, hi = patient_bootstrap(correct[k], local.loc[test, "patient_id"].to_numpy(str))
            row.update({f"top{k}_accuracy": float(correct[k].mean()), f"top{k}_ci_lower": float(lo), f"top{k}_ci_upper": float(hi)})
        rows.append(row)
        guesses = probe.classes_[ranking[:, 0]]
        for index, source in enumerate(np.flatnonzero(test)):
            prediction_rows.append({
                "model": model, "sample_id": local.sample_id.iloc[source], "patient_id": local.patient_id.iloc[source],
                "label": labels[source], "top1_prediction": guesses[index],
                **{f"top{k}_correct": bool(correct[k][index]) for k in TOP_K},
            })
    metadata = {
        "task": task, "title": title, "target_column": column,
        "eligible_samples": int(len(local)), "eligible_patients": int(local.patient_id.nunique()),
        "original_class_count": int(audit.original_label.nunique()),
        "rare_original_class_count": int((audit.sample_count < 5).sum()),
        "rare_samples_mapped_to_other": int(audit.loc[audit.sample_count < 5, "sample_count"].sum()),
        "standardized_class_count": int(local.label.nunique()), "training_class_count": int(len(classes)),
        "train_samples": int(train.sum()), "heldout_samples": int(test.sum()),
        "heldout_only_labels": sorted(set(labels[test]) - set(labels[train])),
        "protocol": "Sample-level target; labels with <5 eligible samples mapped to Other; canonical patient-grouped split; training-only StandardScaler; class-balanced Ridge alpha=1; 2,000 patient-cluster bootstrap replicates.",
    }
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows), audit, metadata


def plot(results: pd.DataFrame, metadata: dict, models: dict[str, str], destination: Path, stem: str) -> None:
    local = results.set_index("model").loc[list(models)]
    positions = np.arange(len(TOP_K)); width = .78 / len(models)
    figure, axis = plt.subplots(figsize=(11.3, 6.8))
    for index, (model, color) in enumerate(models.items()):
        values = 100 * np.array([local.loc[model, f"top{k}_accuracy"] for k in TOP_K])
        lower = 100 * np.array([local.loc[model, f"top{k}_ci_lower"] for k in TOP_K])
        upper = 100 * np.array([local.loc[model, f"top{k}_ci_upper"] for k in TOP_K])
        bars = axis.bar(positions + (index-(len(models)-1)/2)*width, values, width,
                        yerr=np.vstack([values-lower, upper-values]), capsize=3.5,
                        color=color, label=model, error_kw={"elinewidth":1.1,"capthick":1.1})
        for bar, value, high in zip(bars, values, upper):
            axis.text(bar.get_x()+bar.get_width()/2, high+.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5)
    chance = 100*np.asarray(TOP_K)/metadata["training_class_count"]
    axis.hlines(chance, positions-.43, positions+.43, color="#777777", linestyle="--", linewidth=1.3)
    axis.set_xticks(positions, [f"Top-{k}" for k in TOP_K])
    axis.set_ylabel("Held-out sample retrieval accuracy (%)")
    axis.set_title(f"GENIE {metadata['title']} ({metadata['standardized_class_count']} Classes)", fontsize=16, pad=14)
    axis.set_ylim(0, max(upper.max()+7, 10)); axis.grid(axis="y", alpha=.18)
    axis.spines[["top","right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0],[0],color="#777777",linestyle="--",linewidth=1.3)); labels.append("Random guessing")
    axis.legend(handles, labels, frameon=False, ncol=2, loc="upper left")
    figure.text(.5,.022,
        f"{metadata['train_samples']:,} training and {metadata['heldout_samples']:,} held-out samples; patient-grouped split. "
        f"{metadata['rare_original_class_count']} labels ({metadata['rare_samples_mapped_to_other']} samples) mapped to Other. "
        "Error bars: patient-cluster-bootstrap 95% CIs.", ha="center", fontsize=9, color="#444444")
    figure.subplots_adjust(left=.10,right=.98,top=.88,bottom=.16)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf","png","svg"):
        figure.savefig(destination/f"{stem}.{extension}",dpi=240,bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    frame, _, representations = prepare_cohort()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for task in TASKS:
        destination = OUTPUT / task
        destination.mkdir(parents=True, exist_ok=True)
        results, predictions, audit, metadata = evaluate_task(frame, representations, task)
        results.to_csv(destination/"model_topk_metrics.csv", index=False)
        predictions.to_csv(destination/"heldout_sample_predictions.csv.gz", index=False, compression="gzip")
        audit.to_csv(destination/"label_standardization_audit.csv", index=False)
        plot(results, metadata, THREE_MODELS, destination, f"genie_{task}_three_model_retrieval")
        plot(results, metadata, RANK_MODELS, destination, f"genie_{task}_rank_vs_set_retrieval")
        (destination/"manifest.json").write_text(json.dumps({"complete":True, **metadata, "results":results.to_dict("records")}, indent=2)+"\n")
        summary[task] = metadata
        print(task, json.dumps(metadata, indent=2))
        print(results.to_string(index=False))
    (OUTPUT/"manifest.json").write_text(json.dumps({"complete":True,"tasks":summary},indent=2)+"\n")


if __name__ == "__main__":
    main()

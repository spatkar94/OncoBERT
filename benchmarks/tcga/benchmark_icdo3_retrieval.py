"""Benchmark TCGA ICD-O-3 morphology-code retrieval from frozen representations.

Codes represented by fewer than five matched samples are relabeled ``Other``.
Evaluation is sample-level, but cross-validation and bootstrap resampling are
grouped by patient so multiple aliquots from one patient are never independent.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from benchmark_detailed_retrieval import (
    MODELS, TOP_K, N_SPLITS, BOOTSTRAPS, SEED, load_embeddings,
    decision_matrix, patient_cluster_bootstrap,
)


REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
CLINICAL = Path(os.environ.get("ONCOBERT_TCGA_CLINICAL", Path(__file__).resolve().parents[2] / "data" / "raw" / "tcga_pancan_clinical_data.csv"))
OUTPUT = REVISION / "artifacts/tcga_icdo3_histology_retrieval"
TARGET = "International Classification of Diseases for Oncology, Third Edition ICD-O-3 Histology Code"
MIN_SAMPLES = 5


def load_cohort():
    loaded = {name: load_embeddings(path) for name, (path, _) in MODELS.items()}
    samples, patients, _ = loaded["OncoBERT (Set Transformer)"]
    for name, (local_samples, local_patients, _) in loaded.items():
        if not np.array_equal(local_samples, samples):
            raise ValueError(f"Sample order mismatch for {name}")
        if not np.array_equal(local_patients, patients):
            raise ValueError(f"Patient order mismatch for {name}")

    clinical = pd.read_csv(
        CLINICAL, dtype=str,
        usecols=["Patient ID", "Sample ID", "Cancer Type Detailed", TARGET],
    ).drop_duplicates("Sample ID").set_index("Sample ID")
    sample_keys = np.asarray([value[:15] for value in samples], dtype=str)
    matched = clinical.reindex(sample_keys)
    invalid = {"", "nan", "na", "n/a", "[not available]", "unknown"}
    valid = matched[TARGET].notna() & ~matched[TARGET].astype(str).str.strip().str.lower().isin(invalid)
    source_rows = np.flatnonzero(valid.to_numpy())
    original = matched.loc[valid, TARGET].astype(str).to_numpy()
    counts = pd.Series(original).value_counts()
    rare_codes = set(counts[counts < MIN_SAMPLES].index)
    labels = np.asarray(["Other" if value in rare_codes else value for value in original])
    frame = pd.DataFrame({
        "sample_id": samples[source_rows],
        "patient_id": patients[source_rows],
        "original_icdo3_code": original,
        "benchmark_label": labels,
        "detailed_name": matched.loc[valid, "Cancer Type Detailed"].astype(str).to_numpy(),
        "source_row": source_rows,
    })
    return frame, counts, loaded


def evaluate():
    frame, original_counts, loaded = load_cohort()
    labels = frame.benchmark_label.to_numpy(str)
    patients = frame.patient_id.to_numpy(str)
    classes = np.sort(np.unique(labels))
    class_index = {label: index for index, label in enumerate(classes)}
    truth = np.asarray([class_index[label] for label in labels], dtype=int)
    splits = list(StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED,
    ).split(np.zeros(len(frame)), labels, groups=patients))
    for train, test in splits:
        if set(patients[train]) & set(patients[test]):
            raise ValueError("Patient leakage across folds")

    rows, prediction_rows, label_rows = [], [], []
    for model, (_, _, all_vectors) in loaded.items():
        matrix = all_vectors[frame.source_row.to_numpy(int)]
        scores = np.full((len(frame), len(classes)), np.nan, dtype=np.float64)
        fold_number = np.zeros(len(frame), dtype=np.uint8)
        for fold, (train, test) in enumerate(splits, start=1):
            scaler = StandardScaler().fit(matrix[train])
            classifier = RidgeClassifier(
                alpha=1, class_weight="balanced", solver="cholesky",
            ).fit(scaler.transform(matrix[train]), labels[train])
            scores[test] = decision_matrix(classifier, scaler.transform(matrix[test]), classes)
            fold_number[test] = fold
        if not np.isfinite(scores).all() or np.any(fold_number == 0):
            raise ValueError("Incomplete out-of-fold predictions")
        ranking = np.argsort(scores, axis=1)[:, ::-1]
        correct = {
            k: (ranking[:, :min(k, len(classes))] == truth[:, None]).any(1)
            for k in TOP_K
        }
        row = {
            "model": model,
            "representation_dimension": matrix.shape[1],
            "eligible_samples": len(frame),
            "eligible_patients": frame.patient_id.nunique(),
            "benchmark_label_count": len(classes),
            "retained_icdo3_code_count": len(classes) - int("Other" in classes),
        }
        for k in TOP_K:
            lo, hi = patient_cluster_bootstrap(correct[k], patients)
            row[f"top{k}_accuracy"] = float(correct[k].mean())
            row[f"top{k}_ci_lower"] = float(lo)
            row[f"top{k}_ci_upper"] = float(hi)
            row[f"macro_top{k}_recall"] = float(
                pd.Series(correct[k]).groupby(labels).mean().mean()
            )
        rows.append(row)

        predicted = classes[ranking[:, 0]]
        for position in range(len(frame)):
            prediction_rows.append({
                "model": model,
                "sample_id": frame.sample_id.iloc[position],
                "patient_id": patients[position],
                "fold": int(fold_number[position]),
                "original_icdo3_code": frame.original_icdo3_code.iloc[position],
                "benchmark_label": labels[position],
                "top1_prediction": predicted[position],
                **{f"top{k}_correct": bool(correct[k][position]) for k in TOP_K},
            })
        for label in classes:
            positions = np.flatnonzero(labels == label)
            label_rows.append({
                "model": model,
                "benchmark_label": label,
                "samples": len(positions),
                "patients": frame.iloc[positions].patient_id.nunique(),
                **{f"top{k}_recall": float(correct[k][positions].mean()) for k in TOP_K},
            })

    rare = original_counts[original_counts < MIN_SAMPLES].sort_values()
    metadata = {
        "target": "sample-level TCGA ICD-O-3 histology/morphology code",
        "embedding_samples": int(len(loaded["OncoBERT (Set Transformer)"][0])),
        "eligible_samples": int(len(frame)),
        "eligible_patients": int(frame.patient_id.nunique()),
        "original_icdo3_code_count": int(original_counts.size),
        "rare_code_threshold": f"fewer than {MIN_SAMPLES} matched samples",
        "rare_codes_reclassified_as_other": int(len(rare)),
        "rare_samples_reclassified_as_other": int(rare.sum()),
        "benchmark_label_count": int(len(classes)),
        "retained_icdo3_code_count": int(len(classes)-1),
        "rare_code_counts": rare.to_dict(),
        "patients_crossing_folds": 0,
        "protocol": (
            "Direct sample-level TCGA ICD-O-3 histology code; codes with <5 matched samples -> Other; "
            "five-fold StratifiedGroupKFold by patient; training-fold StandardScaler; class-balanced "
            "Ridge alpha=1; no representation fine-tuning; 2,000 patient-cluster bootstrap CIs"
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows), pd.DataFrame(label_rows), metadata


def plot(results, metadata):
    local = results.set_index("model").loc[list(MODELS)]
    positions = np.arange(len(TOP_K)); width = .23
    figure, axis = plt.subplots(figsize=(11.2, 6.8))
    for index, (model, (_, color)) in enumerate(MODELS.items()):
        values = 100*local.loc[model,[f"top{k}_accuracy" for k in TOP_K]].to_numpy(float)
        lower = 100*np.array([local.loc[model,f"top{k}_ci_lower"] for k in TOP_K])
        upper = 100*np.array([local.loc[model,f"top{k}_ci_upper"] for k in TOP_K])
        bars = axis.bar(
            positions+(index-1)*width, values, width,
            yerr=np.vstack([values-lower,upper-values]), capsize=4,
            color=color, label=model, error_kw={"elinewidth":1.2,"capthick":1.2},
        )
        for bar,value,high in zip(bars,values,upper):
            axis.text(bar.get_x()+bar.get_width()/2,high+.7,f"{value:.1f}",
                      ha="center",va="bottom",fontsize=9)
    chance = 100*np.asarray(TOP_K)/metadata["benchmark_label_count"]
    axis.hlines(chance,positions-.42,positions+.42,color="#777777",linestyle="--",linewidth=1.3)
    axis.set_xticks(positions,[f"Top-{k}" for k in TOP_K])
    axis.set_ylabel("Five-fold out-of-fold sample retrieval accuracy (%)")
    axis.set_title(
        f"TCGA ICD-O-3 Histology Retrieval ({metadata['benchmark_label_count']} Labels)",
        fontsize=16,pad=14,
    )
    axis.set_ylim(0,100*local.top10_ci_upper.max()+7)
    axis.grid(axis="y",alpha=.18);axis.spines[["top","right"]].set_visible(False)
    handles,labels=axis.get_legend_handles_labels()
    handles.append(Line2D([0],[0],color="#777777",linestyle="--",linewidth=1.3));labels.append("Random guessing")
    axis.legend(handles,labels,frameon=False,ncol=2,loc="upper left")
    figure.text(
        .5,.022,
        f"{metadata['eligible_samples']:,} samples from {metadata['eligible_patients']:,} patients; patient-grouped five-fold evaluation. "
        f"{metadata['rare_codes_reclassified_as_other']} codes ({metadata['rare_samples_reclassified_as_other']} samples) with <5 samples were grouped as Other.",
        ha="center",fontsize=9,color="#444444",
    )
    figure.subplots_adjust(left=.10,right=.98,top=.88,bottom=.16)
    for extension in ("pdf","png","svg"):
        figure.savefig(OUTPUT/f"tcga_icdo3_histology_retrieval.{extension}",dpi=240,bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    results,predictions,labels,metadata=evaluate()
    results.to_csv(OUTPUT/"model_topk_metrics.csv",index=False)
    predictions.to_csv(OUTPUT/"out_of_fold_sample_predictions.csv.gz",index=False,compression="gzip")
    labels.to_csv(OUTPUT/"per_label_topk_recall.csv.gz",index=False,compression="gzip")
    plot(results,metadata)
    (OUTPUT/"manifest.json").write_text(json.dumps({
        "complete":True,**metadata,
        "cross_validation":{"folds":N_SPLITS,"group":"patient","seed":SEED},
        "bootstrap":{"unit":"patient cluster","replicates":BOOTSTRAPS,"seed":SEED},
        "results":results.to_dict("records"),
    },indent=2)+"\n")
    print(json.dumps(metadata,indent=2));print(results.to_string(index=False))


if __name__ == "__main__":
    main()

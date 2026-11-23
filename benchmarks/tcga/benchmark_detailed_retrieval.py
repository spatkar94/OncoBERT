"""Benchmark detailed TCGA OncoTree-code retrieval from frozen representations.

The PanCanAtlas ``Oncotree Code`` assigned to each profiled sample is used as
the target. Cross-validation is grouped by patient, preventing multiple tumor
aliquots from one patient from crossing folds. Codes represented by fewer than
five eligible patients are reported and excluded because they cannot support a
five-fold out-of-fold evaluation without a missing training class.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


REVISION = Path(os.environ.get("ONCOBERT_WORK_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ONCOBERT_TCGA_MODEL_ROOT", Path(__file__).resolve().parents[2] / "data" / "derived" / "tcga_mc3_three_state_set_transformer"))
CLINICAL = Path(os.environ.get("ONCOBERT_TCGA_CLINICAL", Path(__file__).resolve().parents[2] / "data" / "raw" / "tcga_pancan_clinical_data.csv"))
SET_EMBEDDINGS = ROOT / "embeddings.h5"
GAT_EMBEDDINGS = ROOT / "molecular_subtype_benchmark/representations/mutation_projector_gat_embeddings.h5"
NMF_EMBEDDINGS = ROOT / "molecular_subtype_benchmark/representations/nmf_k13_embeddings.h5"
OUTPUT = REVISION / "artifacts/tcga_detailed_oncotree_retrieval"

MODELS = {
    "NMF": (NMF_EMBEDDINGS, "#9c755f"),
    "MutationProjector (GAT)": (GAT_EMBEDDINGS, "#59a14f"),
    "OncoBERT (Set Transformer)": (SET_EMBEDDINGS, "#4e79a7"),
}
TOP_K = (1, 3, 5, 10)
N_SPLITS = 5
BOOTSTRAPS = 2_000
SEED = 42


def load_embeddings(path: Path):
    with h5py.File(path) as handle:
        sample_ids = handle["sample_ids"].asstr()[:]
        patient_ids = handle["sample_patients"].asstr()[:]
        vectors = handle["sample_embeddings"][:].astype(np.float64)
    return sample_ids, patient_ids, vectors


def load_cohort():
    loaded = {name: load_embeddings(path) for name, (path, _) in MODELS.items()}
    reference_samples, reference_patients, _ = loaded["OncoBERT (Set Transformer)"]
    for name, (sample_ids, patient_ids, _) in loaded.items():
        if not np.array_equal(sample_ids, reference_samples):
            raise ValueError(f"Sample order mismatch for {name}")
        if not np.array_equal(patient_ids, reference_patients):
            raise ValueError(f"Patient order mismatch for {name}")

    clinical = pd.read_csv(
        CLINICAL, dtype=str,
        usecols=["Patient ID", "Sample ID", "Cancer Type Detailed", "Oncotree Code"],
    ).drop_duplicates("Sample ID").set_index("Sample ID")
    # PanCanAtlas uses TCGA sample IDs without the vial letter (for example
    # TCGA-02-0003-01), whereas MC3 identifies the full sequenced aliquot. The
    # first 15 characters are the shared patient/sample-type identifier.
    vial_ids = np.asarray([value[:15] for value in reference_samples], dtype=str)
    matched = clinical.reindex(vial_ids)
    invalid = {"", "nan", "na", "n/a", "unknown", "not available"}
    valid = (
        matched["Oncotree Code"].notna()
        & ~matched["Oncotree Code"].astype(str).str.strip().str.lower().isin(invalid)
    ).to_numpy()
    frame = pd.DataFrame({
        "sample_id": reference_samples[valid],
        "vial_id": vial_ids[valid],
        "patient_id": reference_patients[valid],
        "oncotree_code": matched["Oncotree Code"].to_numpy(str)[valid],
        "detailed_name": matched["Cancer Type Detailed"].to_numpy(str)[valid],
        "source_row": np.flatnonzero(valid),
    })

    patient_code = frame[["patient_id", "oncotree_code"]].drop_duplicates()
    code_patient_counts = patient_code.groupby("oncotree_code").patient_id.nunique()
    retained_codes = code_patient_counts[code_patient_counts >= N_SPLITS].index
    retained = frame.oncotree_code.isin(retained_codes)
    excluded = frame.loc[~retained].copy()
    frame = frame.loc[retained].reset_index(drop=True)
    return frame, excluded, code_patient_counts, loaded


def decision_matrix(classifier, values, global_classes):
    local = classifier.decision_function(values)
    if local.ndim == 1:
        local = np.column_stack([-local, local])
    if not np.array_equal(classifier.classes_, global_classes):
        raise ValueError("A training fold is missing an eligible OncoTree code")
    return np.asarray(local, dtype=np.float64)


def patient_cluster_bootstrap(correct, patient_ids):
    local = pd.DataFrame({"patient_id": patient_ids, "correct": correct.astype(float)})
    grouped = local.groupby("patient_id", sort=False).correct.agg(["sum", "count"])
    numerators = grouped["sum"].to_numpy(float)
    denominators = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    estimates = np.empty(BOOTSTRAPS, dtype=float)
    for start in range(0, BOOTSTRAPS, 100):
        stop = min(start + 100, BOOTSTRAPS)
        draws = rng.integers(0, len(grouped), size=(stop-start, len(grouped)), dtype=np.int32)
        estimates[start:stop] = numerators[draws].sum(1) / denominators[draws].sum(1)
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def evaluate():
    frame, excluded, code_patient_counts, loaded = load_cohort()
    labels = frame.oncotree_code.to_numpy(str)
    patients = frame.patient_id.to_numpy(str)
    classes = np.sort(np.unique(labels))
    class_index = {label: index for index, label in enumerate(classes)}
    truth = np.asarray([class_index[label] for label in labels], dtype=int)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(frame)), labels, groups=patients))
    for train, test in splits:
        if set(patients[train]) & set(patients[test]):
            raise ValueError("Patient leakage across folds")

    rows, prediction_rows, lineage_rows = [], [], []
    for model, (_, _, all_vectors) in loaded.items():
        matrix = all_vectors[frame.source_row.to_numpy(int)]
        scores = np.full((len(frame), len(classes)), np.nan, dtype=np.float64)
        fold_number = np.zeros(len(frame), dtype=np.uint8)
        for fold, (train, test) in enumerate(splits, start=1):
            scaler = StandardScaler().fit(matrix[train])
            classifier = RidgeClassifier(
                alpha=1, class_weight="balanced", solver="cholesky"
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
            "lineage_count": len(classes),
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
                "oncotree_code": labels[position],
                "detailed_name": frame.detailed_name.iloc[position],
                "top1_prediction": predicted[position],
                **{f"top{k}_correct": bool(correct[k][position]) for k in TOP_K},
            })
        for code in classes:
            positions = np.flatnonzero(labels == code)
            lineage_rows.append({
                "model": model,
                "oncotree_code": code,
                "detailed_name": frame.detailed_name.iloc[positions[0]],
                "samples": len(positions),
                "patients": frame.iloc[positions].patient_id.nunique(),
                **{f"top{k}_recall": float(correct[k][positions].mean()) for k in TOP_K},
            })

    metadata = {
        "target": "sample-level PanCanAtlas Oncotree Code",
        "embedding_samples": int(len(loaded["OncoBERT (Set Transformer)"][0])),
        "eligible_samples_before_frequency_filter": int(len(frame) + len(excluded)),
        "eligible_patients_before_frequency_filter": int(
            pd.concat([frame[["patient_id"]], excluded[["patient_id"]]]).patient_id.nunique()
        ),
        "evaluated_samples": int(len(frame)),
        "evaluated_patients": int(frame.patient_id.nunique()),
        "all_lineage_count": int(code_patient_counts.size),
        "evaluated_lineage_count": int(len(classes)),
        "minimum_patients_per_evaluated_lineage": N_SPLITS,
        "excluded_rare_lineages": code_patient_counts[code_patient_counts < N_SPLITS].sort_values().to_dict(),
        "excluded_rare_samples": int(len(excluded)),
        "patients_crossing_folds": 0,
        "protocol": (
            "Direct sample-level PanCanAtlas Oncotree Code; five-fold StratifiedGroupKFold by patient; "
            "codes require >=5 patients; training-fold StandardScaler; class-balanced Ridge alpha=1; "
            "no representation fine-tuning; 2,000-replicate patient-cluster bootstrap CIs"
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows), pd.DataFrame(lineage_rows), metadata


def plot(results, metadata):
    local = results.set_index("model").loc[list(MODELS)]
    positions = np.arange(len(TOP_K)); width = .23
    figure, axis = plt.subplots(figsize=(11.2, 6.8))
    for index, (model, (_, color)) in enumerate(MODELS.items()):
        values = 100 * local.loc[model, [f"top{k}_accuracy" for k in TOP_K]].to_numpy(float)
        lower = 100 * np.array([local.loc[model, f"top{k}_ci_lower"] for k in TOP_K])
        upper = 100 * np.array([local.loc[model, f"top{k}_ci_upper"] for k in TOP_K])
        bars = axis.bar(
            positions + (index-1)*width, values, width,
            yerr=np.vstack([values-lower, upper-values]), capsize=4,
            color=color, label=model, error_kw={"elinewidth":1.2,"capthick":1.2},
        )
        for bar, value, high in zip(bars, values, upper):
            axis.text(bar.get_x()+bar.get_width()/2, high+.7, f"{value:.1f}",
                      ha="center", va="bottom", fontsize=9)
    chance = 100 * np.asarray(TOP_K) / metadata["evaluated_lineage_count"]
    axis.hlines(chance, positions-.42, positions+.42, color="#777777",
                linestyle="--", linewidth=1.3)
    axis.set_xticks(positions, [f"Top-{k}" for k in TOP_K])
    axis.set_ylabel("Five-fold out-of-fold sample retrieval accuracy (%)")
    axis.set_title(
        f"TCGA Detailed OncoTree Retrieval ({metadata['evaluated_lineage_count']} Lineages)",
        fontsize=16, pad=14,
    )
    axis.set_ylim(0, 100*local.top10_ci_upper.max()+7)
    axis.grid(axis="y", alpha=.18); axis.spines[["top","right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0],[0],color="#777777",linestyle="--",linewidth=1.3))
    labels.append("Random guessing")
    axis.legend(handles, labels, frameon=False, ncol=2, loc="upper left")
    figure.text(
        .5, .022,
        f"{metadata['evaluated_samples']:,} samples from {metadata['evaluated_patients']:,} patients; "
        "patient-grouped five-fold evaluation. Error bars are patient-cluster-bootstrap 95% CIs.",
        ha="center", fontsize=9, color="#444444",
    )
    figure.subplots_adjust(left=.10,right=.98,top=.88,bottom=.16)
    for extension in ("pdf","png","svg"):
        figure.savefig(OUTPUT/f"tcga_detailed_oncotree_retrieval.{extension}",dpi=240,bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results, predictions, lineages, metadata = evaluate()
    results.to_csv(OUTPUT/"model_topk_metrics.csv", index=False)
    predictions.to_csv(OUTPUT/"out_of_fold_sample_predictions.csv.gz", index=False, compression="gzip")
    lineages.to_csv(OUTPUT/"per_lineage_topk_recall.csv.gz", index=False, compression="gzip")
    plot(results, metadata)
    (OUTPUT/"manifest.json").write_text(json.dumps({
        "complete": True, **metadata,
        "cross_validation": {"folds": N_SPLITS, "group": "patient", "seed": SEED},
        "bootstrap": {"unit":"patient cluster", "replicates":BOOTSTRAPS, "seed":SEED},
        "results": results.to_dict("records"),
    }, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

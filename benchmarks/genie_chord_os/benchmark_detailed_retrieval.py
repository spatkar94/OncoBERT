"""Benchmark detailed GENIE OncoTree-lineage retrieval from pretrained embeddings.

Targets are formal ``ONCOTREE_CODE`` values, which retain tissue-specific
distinctions sometimes collapsed by ``CANCER_TYPE_DETAILED`` display names.
The canonical shared-200 train/test partition and representation matrices are
reused without fine-tuning. Patients with more than one OncoTree lineage are
excluded because a single patient-level embedding has no unique target label.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

import anndata as ad
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
FINAL = EXPERIMENTS / "final_shared200_five_encoder_benchmark_20261029"
COVARIATES = FINAL / "canonical_assets/covariates/matched_train_test_covariates.csv.gz"
CLINICAL = Path(os.environ.get("ONCOBERT_GENIE_CLINICAL", Path(__file__).resolve().parents[2] / "data" / "raw" / "genie" / "data_clinical_sample.txt"))
OUTPUT = REVISION / "artifacts/genie_detailed_oncotree_retrieval"

REPRESENTATIONS = {
    "NMF": EXPERIMENTS / "nmf_gat_set_merged_diagnostics_shared200_v20/nmf_projection/nmf_k13/patients.h5ad",
    "MutationProjector (GAT)": EXPERIMENTS / "candidate_mutation_projector_dim10_mean_shared200_v20/evaluation/representation_diagnostics/mutation_projector_gat10_three_state/patients.h5ad",
    "OncoBERT (Set Transformer)": EXPERIMENTS / "candidate_three_state_logofunc_shared200_v20/evaluation/representation/three_state_logofunc/patients.h5ad",
    "ESM2 rank": FINAL / "canonical_assets/representations/rank_esm2/patients.h5ad",
    "GenePT rank": FINAL / "canonical_assets/representations/rank_genept/patients.h5ad",
    "STRING rank": FINAL / "canonical_assets/representations/rank_string/patients.h5ad",
}
THREE_MODELS = ["NMF", "MutationProjector (GAT)", "OncoBERT (Set Transformer)"]
RANK_MODELS = ["ESM2 rank", "GenePT rank", "STRING rank", "OncoBERT (Set Transformer)"]
TOP_K = (1, 3, 5, 10)
BOOTSTRAPS = 2_000
SEED = 42


def canonical_patient_id(value: pd.Series) -> pd.Series:
    """Match GENIE-MSK-P-* clinical identifiers to canonical P-* IDs."""
    return value.str.replace(r"^GENIE-MSK-", "", regex=True)


def build_cohort() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    covariates = pd.read_csv(COVARIATES, index_col="patient_id")
    clinical = pd.read_csv(
        CLINICAL,
        sep="\t",
        comment="#",
        dtype=str,
        low_memory=False,
        usecols=[
            "PATIENT_ID", "CANCER_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE"
        ],
    )
    clinical["patient_id"] = canonical_patient_id(clinical.PATIENT_ID)
    invalid = {"", "unknown", "not collected", "not applicable"}
    clinical = clinical[
        clinical.ONCOTREE_CODE.notna()
        & ~clinical.ONCOTREE_CODE.str.strip().str.lower().isin(invalid)
    ].copy()

    # A code maps to one display name in GENIE v20. Broad cancer type is kept
    # only for descriptive summaries and does not enter model fitting.
    grouped = clinical.groupby("patient_id", sort=False).agg(
        lineage_count=("ONCOTREE_CODE", "nunique"),
        oncotree_code=("ONCOTREE_CODE", lambda x: "|".join(sorted(set(x)))),
        detailed_name=("CANCER_TYPE_DETAILED", lambda x: "|".join(sorted(set(x)))),
        broad_cancer_type=("CANCER_TYPE", lambda x: "|".join(sorted(set(x)))),
    )
    joined = covariates.join(grouped, how="left")
    complete_covariates = joined[["assay", "institution", "log1p_tmb"]].notna().all(axis=1)
    eligible = joined.lineage_count.eq(1) & complete_covariates
    frame = joined.loc[eligible].copy()

    train = frame.split.eq("train")
    test = frame.split.eq("test")
    train_codes = set(frame.loc[train, "oncotree_code"])
    test_codes = set(frame.loc[test, "oncotree_code"])
    unseen = sorted(test_codes - train_codes)
    metadata = {
        "target": "ONCOTREE_CODE",
        "target_description": (
            "Formal detailed OncoTree lineage; tissue-specific codes retained even "
            "when display names are shared"
        ),
        "canonical_shared200_patients": int(len(covariates)),
        "patients_with_valid_lineage": int(joined.lineage_count.notna().sum()),
        "ambiguous_multilineage_patients_excluded": int(joined.lineage_count.gt(1).sum()),
        "eligible_patients": int(len(frame)),
        "train_patients": int(train.sum()),
        "heldout_patients": int(test.sum()),
        "eligible_lineage_count": int(frame.oncotree_code.nunique()),
        "training_lineage_count": int(frame.loc[train, "oncotree_code"].nunique()),
        "heldout_lineage_count": int(frame.loc[test, "oncotree_code"].nunique()),
        "heldout_only_lineages": unseen,
        "heldout_only_patient_count": int(
            (test & frame.oncotree_code.isin(unseen)).sum()
        ),
        "protocol": (
            "Canonical shared-200 split; one unambiguous valid ONCOTREE_CODE per patient; "
            "training-only StandardScaler; class-balanced RidgeClassifier alpha=1, "
            "solver=cholesky; no representation fine-tuning; test-only lineages retained "
            "as incorrect retrievals; paired held-out-patient bootstrap 95% CIs"
        ),
    }
    return frame, clinical, metadata


def load_representation(path: Path, patients: pd.Index) -> tuple[np.ndarray, int]:
    obj = ad.read_h5ad(path)
    positions = pd.Index(obj.obs_names.astype(str)).get_indexer(patients)
    if (positions < 0).any():
        raise ValueError(f"{path}: missing {(positions < 0).sum()} eligible patients")
    matrix = np.asarray(obj.X)[positions].astype(np.float64)
    return matrix, matrix.shape[1]


def paired_bootstrap(correctness: dict[str, dict[int, np.ndarray]]) -> dict:
    n = len(next(iter(next(iter(correctness.values())).values())))
    rng = np.random.default_rng(SEED)
    estimates = {
        model: {k: np.empty(BOOTSTRAPS, dtype=float) for k in TOP_K}
        for model in correctness
    }
    for start in range(0, BOOTSTRAPS, 100):
        stop = min(start + 100, BOOTSTRAPS)
        draws = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        for model in correctness:
            for k in TOP_K:
                estimates[model][k][start:stop] = correctness[model][k][draws].mean(axis=1)
    return estimates


def evaluate(frame: pd.DataFrame, metadata: dict):
    train = frame.split.eq("train").to_numpy()
    test = frame.split.eq("test").to_numpy()
    labels = frame.oncotree_code.astype(str).to_numpy()
    train_classes = np.sort(np.unique(labels[train]))
    class_to_index = {label: index for index, label in enumerate(train_classes)}
    test_truth = np.array([class_to_index.get(value, -1) for value in labels[test]])

    correctness, model_payload = {}, {}
    for model, path in REPRESENTATIONS.items():
        matrix, dimension = load_representation(path, frame.index)
        scaler = StandardScaler().fit(matrix[train])
        classifier = RidgeClassifier(
            alpha=1, class_weight="balanced", solver="cholesky"
        ).fit(scaler.transform(matrix[train]), labels[train])
        np.testing.assert_array_equal(classifier.classes_, train_classes)
        scores = classifier.decision_function(scaler.transform(matrix[test]))
        ranking = np.argsort(scores, axis=1)[:, ::-1]
        correctness[model] = {
            k: (test_truth >= 0)
            & (ranking[:, : min(k, len(train_classes))] == test_truth[:, None]).any(axis=1)
            for k in TOP_K
        }
        model_payload[model] = {
            "dimension": dimension,
            "ranking": ranking,
            "classes": classifier.classes_,
        }
        del matrix, scores

    bootstrap = paired_bootstrap(correctness)
    rows, prediction_rows, per_lineage_rows = [], [], []
    heldout_patients = frame.index[test]
    heldout_codes = labels[test]
    code_to_name = (
        frame[["oncotree_code", "detailed_name"]]
        .drop_duplicates("oncotree_code")
        .set_index("oncotree_code")
        .detailed_name
        .to_dict()
    )
    for model, payload in model_payload.items():
        row = {
            "model": model,
            "representation_dimension": payload["dimension"],
            "train_patients": metadata["train_patients"],
            "heldout_patients": metadata["heldout_patients"],
            "eligible_lineage_count": metadata["eligible_lineage_count"],
            "training_lineage_count": metadata["training_lineage_count"],
        }
        for k in TOP_K:
            values = correctness[model][k]
            lo, hi = np.quantile(bootstrap[model][k], [0.025, 0.975])
            row[f"top{k}_accuracy"] = float(values.mean())
            row[f"top{k}_ci_lower"] = float(lo)
            row[f"top{k}_ci_upper"] = float(hi)
            # Macro top-k recall gives each held-out lineage equal weight.
            by_lineage = pd.Series(values).groupby(heldout_codes).mean()
            row[f"macro_top{k}_recall"] = float(by_lineage.mean())
        rows.append(row)

        predicted = payload["classes"][payload["ranking"][:, 0]]
        for index, patient in enumerate(heldout_patients):
            prediction_rows.append({
                "model": model,
                "patient_id": patient,
                "oncotree_code": heldout_codes[index],
                "detailed_name": code_to_name.get(heldout_codes[index], ""),
                "top1_prediction": predicted[index],
                **{f"top{k}_correct": bool(correctness[model][k][index]) for k in TOP_K},
            })
        for code, indices in pd.Series(np.arange(len(heldout_codes))).groupby(heldout_codes):
            positions = indices.to_numpy()
            per_lineage_rows.append({
                "model": model,
                "oncotree_code": code,
                "detailed_name": code_to_name.get(code, ""),
                "heldout_patients": len(positions),
                **{
                    f"top{k}_recall": float(correctness[model][k][positions].mean())
                    for k in TOP_K
                },
            })

    return (
        pd.DataFrame(rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(per_lineage_rows),
    )


def plot_group(results: pd.DataFrame, models: list[str], filename: str, title: str):
    palette = {
        "NMF": "#9c755f",
        "MutationProjector (GAT)": "#59a14f",
        "OncoBERT (Set Transformer)": "#4e79a7",
        "ESM2 rank": "#4e79a7",
        "GenePT rank": "#f28e2b",
        "STRING rank": "#59a14f",
    }
    local = results.set_index("model").loc[models]
    # Preserve the established purple OncoBERT color in rank-versus-set plots,
    # while retaining blue in the NMF/GAT/OncoBERT comparison.
    if "ESM2 rank" in models:
        palette["OncoBERT (Set Transformer)"] = "#b07aa1"
    positions = np.arange(len(TOP_K), dtype=float)
    width = min(0.24, 0.78 / len(models))
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    figure, axis = plt.subplots(figsize=(11.3, 6.8))
    for offset, model in zip(offsets, models):
        values = 100 * local.loc[model, [f"top{k}_accuracy" for k in TOP_K]].to_numpy(float)
        lower = 100 * np.array([local.loc[model, f"top{k}_ci_lower"] for k in TOP_K])
        upper = 100 * np.array([local.loc[model, f"top{k}_ci_upper"] for k in TOP_K])
        bars = axis.bar(
            positions + offset, values, width,
            yerr=np.vstack([values - lower, upper - values]), capsize=3.5,
            color=palette[model], label=model,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
        )
        for bar, value, high in zip(bars, values, upper):
            axis.text(bar.get_x() + bar.get_width()/2, high + .35, f"{value:.1f}",
                      ha="center", va="bottom", fontsize=8.5)
    chance = 100 * np.asarray(TOP_K) / int(local.training_lineage_count.iloc[0])
    axis.hlines(chance, positions - .43, positions + .43, color="#777777",
                linestyle="--", linewidth=1.3)
    axis.set_xticks(positions, [f"Top-{k}" for k in TOP_K])
    axis.set_ylabel("Held-out retrieval accuracy (%)")
    figure.suptitle(title, fontsize=16, y=.985)
    axis.set_ylim(0, 100 * local.top10_ci_upper.max() + 6)
    axis.grid(axis="y", alpha=.18)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#777777", linestyle="--", linewidth=1.3))
    labels.append("Random guessing")
    axis.legend(handles, labels, frameon=False, ncol=2, loc="upper left")
    figure.text(
        .5, .022,
        "Uncollapsed formal OncoTree codes; class-balanced Ridge probe with training-only standardization. "
        "Error bars are paired patient-bootstrap 95% CIs; test-only lineages remain in the denominator.",
        ha="center", fontsize=9, color="#444444", wrap=True,
    )
    figure.subplots_adjust(left=.10, right=.98, top=.87, bottom=.16)
    for extension in ("pdf", "png", "svg"):
        figure.savefig(OUTPUT / f"{filename}.{extension}", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, _, metadata = build_cohort()
    results, predictions, per_lineage = evaluate(frame, metadata)
    results.to_csv(OUTPUT / "model_topk_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "heldout_patient_predictions.csv.gz", index=False, compression="gzip")
    per_lineage.to_csv(OUTPUT / "per_lineage_topk_recall.csv.gz", index=False, compression="gzip")
    support = (
        frame.groupby(["oncotree_code", "detailed_name", "split"])
        .size().unstack(fill_value=0).reset_index()
    )
    support.to_csv(OUTPUT / "lineage_support.csv", index=False)
    plot_group(
        results, THREE_MODELS, "nmf_gat_oncobert_detailed_oncotree_retrieval",
        f"GENIE Detailed OncoTree Lineage Retrieval ({metadata['eligible_lineage_count']} Lineages)",
    )
    plot_group(
        results, RANK_MODELS, "rank_encoders_oncobert_detailed_oncotree_retrieval",
        f"GENIE Detailed OncoTree Lineage Retrieval ({metadata['eligible_lineage_count']} Lineages)",
    )
    manifest = {
        "complete": True,
        **metadata,
        "bootstrap": {"replicates": BOOTSTRAPS, "seed": SEED,
                      "paired_across_models_and_metrics": True},
        "representations": {model: str(path) for model, path in REPRESENTATIONS.items()},
        "results": results.to_dict("records"),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

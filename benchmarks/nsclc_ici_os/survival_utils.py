"""Benchmark one frozen functional-state representation with raw-value Cox models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmark_clean_chord_os import (ALPHAS, atomic_json, checksum, evaluate,
                                      fit_path, load, safe_concordance, survival)


MODEL = "functional_state_gofcards"


def nested_fit_raw(x, y, outer_train, outer_test, fold):
    train_y, train_x = y[outer_train], x[outer_train]
    inner = StratifiedKFold(3, shuffle=True, random_state=4200 + fold)
    validation = np.empty((3, len(ALPHAS)))
    notes = []
    for index, (fit, val) in enumerate(inner.split(train_x, train_y["event"])):
        model, warnings = fit_path(train_x[fit], train_y[fit])
        notes.extend(warnings)
        for alpha_index, alpha in enumerate(ALPHAS):
            validation[index, alpha_index] = safe_concordance(
                train_y[val], model.predict(train_x[val], alpha=float(alpha)))
    mean = validation.mean(axis=0)
    chosen = int(np.argmax(mean))
    alpha = float(ALPHAS[chosen])
    model, warnings = fit_path(train_x, train_y)
    notes.extend(warnings)
    prediction = model.predict(x[outer_test], alpha=alpha)
    assert np.isfinite(prediction).all()
    return prediction, {
        "selected_alpha": alpha,
        "inner_mean_c": mean.tolist(),
        "inner_fold_c": validation.tolist(),
        "warnings": notes,
        "preprocessing": "none; raw stored values",
    }


def main(root, cancer_index):
    root = root.resolve()
    frame, _ = load(root)
    cancers = json.loads((root / "prepared/audit.json").read_text())["benchmark_cancers"]
    cancer = cancers[cancer_index]
    folds = pd.read_csv(root / "benchmark/fold_assignments.csv")
    np.testing.assert_array_equal(folds.PATIENT_ID, frame.PATIENT_ID)
    local = np.flatnonzero(frame.CANCER_TYPE.eq(cancer))
    cohort = frame.iloc[local].reset_index(drop=True)
    assignments = folds.iloc[local].fold.to_numpy()
    y = survival(cohort)
    x = np.load(root / "representations/functional_state.npy", mmap_mode="r")[local].astype(np.float64)
    assert x.shape == (len(local), 128) and np.isfinite(x).all()
    slug = "".join(character.lower() if character.isalnum() else "_" for character in cancer)
    destination = root / "benchmark/raw/groups" / slug
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for fold in range(5):
        test = np.flatnonzero(assignments == fold)
        train = np.flatnonzero(assignments != fold)
        risk, fit = nested_fit_raw(x, y, train, test, fold)
        metrics = evaluate(y[train], y[test], risk)
        record = {
            "cancer": cancer,
            "model": MODEL,
            "fold": fold,
            "train_patients": len(train),
            "test_patients": len(test),
            "test_deaths": int(y["event"][test].sum()),
            **metrics,
            **fit,
            "fold_sha256": checksum(root / "benchmark/fold_assignments.csv"),
        }
        pd.DataFrame({
            "patient_id": cohort.PATIENT_ID.iloc[test],
            "sample_id": cohort.SAMPLE_ID.iloc[test],
            "cancer": cancer,
            "fold": fold,
            "event": y["event"][test],
            "os_months": y["time"][test],
            "risk": risk,
        }).to_csv(destination / f"{MODEL}_fold{fold}_predictions.csv.gz", index=False)
        atomic_json(record, destination / f"{MODEL}_fold{fold}.json")
        records.append(record)
        print(cancer, MODEL, "fold", fold, "C", metrics["harrell_c"],
              "AUC12", metrics["auc_12"], flush=True)
    pd.DataFrame([
        {key: value for key, value in record.items() if not isinstance(value, (dict, list))}
        for record in records
    ]).to_csv(destination / "fold_metrics.csv", index=False)
    atomic_json({"complete": True, "cancer": cancer, "model": MODEL, "folds": 5,
                 "preprocessing": "none; raw stored values"},
                destination / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cancer-index", type=int, required=True)
    args = parser.parse_args()
    main(args.root, args.cancer_index)

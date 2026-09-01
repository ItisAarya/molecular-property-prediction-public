"""
src/train/train_ml.py

Random Forest baseline on ECFP fingerprints.

Classification datasets are trained one Random Forest per task. Regression datasets
get a single forest on standard-scaled features.

MISSING LABELS
--------------
Tox21 is sparsely measured: a molecule is not tested in every assay, and those
(molecule, task) pairs are stored as NaN by scripts/prep_moleculenet.py.

scikit-learn refuses NaN targets, and rightly so -- there is nothing to learn from a
measurement that was never taken. For each task we therefore fit only on the rows where
that task actually has a label. This is also the chemically correct thing to do: a
compound that was never run through an assay is not evidence of a negative result.

A task can end up with too few usable labels to fit anything (no labelled rows, or only
one class present). Rather than crashing or silently dropping the column -- which would
misalign the prediction matrix that later stacking/ensembling code depends on -- we emit
a constant prediction for that task and record it in the run summary.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA_DIR = "data"
MODELS_DIR = "models"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"

for d in (MODELS_DIR, PRED_DIR, MET_DIR):
    os.makedirs(d, exist_ok=True)

N_TREES_CLS = 500
N_TREES_REG = 800
SEED = 42


def load_npz(ds, split):
    """Return X, y (NaN = never measured) and the SMILES for one split."""
    d = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"))
    y = d["y"].astype(np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return d["X"].astype(np.float32), y, d["smiles"]


def fit_one_task(Xtr, y_task, Xva, Xte, task_idx, ds):
    """
    Fit a Random Forest for a single classification task.

    Returns (p_valid, p_test, info) where the probabilities are 1-D arrays of positive-class
    probability and `info` records what happened -- useful when a task is too sparse to fit.
    """
    labelled = ~np.isnan(y_task)
    n_labelled = int(labelled.sum())
    classes = np.unique(y_task[labelled]) if n_labelled else np.array([])

    # Degenerate cases: nothing to fit. Emit a constant so the prediction matrix keeps
    # its shape, and say so loudly.
    if n_labelled == 0 or len(classes) < 2:
        const = float(classes[0]) if len(classes) == 1 else 0.5
        reason = "no labelled rows" if n_labelled == 0 else f"only one class present ({classes[0]:.0f})"
        print(f"    task {task_idx}: SKIPPED -- {reason}; emitting constant {const}")
        info = {
            "task": task_idx,
            "fitted": False,
            "reason": reason,
            "n_labelled": n_labelled,
            "constant": const,
        }
        return (
            np.full(Xva.shape[0], const, dtype=np.float32),
            np.full(Xte.shape[0], const, dtype=np.float32),
            info,
        )

    rf = RandomForestClassifier(
        n_estimators=N_TREES_CLS,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    # THE FIX: train only on molecules that actually have this measurement.
    rf.fit(Xtr[labelled], y_task[labelled])
    joblib.dump(rf, os.path.join(MODELS_DIR, f"{ds}_rf_task{task_idx}.pkl"))

    # Column 1 is P(y=1); rf.classes_ is sorted so index 1 is the positive class.
    p_va = rf.predict_proba(Xva)[:, 1].astype(np.float32)
    p_te = rf.predict_proba(Xte)[:, 1].astype(np.float32)

    n_pos = int((y_task[labelled] == 1).sum())
    print(
        f"    task {task_idx}: fitted on {n_labelled}/{len(y_task)} molecules "
        f"({n_pos} positive, {n_pos / n_labelled * 100:.1f}%)"
    )
    info = {
        "task": task_idx,
        "fitted": True,
        "n_labelled": n_labelled,
        "n_positive": n_pos,
        "n_dropped_unmeasured": int(len(y_task) - n_labelled),
    }
    return p_va, p_te, info


def train_classification(ds, Xtr, ytr, Xva, Xte):
    """Fit one forest per task; return stacked [N, T] probability matrices."""
    n_tasks = ytr.shape[1]
    probs_va, probs_te, infos = [], [], []

    for t in range(n_tasks):
        p_va, p_te, info = fit_one_task(Xtr, ytr[:, t], Xva, Xte, t, ds)
        probs_va.append(p_va)
        probs_te.append(p_te)
        infos.append(info)

    return np.vstack(probs_va).T, np.vstack(probs_te).T, infos


def train_regression(ds, Xtr, ytr, Xva, Xte):
    """Fit a single forest on scaled features. Rows without a target are dropped."""
    y = ytr[:, 0]
    labelled = ~np.isnan(y)
    if labelled.sum() < len(y):
        print(f"    dropping {int((~labelled).sum())} rows with no target")

    scaler = StandardScaler().fit(Xtr[labelled])
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ds}_rf_scaler.pkl"))

    rf = RandomForestRegressor(n_estimators=N_TREES_REG, random_state=SEED, n_jobs=-1)
    rf.fit(scaler.transform(Xtr[labelled]), y[labelled])
    joblib.dump(rf, os.path.join(MODELS_DIR, f"{ds}_rf.pkl"))
    print(f"    fitted on {int(labelled.sum())}/{len(y)} molecules")

    pred_va = rf.predict(scaler.transform(Xva)).reshape(-1, 1)
    pred_te = rf.predict(scaler.transform(Xte)).reshape(-1, 1)
    infos = [{"task": 0, "fitted": True, "n_labelled": int(labelled.sum())}]
    return pred_va, pred_te, infos


def train_one_dataset(ds):
    Xtr, ytr, _ = load_npz(ds, "train")
    Xva, yva, _ = load_npz(ds, "valid")
    Xte, yte, _ = load_npz(ds, "test")

    print(f"\n=== {ds} (RF) ===")
    cls = is_classification(ds)

    if cls:
        pred_va, pred_te, infos = train_classification(ds, Xtr, ytr, Xva, Xte)
        m_va = cls_metrics(yva, pred_va)
        m_te = cls_metrics(yte, pred_te)
    else:
        pred_va, pred_te, infos = train_regression(ds, Xtr, ytr, Xva, Xte)
        m_va = reg_metrics(yva, pred_va)
        m_te = reg_metrics(yte, pred_te)

    np.save(os.path.join(PRED_DIR, f"{ds}_rf_valid.npy"), pred_va)
    np.save(os.path.join(PRED_DIR, f"{ds}_rf_test.npy"), pred_te)

    pd.DataFrame([m_va]).to_csv(os.path.join(MET_DIR, f"{ds}_rf_valid.csv"), index=False)
    pd.DataFrame([m_te]).to_csv(os.path.join(MET_DIR, f"{ds}_rf_test.csv"), index=False)

    # Record which tasks were actually trainable -- needed when interpreting a macro
    # average that quietly skipped some columns.
    with open(os.path.join(MET_DIR, f"{ds}_rf_fit_report.json"), "w") as f:
        json.dump({"dataset": ds, "tasks": infos}, f, indent=2)

    print(f"  VALID: {m_va}")
    print(f"  TEST:  {m_te}")


def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta.keys():
        train_one_dataset(ds)


if __name__ == "__main__":
    main()

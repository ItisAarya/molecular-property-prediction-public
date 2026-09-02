"""
src/train/train_hybrid.py

Stacking meta-learner over the base models (RF, GNN, transformer).

    python -m src.train.train_hybrid

Writes:
    results/preds/<ds>_hybrid_oof.npy    cross-fitted, for calibration and thresholds
    results/preds/<ds>_hybrid_valid.npy  model selection
    results/preds/<ds>_hybrid_test.npy   final evaluation

WHAT CHANGED, AND WHY
---------------------
The inherited version fitted the meta-learner on the *validation* split and then reported
its validation metrics on those same rows. Fitting and scoring on the same data measures
memorisation. It also consumed the one split that every later stage -- ensemble weights,
calibration, thresholds, model selection -- was already using.

Now the meta-learner is fitted on out-of-fold base predictions over `train`
(src/train/make_oof.py). Those are honest: each was produced by a base model that never
saw that molecule. Three consequences:

  * The meta level gets the whole training set instead of a few hundred validation rows.
    ClinTox's CT_TOX task goes from ~2 usable positives to ~95.
  * `valid` is freed up for what it should be doing: early stopping and model selection.
  * The base predictions the meta-learner learns from carry the same scaffold-shift
    penalty it will face at test time, because the folds are scaffold-disjoint.

CROSS-FITTING
-------------
The meta-learner also emits `hybrid_oof.npy`: a prediction for every training molecule,
each from a meta-learner fitted without it. Calibration and threshold tuning consume that
rather than an in-sample fit.

Without this the leak simply moves up one level. Calibrators consume meta-learner
predictions; on rows the meta-learner was fitted on, those predictions are over-confident,
and a calibrator fitted against them would learn to correct a distortion that does not
exist on unseen data.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.data.splits import load_smiles, load_y, scaffold_kfold_indices
from src.eval.metrics import is_classification, cls_metrics, reg_metrics
from src.utils.seed import set_seed

DATA_DIR = "data"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
BASE_MODELS = ["rf", "gnn", "trf"]
N_FOLDS = 5

os.makedirs(MET_DIR, exist_ok=True)


def load_pred(ds, model, split):
    """Load a base model's predictions for one split; always returns 2-D."""
    p = np.load(os.path.join(PRED_DIR, f"{ds}_{model}_{split}.npy"))
    return p.reshape(-1, 1) if p.ndim == 1 else p


def stack_features(ds, split):
    """Horizontally concatenate every base model's predictions into the meta-features."""
    return np.hstack([load_pred(ds, m, split) for m in BASE_MODELS])


def new_model(cls):
    """A fresh meta-learner. Deliberately simple -- few effective rows, 3*T features."""
    if cls:
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    return Ridge(alpha=1.0)


def fit_predict_task(cls, X_fit, y_fit, X_apply):
    """
    Fit one task's meta-learner and apply it.

    Returns None when the task cannot be fitted (no labelled rows, or a single class), so
    the caller can fall back rather than crash.
    """
    labelled = ~np.isnan(y_fit)
    if labelled.sum() == 0:
        return None
    if cls and len(np.unique(y_fit[labelled])) < 2:
        return None

    model = new_model(cls)
    model.fit(X_fit[labelled], y_fit[labelled])
    return model.predict_proba(X_apply)[:, 1] if cls else model.predict(X_apply)


def run_ds(ds):
    set_seed()
    cls = is_classification(ds)

    X_oof = stack_features(ds, "oof")
    X_va = stack_features(ds, "valid")
    X_te = stack_features(ds, "test")

    y_tr = load_y(ds, "train")
    y_va = load_y(ds, "valid")
    y_te = load_y(ds, "test")
    n_tasks = y_tr.shape[1]

    print(f"\n=== {ds} (HYBRID) ===")
    print(f"  meta-learner fitted on {X_oof.shape[0]} out-of-fold rows "
          f"x {X_oof.shape[1]} base features ({len(BASE_MODELS)} models x {n_tasks} tasks)")

    folds = scaffold_kfold_indices(ds, load_smiles(ds, "train"), n_folds=N_FOLDS)

    P_oof = np.full_like(y_tr, np.nan, dtype=np.float32)
    P_va = np.zeros((X_va.shape[0], n_tasks), dtype=np.float32)
    P_te = np.zeros((X_te.shape[0], n_tasks), dtype=np.float32)
    skipped = []

    for t in range(n_tasks):
        # Cross-fitted predictions over train, for calibration and thresholds downstream.
        for hold in folds:
            fit_idx = np.setdiff1d(np.arange(len(y_tr)), hold)
            out = fit_predict_task(cls, X_oof[fit_idx], y_tr[fit_idx, t], X_oof[hold])
            if out is None:
                out = 0.5 if cls else float(np.nanmean(y_tr[:, t]))
            P_oof[hold, t] = out

        # Final meta-learner: all out-of-fold rows, applied to valid and test together.
        full = fit_predict_task(cls, X_oof, y_tr[:, t], np.vstack([X_va, X_te]))
        if full is None:
            skipped.append(t)
            fallback = 0.5 if cls else float(np.nanmean(y_tr[:, t]))
            P_va[:, t] = fallback
            P_te[:, t] = fallback
        else:
            P_va[:, t] = full[: X_va.shape[0]]
            P_te[:, t] = full[X_va.shape[0]:]

    if skipped:
        print(f"  [warn] tasks {skipped} not fittable; constant fallback used")

    for tag, arr in [("oof", P_oof), ("valid", P_va), ("test", P_te)]:
        np.save(os.path.join(PRED_DIR, f"{ds}_hybrid_{tag}.npy"), arr)

    score = cls_metrics if cls else (lambda a, b: reg_metrics(a, b, ds))
    m_oof, m_va, m_te = score(y_tr, P_oof), score(y_va, P_va), score(y_te, P_te)

    pd.DataFrame([m_va]).to_csv(os.path.join(MET_DIR, f"{ds}_hybrid_valid.csv"), index=False)
    pd.DataFrame([m_te]).to_csv(os.path.join(MET_DIR, f"{ds}_hybrid_test.csv"), index=False)

    key = "auc" if cls else "rmse"
    print(f"  {key}: oof={m_oof[key]:.4f}  valid={m_va[key]:.4f}  test={m_te[key]:.4f}")
    return {"dataset": ds, f"oof_{key}": m_oof[key],
            f"valid_{key}": m_va[key], f"test_{key}": m_te[key]}


def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    rows = [run_ds(ds) for ds in meta]
    pd.DataFrame(rows).to_csv(os.path.join(MET_DIR, "hybrid_summary.csv"), index=False)
    print("\nNote: valid is now used only for selection; the meta-learner never saw it.")


if __name__ == "__main__":
    main()

"""
src/eval/metrics.py

Shared metric functions for classification and regression.

TWO THINGS THIS MODULE IS CAREFUL ABOUT
---------------------------------------

1. MISSING LABELS
   Tox21 is sparsely measured; unmeasured (molecule, task) pairs are NaN. A metric must
   never score a prediction against a label that does not exist, so every metric masks
   NaN before computing anything.

2. REPORTING UNITS FOR REGRESSION
   DeepChem z-scores the ESOL and Lipophilicity labels (mean 0, std 1) before training.
   Training on normalized targets is fine, but an RMSE reported in those units is
   meaningless to a chemist and not comparable to published numbers:

       ESOL           RMSE 0.513 normalized  =  0.513 * 2.0667  =  1.06 logS
       Lipophilicity  RMSE 0.633 normalized  =  0.633 * 1.2110  =  0.77 logD

   The first number looks close to state of the art; the second is around the literature
   average. So `reg_metrics` converts back to the dataset's original chemical units using
   the constants saved by scripts/prep_moleculenet.py, and reports those as `rmse` / `mae`.
   The normalized values are still returned as `rmse_norm` / `mae_norm` for traceability.

   Note that R-squared needs no conversion: it is a ratio of sums of squares, so an affine
   rescaling of both truth and prediction cancels out exactly.
"""

import json
import os

import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
    average_precision_score, r2_score, mean_absolute_error,
)

# Prefer the modern sklearn RMSE API; fall back for older versions.
try:
    from sklearn.metrics import root_mean_squared_error as _rmse

    def _rmse_value(y_true, y_pred):
        return float(_rmse(y_true, y_pred))
except ImportError:
    from sklearn.metrics import mean_squared_error

    def _rmse_value(y_true, y_pred):
        return float(mean_squared_error(y_true, y_pred, squared=False))


DATA_DIR = "data"
CLASSIFICATION_DATASETS = {"tox21", "bbbp", "clintox"}

_META_CACHE = None


def is_classification(dsname):
    return dsname.lower() in CLASSIFICATION_DATASETS


def _meta():
    """Load and cache data/dataset_meta.json."""
    global _META_CACHE
    if _META_CACHE is None:
        path = os.path.join(DATA_DIR, "dataset_meta.json")
        with open(path) as f:
            _META_CACHE = json.load(f)
    return _META_CACHE


def label_scaling(dataset, n_tasks):
    """
    Return (mean, std) arrays of shape (n_tasks,) for converting normalized labels back
    to chemical units. Falls back to the identity (0, 1) when the dataset is unknown or
    its labels were never scaled.
    """
    identity = (np.zeros(n_tasks), np.ones(n_tasks))
    if dataset is None:
        return identity
    try:
        info = _meta()[dataset]
    except (KeyError, FileNotFoundError):
        return identity

    mean = np.asarray(info.get("y_mean", [0.0] * n_tasks), dtype=np.float64)
    std = np.asarray(info.get("y_std", [1.0] * n_tasks), dtype=np.float64)
    if mean.size != n_tasks:  # single shared constant broadcast across tasks
        mean = np.resize(mean, n_tasks)
        std = np.resize(std, n_tasks)
    return mean, std


def _as_2d(a):
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------
def cls_metrics(y_true, y_prob, thr=0.5):
    """
    Macro-averaged classification metrics over tasks, ignoring NaN labels.

    A task is skipped when the labelled rows contain only one class, because AUC is
    undefined there. `n_tasks_scored` records how many actually contributed, so a macro
    average is never mistaken for a complete one.
    """
    y_true = _as_2d(y_true)
    y_prob = _as_2d(y_prob)
    y_pred = (y_prob >= thr).astype(int)

    aucs, aps, accs, bals, f1s = [], [], [], [], []
    for t in range(y_true.shape[1]):
        yt = y_true[:, t]
        mask = ~np.isnan(yt)
        yt = yt[mask]
        yp = y_prob[mask, t]
        yb = y_pred[mask, t]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        aps.append(average_precision_score(yt, yp))
        accs.append(accuracy_score(yt, yb))
        bals.append(balanced_accuracy_score(yt, yb))
        f1s.append(f1_score(yt, yb, zero_division=0))

    def _mean(vals):
        return float(np.nanmean(vals)) if vals else float("nan")

    return {
        "auc": _mean(aucs),
        "ap": _mean(aps),
        "acc": _mean(accs),
        "bal_acc": _mean(bals),
        "f1": _mean(f1s),
        "n_tasks_scored": len(aucs),
    }


# --------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------
def reg_metrics(y_true, y_pred, dataset=None):
    """
    Regression metrics in the dataset's original chemical units.

    `dataset` is the MoleculeNet name (e.g. "esol"); it is used to look up the z-scoring
    constants so the returned rmse / mae are in logS, logD, etc. Pass None if the values
    handed in are already unscaled.

    Returns rmse / mae (chemical units), r2 (scale-invariant), rmse_norm / mae_norm
    (normalized units, for comparison with the pre-fix numbers), and n_scored.
    """
    y_true = _as_2d(y_true)
    y_pred = _as_2d(y_pred)

    mean, std = label_scaling(dataset, y_true.shape[1])

    # Score only entries that carry a real measurement.
    mask = ~np.isnan(y_true)
    if not mask.any():
        nan = float("nan")
        return {"rmse": nan, "mae": nan, "r2": nan,
                "rmse_norm": nan, "mae_norm": nan, "n_scored": 0}

    # Broadcast the per-task constants across rows, then select the measured entries.
    std_full = np.broadcast_to(std, y_true.shape)[mask]
    mean_full = np.broadcast_to(mean, y_true.shape)[mask]

    t_norm, p_norm = y_true[mask], y_pred[mask]
    t_raw = t_norm * std_full + mean_full
    p_raw = p_norm * std_full + mean_full

    return {
        "rmse": _rmse_value(t_raw, p_raw),
        "mae": float(mean_absolute_error(t_raw, p_raw)),
        "r2": float(r2_score(t_raw, p_raw)),
        "rmse_norm": _rmse_value(t_norm, p_norm),
        "mae_norm": float(mean_absolute_error(t_norm, p_norm)),
        "n_scored": int(mask.sum()),
    }

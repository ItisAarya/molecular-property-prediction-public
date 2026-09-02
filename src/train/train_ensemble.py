"""
src/train/train_ensemble.py

Weighted blend of the strongest base models.

    python -m src.train.train_ensemble

Writes results/preds/<ds>_ens_{oof,valid,test}.npy and the chosen weights to
results/metrics/<ds>_ens_weights.json.

WHAT CHANGED, AND WHY
---------------------
The inherited version ranked the candidate models on the validation split and then
grid-searched the blend weights on that same split -- while calibration, threshold tuning
and final model selection were all also using it. Weights chosen by maximising a score on
a few hundred rows, then reported on those rows, are fitted parameters masquerading as an
evaluation.

Both the ranking and the weight search now run on out-of-fold predictions over `train`
(src/train/make_oof.py), so `valid` is untouched and available for model selection.

The blend is also cross-fitted: `ens_oof.npy` holds a prediction for every training
molecule produced by weights chosen without it, which is what calibration and thresholds
consume. Weights are only two or three numbers, so the effect is much smaller than for the
meta-learner -- but the whole point of this phase is that no stage is fitted and scored on
the same rows, and a two-parameter fit is still a fit.
"""

import json
import os
from itertools import product

import numpy as np

from src.data.splits import load_smiles, load_y, scaffold_kfold_indices
from src.eval.metrics import is_classification, cls_metrics, reg_metrics

PRED = "results/preds"
MET = "results/metrics"
DATA = "data"

CANDIDATES = ["rf", "gnn", "trf", "hybrid"]
TOPK = 2
GRID_STEP = 0.10
N_FOLDS = 5

os.makedirs(MET, exist_ok=True)


def load_pred(ds, model, split):
    """Prefer a calibrated file when one exists, else the raw predictions."""
    for name in (f"{ds}_{model}_{split}_cal.npy", f"{ds}_{model}_{split}.npy"):
        path = os.path.join(PRED, name)
        if os.path.exists(path):
            p = np.load(path)
            return p.reshape(-1, 1) if p.ndim == 1 else p
    return None


def available(ds):
    return [m for m in CANDIDATES
            if all(load_pred(ds, m, s) is not None for s in ("oof", "valid", "test"))]


def score_of(y, p, ds, cls):
    """Higher is always better, so the same comparison works for both task types."""
    return cls_metrics(y, p)["auc"] if cls else -reg_metrics(y, p, ds)["rmse"]


def weight_grid(n, step=GRID_STEP):
    """All weight vectors on the simplex, at the given resolution."""
    if n == 1:
        yield np.array([1.0])
        return
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    for w in product(vals, repeat=n):
        if abs(sum(w) - 1.0) < 1e-9:
            yield np.array(w)


def blend(w, arrays):
    out = np.zeros_like(arrays[0], dtype=np.float64)
    for wi, a in zip(w, arrays):
        out += wi * a
    return out


def best_weights(y, preds, ds, cls, rows=None):
    """Grid-search the blend weights that maximise the score on `rows` (default: all)."""
    sel = slice(None) if rows is None else rows
    subset = [p[sel] for p in preds]
    y_sub = y[sel]

    best_w, best_s = None, -np.inf
    for w in weight_grid(len(preds)):
        s = score_of(y_sub, blend(w, subset), ds, cls)
        if s > best_s:
            best_w, best_s = w, s
    return best_w


def run_one(ds):
    cls = is_classification(ds)
    models = available(ds)
    if not models:
        print(f"{ds}: ensemble skipped (no base predictions)")
        return

    y_tr = load_y(ds, "train")
    y_va = load_y(ds, "valid")
    y_te = load_y(ds, "test")

    # Rank candidates on out-of-fold performance, not on validation.
    ranked = sorted(models, key=lambda m: score_of(y_tr, load_pred(ds, m, "oof"), ds, cls),
                    reverse=True)
    chosen = ranked[:max(1, min(TOPK, len(ranked)))]

    oof = [load_pred(ds, m, "oof") for m in chosen]
    va = [load_pred(ds, m, "valid") for m in chosen]
    te = [load_pred(ds, m, "test") for m in chosen]

    # Final weights: all out-of-fold rows.
    w = best_weights(y_tr, oof, ds, cls)

    # Cross-fitted blend over train, so calibration sees weights chosen without those rows.
    folds = scaffold_kfold_indices(ds, load_smiles(ds, "train"), n_folds=N_FOLDS)
    P_oof = np.zeros_like(oof[0], dtype=np.float64)
    for hold in folds:
        fit_idx = np.setdiff1d(np.arange(len(y_tr)), hold)
        w_fold = best_weights(y_tr, oof, ds, cls, rows=fit_idx)
        P_oof[hold] = blend(w_fold, [p[hold] for p in oof])

    P_va, P_te = blend(w, va), blend(w, te)

    for tag, arr in [("oof", P_oof), ("valid", P_va), ("test", P_te)]:
        np.save(os.path.join(PRED, f"{ds}_ens_{tag}.npy"), arr.astype(np.float32))

    with open(os.path.join(MET, f"{ds}_ens_weights.json"), "w") as f:
        json.dump({"models": chosen, "weights": [float(x) for x in w],
                   "grid_step": GRID_STEP, "topk": TOPK,
                   "fitted_on": "out-of-fold predictions over train"}, f, indent=2)

    key = "auc" if cls else "rmse"
    score = cls_metrics if cls else (lambda a, b: reg_metrics(a, b, ds))
    m_va, m_te = score(y_va, P_va), score(y_te, P_te)
    weights = dict(zip(chosen, [round(float(x), 2) for x in w]))
    print(f"{ds} ENS {weights} | valid {key}={m_va[key]:.4f} | test {key}={m_te[key]:.4f}")


def main():
    meta = json.load(open(os.path.join(DATA, "dataset_meta.json")))
    for ds in meta:
        run_one(ds)


if __name__ == "__main__":
    main()

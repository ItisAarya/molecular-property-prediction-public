"""
src/eval/thresholds.py

Per-task decision thresholds for the classification models.

    python -m src.eval.thresholds

Writes results/metrics/<ds>_<model>_thresholds.json.

WHAT CHANGED, AND WHY
---------------------
A threshold is a fitted parameter. The inherited version searched for it on the validation
split and then reported thresholded validation metrics from those same rows, which flatters
the result -- with a few hundred rows and a rare positive class, the best threshold on that
sample is partly fitted to its noise.

Thresholds are now chosen on out-of-fold predictions over `train` (cross-fitted, for the
hybrid and ensemble), so validation stays clean for model selection.

The inherited version also used different objectives for different models -- balanced
accuracy for the base learners but F1 for the ensemble. That makes the models
non-comparable: F1 ignores true negatives while balanced accuracy weights both classes
equally, so the two objectives pick systematically different operating points. Every model
now uses the same objective.

WHY BALANCED ACCURACY
---------------------
These datasets are heavily imbalanced (Tox21 is ~4% positive on some assays). Plain
accuracy is maximised by predicting the majority class everywhere, and F1 ignores true
negatives entirely, which for a toxicity screen is the wrong thing to ignore -- correctly
clearing a safe compound has real value. Balanced accuracy is the mean of sensitivity and
specificity, so both classes count regardless of how rare one is.

Tasks with too few positives to fit on keep the default 0.5 and are recorded as such,
rather than adopting a threshold tuned on a handful of molecules.
"""

import json
import os

import numpy as np
import pandas as pd

from src.data.splits import enough_positives, load_y
from src.eval.metrics import is_classification

DATA = "data"
PRED = "results/preds"
OUT = "results/metrics"

MODELS = ["rf", "gnn", "trf", "hybrid", "ens"]
OBJECTIVE = "bal_acc"
GRID = np.linspace(0.05, 0.95, 37)  # 0.025 resolution

os.makedirs(OUT, exist_ok=True)


def load_pred(ds, model, split):
    """Prefer the calibrated file when present, matching what evaluation will use."""
    for name in (f"{ds}_{model}_{split}_cal.npy", f"{ds}_{model}_{split}.npy"):
        path = os.path.join(PRED, name)
        if os.path.exists(path):
            p = np.load(path)
            return p.reshape(-1, 1) if p.ndim == 1 else p
    return None


def balanced_accuracy(y, p, thr):
    """Mean of sensitivity and specificity at the given threshold."""
    pred = (p >= thr).astype(int)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    sens = (pred[pos] == 1).mean()
    spec = (pred[neg] == 0).mean()
    return 0.5 * (sens + spec)


def best_threshold(y_task, p_task):
    """Scan the grid and return the threshold maximising balanced accuracy."""
    mask = ~np.isnan(y_task)
    y, p = y_task[mask], p_task[mask]
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5
    scores = [balanced_accuracy(y, p, t) for t in GRID]
    return float(GRID[int(np.nanargmax(scores))])


def main():
    meta = json.load(open(os.path.join(DATA, "dataset_meta.json")))
    rows = []

    for ds in meta:
        if not is_classification(ds):
            continue
        y_tr = load_y(ds, "train")

        for model in MODELS:
            p_oof = load_pred(ds, model, "oof")
            if p_oof is None:
                continue

            thresholds, defaulted = [], []
            for t in range(p_oof.shape[1]):
                if not enough_positives(y_tr, t):
                    thresholds.append(0.5)
                    defaulted.append(t)
                else:
                    thresholds.append(best_threshold(y_tr[:, t], p_oof[:, t]))

            with open(os.path.join(OUT, f"{ds}_{model}_thresholds.json"), "w") as f:
                json.dump({
                    "thresholds": thresholds,
                    "metric": OBJECTIVE,
                    "fitted_on": "out-of-fold predictions over train",
                    "defaulted_tasks": defaulted,
                }, f, indent=2)

            rows.append({"dataset": ds, "model": model, "metric": OBJECTIVE,
                         "thresholds": thresholds, "n_defaulted": len(defaulted)})
            note = f"  ({len(defaulted)} task(s) kept 0.5)" if defaulted else ""
            shown = ", ".join(f"{t:.2f}" for t in thresholds[:6])
            more = " ..." if len(thresholds) > 6 else ""
            print(f"  {ds}-{model:<7} [{shown}{more}]{note}")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "thresholds_summary.csv"), index=False)
        print(f"\nAll models use the same objective ({OBJECTIVE}), fitted out-of-fold.")


if __name__ == "__main__":
    main()

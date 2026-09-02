"""
src/eval/report_with_thresholds.py

Build the final results table: every model, every dataset, calibrated probabilities and
out-of-fold thresholds applied.

    python -m src.eval.report_with_thresholds

Writes results/metrics/final_report_thresholded.csv with `oof_*`, `val_*` and `test_*`
columns.

WHY THREE SPLITS ARE REPORTED
-----------------------------
Each answers a different question, and keeping them side by side makes a leak visible
rather than invisible:

    oof_*   how the model does on training molecules it did not train on. This is the
            surface every post-hoc stage was fitted on, so it is optimistic by
            construction and must never be used to choose anything.
    val_*   the honest selection surface. Nothing is fitted on it any more, so this is
            what src/eval/select_winners.py reads.
    test_*  reported once, at the end.

A large val-to-test gap is now interpretable. Before this phase every gap looked the same
whether it came from a leak or from genuine distribution shift. Now they separate: Tox21's
gap collapsed from 0.080 to 0.011 once the meta-learner stopped being fitted on the
validation rows, while BBBP's stayed at 0.23 because its cause is a label shift (the
training split is 82% positive against 52% in test), which no protocol change can remove.
"""

import json
import os

import numpy as np
import pandas as pd

from src.data.splits import load_y
from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA = "data"
PRED = "results/preds"
MET = "results/metrics"
MODELS = ["rf", "gnn", "trf", "hybrid", "ens"]

os.makedirs(MET, exist_ok=True)


def load_pred(ds, model, split):
    """Prefer calibrated predictions when they exist."""
    for name in (f"{ds}_{model}_{split}_cal.npy", f"{ds}_{model}_{split}.npy"):
        path = os.path.join(PRED, name)
        if os.path.exists(path):
            p = np.load(path)
            return p.reshape(-1, 1) if p.ndim == 1 else p
    return None


def load_thresholds(ds, model, n_tasks):
    path = os.path.join(MET, f"{ds}_{model}_thresholds.json")
    if not os.path.exists(path):
        return [0.5] * n_tasks
    thr = json.load(open(path))["thresholds"]
    if len(thr) < n_tasks:
        thr = list(thr) + [0.5] * (n_tasks - len(thr))
    return thr[:n_tasks]


def score(ds, model, split, cls, thresholds):
    """Metrics for one model on one split, with per-task thresholds applied."""
    p = load_pred(ds, model, split)
    if p is None:
        return None
    y = load_y(ds, "train" if split == "oof" else split)
    if p.shape[0] != y.shape[0]:
        return None

    if not cls:
        return reg_metrics(y, p, ds)

    # Threshold per task, then macro-average. cls_metrics takes one threshold, so the
    # tasks are scored individually and averaged here.
    per_task = []
    for t in range(p.shape[1]):
        m = ~np.isnan(y[:, t])
        if m.sum() == 0 or len(np.unique(y[m, t])) < 2:
            continue
        per_task.append(cls_metrics(y[m, t], p[m, t], thr=thresholds[t]))
    if not per_task:
        return None

    keys = ["auc", "ap", "acc", "bal_acc", "f1"]
    out = {k: float(np.mean([r[k] for r in per_task])) for k in keys}
    out["n_tasks_scored"] = len(per_task)
    return out


def main():
    meta = json.load(open(os.path.join(DATA, "dataset_meta.json")))
    rows = []

    for ds in meta:
        cls = is_classification(ds)
        n_tasks = len(meta[ds]["tasks"])

        for model in MODELS:
            thresholds = load_thresholds(ds, model, n_tasks)
            scores = {s: score(ds, model, s, cls, thresholds) for s in ("oof", "valid", "test")}
            if scores["test"] is None:
                continue

            row = {"dataset": ds, "model": model}
            for split, prefix in (("oof", "oof"), ("valid", "val"), ("test", "test")):
                if scores[split]:
                    row.update({f"{prefix}_{k}": v for k, v in scores[split].items()})
            rows.append(row)

            key = "auc" if cls else "rmse"
            o = row.get(f"oof_{key}", float("nan"))
            v = row.get(f"val_{key}", float("nan"))
            t = row.get(f"test_{key}", float("nan"))
            print(f"  {ds:<15}{model:<8} {key}: oof={o:.4f}  val={v:.4f}  test={t:.4f}  "
                  f"(val-test gap {v - t:+.4f})")

    df = pd.DataFrame(rows)
    out = os.path.join(MET, "final_report_thresholded.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("Selection must use val_*; oof_* is the surface the post-hoc stages were fitted on.")


if __name__ == "__main__":
    main()

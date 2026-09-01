"""
src/eval/recompute_metrics.py

Recompute every model's metrics from the prediction files already on disk.

    python -m src.eval.recompute_metrics

WHY THIS EXISTS
---------------
Training writes its predictions to results/preds/*.npy. Everything after that -- metrics,
calibration, thresholds -- is a pure function of those predictions and the labels. So when
a metric definition changes there is no reason to retrain: retraining a frozen ChemBERTa
over five datasets costs well over an hour of CPU and, worse, would mix a metric change
together with a fresh training run, making it impossible to attribute a difference to
either one.

This script recomputes the per-model valid/test CSVs in place, so every number in
results/metrics/ is guaranteed to come from the current definition in src/eval/metrics.py.
"""

import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA_DIR = "data"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"

MODELS = ["rf", "gnn", "trf", "hybrid", "ens"]
SPLITS = ["valid", "test"]


def load_y(ds, split):
    y = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"))["y"].astype(np.float32)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def load_pred(ds, model, split, calibrated=False):
    """Load a prediction file, optionally preferring the calibrated variant."""
    if calibrated:
        cal = os.path.join(PRED_DIR, f"{ds}_{model}_{split}_cal.npy")
        if os.path.exists(cal):
            return np.load(cal)
    raw = os.path.join(PRED_DIR, f"{ds}_{model}_{split}.npy")
    return np.load(raw) if os.path.exists(raw) else None


def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    rows = []

    for ds in meta:
        cls = is_classification(ds)
        for model in MODELS:
            for split in SPLITS:
                p = load_pred(ds, model, split)
                if p is None:
                    continue
                y = load_y(ds, split)
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                if p.shape[0] != y.shape[0]:
                    print(
                        f"  [skip] {ds} {model} {split}: "
                        f"{p.shape[0]} predictions vs {y.shape[0]} labels"
                    )
                    continue

                m = cls_metrics(y, p) if cls else reg_metrics(y, p, ds)
                pd.DataFrame([m]).to_csv(
                    os.path.join(MET_DIR, f"{ds}_{model}_{split}.csv"), index=False
                )
                rows.append({"dataset": ds, "model": model, "split": split, **m})

    if not rows:
        print("No predictions found.")
        return

    df = pd.DataFrame(rows)
    out = os.path.join(MET_DIR, "base_model_report.csv")
    df.to_csv(out, index=False)
    print(f"Recomputed {len(rows)} model/split combinations -> {out}\n")

    for ds in df.dataset.unique():
        sub = df[(df.dataset == ds) & (df.split == "test")]
        if sub.empty:
            continue
        if is_classification(ds):
            cols = ["model", "auc", "ap", "f1", "n_tasks_scored"]
        else:
            cols = ["model", "rmse", "mae", "r2", "rmse_norm"]
        print(f"  {ds} (test)")
        print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print()


if __name__ == "__main__":
    main()

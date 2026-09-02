"""
src/eval/select_winners.py

Pick the best model per dataset and record everything needed to reproduce it.

    python -m src.eval.select_winners

Writes results/metrics/winners.json.

WHAT CHANGED, AND WHY
---------------------
The inherited version selected on `test_auc` / `test_rmse`. That is model selection on the
test set: with five candidates per dataset, picking the one that happens to score best on
783 test molecules and then reporting that same score is a best-of-five maximum, not an
estimate of generalisation. It is the single most serious methodological problem in the
inherited pipeline, because it silently inflates every headline number.

Selection now uses `val_*`. The validation split is no longer fitted on by anything -- the
meta-learner, ensemble weights, calibrators and thresholds all moved to out-of-fold
predictions over `train` -- so it is a clean surface for choosing between models, and the
test score is only read after the choice is made.

The recorded `selection_gap` (val minus test for the chosen model) is kept deliberately.
It is the honest measure of how much optimism remains, and it separates the two causes we
can now distinguish: a leak, which this phase removed, and genuine distribution shift,
which it cannot.
"""

import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import is_classification

MET = "results/metrics"
DATA = "data"
MODELS = ["rf", "gnn", "trf", "hybrid", "ens"]

# Selection metric and direction, per task type. Deliberately a *validation* column.
PRIMARY = {
    "classification": ("val_auc", "max"),
    "regression": ("val_rmse", "min"),
}


def load_report():
    path = os.path.join(MET, "final_report_thresholded.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run: python -m src.eval.report_with_thresholds first.")
    return pd.read_csv(path)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    df = load_report()
    winners = {}

    for ds in sorted(df["dataset"].unique()):
        task_type = "classification" if is_classification(ds) else "regression"
        metric, mode = PRIMARY[task_type]

        sub = df[(df["dataset"] == ds) & (df["model"].isin(MODELS))].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            print(f"  {ds}: no candidate with a usable {metric}")
            continue

        idx = sub[metric].idxmax() if mode == "max" else sub[metric].idxmin()
        row = sub.loc[idx]
        model = str(row["model"])

        test_metric = metric.replace("val_", "test_")
        val_v, test_v = float(row[metric]), float(row.get(test_metric, np.nan))

        entry = {
            "dataset": ds,
            "model": model,
            "selected_on": metric,
            "selection_value": val_v,
            "test_value": test_v,
            "selection_gap": val_v - test_v,
            "test_metrics": {c: float(row[c]) for c in row.index
                             if c.startswith("test_") and pd.notna(row[c])},
        }

        if task_type == "classification":
            thr = load_json(os.path.join(MET, f"{ds}_{model}_thresholds.json"))
            entry["thresholds"] = thr
            entry["calibration"] = load_json(
                os.path.join(MET, f"{ds}_{model}_calibration_methods.json")
            )
        if model == "ens":
            entry["ensemble"] = load_json(os.path.join(MET, f"{ds}_ens_weights.json"))

        winners[ds] = entry

        # Show the runner-up: if it is within noise of the winner, the choice is arbitrary
        # and the paper should say so rather than implying a clear victory.
        ordered = sub.sort_values(metric, ascending=(mode == "min"))
        runner = ordered.iloc[1] if len(ordered) > 1 else None
        margin = f", next best {runner['model']} at {runner[metric]:.4f}" if runner is not None else ""
        print(f"  {ds:<15} -> {model:<7} {metric}={val_v:.4f}  "
              f"{test_metric}={test_v:.4f}  gap={val_v - test_v:+.4f}{margin}")

    out = os.path.join(MET, "winners.json")
    with open(out, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nWrote {out}")
    print("Selected on validation; the test column was read only after the choice was made.")


if __name__ == "__main__":
    main()

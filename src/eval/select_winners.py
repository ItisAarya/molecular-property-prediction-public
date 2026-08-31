# src/eval/select_winners.py
import os, json
import numpy as np
import pandas as pd
from src.eval.metrics import is_classification

MET = "results/metrics"
DATA = "data"

PRIMARY = {
    "classification": ("test_auc", "max"),   # maximise AUC
    "regression": ("test_rmse", "min"),      # minimise RMSE
}

MODELS = ["rf", "gnn", "trf", "hybrid", "ens"]


def load_report():
    path = os.path.join(MET, "final_report_thresholded.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run: python -m src.eval.report_with_thresholds first.")
    return pd.read_csv(path)


def load_thresholds(dataset: str, model: str):
    path = os.path.join(MET, f"{dataset}_{model}_thresholds.json")
    return json.load(open(path)) if os.path.exists(path) else None


def load_ens_weights(dataset: str):
    path = os.path.join(MET, f"{dataset}_ens_weights.json")
    return json.load(open(path)) if os.path.exists(path) else None


def task_count(dataset: str) -> int:
    yv_path = os.path.join(DATA, f"{dataset}_valid_ecfp.npz")
    arr = np.load(yv_path)["y"]
    return arr.shape[1] if arr.ndim == 2 else 1


def fallback_thresholds(dataset: str):
    """If no thresholds file exists, fall back to 0.5 per task."""
    T = task_count(dataset)
    return {"metric": "fallback_0.5", "thresholds": [0.5] * T}


def best_row_by(df_ds: pd.DataFrame, metric: str, mode: str) -> pd.Series:
    # ensure numeric and drop NaNs
    df_ds = df_ds.copy()
    df_ds[metric] = pd.to_numeric(df_ds[metric], errors="coerce")
    df_ds = df_ds.dropna(subset=[metric])
    if df_ds.empty:
        raise ValueError(f"No rows with numeric metric '{metric}' to choose from.")
    idx = df_ds[metric].idxmax() if mode == "max" else df_ds[metric].idxmin()
    return df_ds.loc[idx]


def main():
    df = load_report()
    winners = {}

    for ds in sorted(df["dataset"].unique()):
        task_type = "classification" if is_classification(ds) else "regression"
        metric, mode = PRIMARY[task_type]

        df_ds = df[(df["dataset"] == ds) & (df["model"].isin(MODELS))]
        if df_ds.empty:
            continue
        # some models may lack the chosen metric
        df_ds = df_ds[~df_ds[metric].isna()]
        if df_ds.empty:
            continue

        row = best_row_by(df_ds, metric, mode)
        best_model = str(row["model"])

        # collect test_* metrics into a dict
        all_metrics = {}
        for c in row.index:
            if c.startswith("test_"):
                try:
                    all_metrics[c] = float(row[c])
                except Exception:
                    # keep as raw if not numeric
                    all_metrics[c] = row[c]

        entry = {
            "dataset": ds,
            "model": best_model,
            "primary_metric": metric,
            "primary_value": float(row[metric]),
            "all_metrics": all_metrics,
        }

        # thresholds for classification models
        if task_type == "classification":
            th = load_thresholds(ds, best_model)
            entry["thresholds"] = th if th is not None else fallback_thresholds(ds)

        # ensemble weights if applicable
        if best_model == "ens":
            w = load_ens_weights(ds)
            if w is not None:
                entry["ensemble"] = w

        winners[ds] = entry

    out_path = os.path.join(MET, "winners.json")
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print("Saved winners →", out_path)
    print(json.dumps(winners, indent=2))


if __name__ == "__main__":
    main()

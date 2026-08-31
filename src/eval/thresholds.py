# src/eval/thresholds.py
import os, json, numpy as np, pandas as pd
from src.eval.metrics import is_classification, cls_metrics

DATA="data"; PRED="results/preds"; OUT="results/metrics"
os.makedirs(OUT, exist_ok=True)

# Include ensemble
MODELS = ["rf","gnn","trf","hybrid","ens"]

# Metric to optimise thresholds per model
MODEL_THRESH_METRIC = {
    "rf": "bal_acc",
    "gnn": "bal_acc",
    "trf": "bal_acc",
    "hybrid": "bal_acc",
    "ens": "f1",   # Ensemble: push F1 for class-imbalanced sets
}

def load_y(ds, split):
    d = np.load(os.path.join(DATA, f"{ds}_{split}_ecfp.npz"))
    return d["y"].astype(np.float32)

def load_pred(ds, model, split):
    # Ensembles are already final; others may have *_cal.npy
    cal = os.path.join(PRED, f"{ds}_{model}_{split}_cal.npy")
    raw = os.path.join(PRED, f"{ds}_{model}_{split}.npy")
    path = cal if os.path.exists(cal) else raw
    return np.load(path) if os.path.exists(path) else None

def best_thresholds(y_val, p_val, metric="bal_acc"):
    if p_val.ndim == 1: p_val = p_val[:,None]
    T = p_val.shape[1]
    thr_grid = np.linspace(0.05, 0.95, 19)
    out = [0.5]*T
    for t in range(T):
        yt = y_val[:,t] if T>1 else y_val.reshape(-1)
        m = ~np.isnan(yt)
        if m.sum()==0 or len(np.unique(yt[m]))<2:
            out[t] = 0.5; continue
        best_s, best_th = -1.0, 0.5
        for th in thr_grid:
            res = cls_metrics(yt[m], p_val[m,t], thr=th)
            score = res["bal_acc"] if metric=="bal_acc" else res["f1"]
            if score > best_s:
                best_s, best_th = score, th
        out[t] = float(best_th)
    return out

def main():
    meta = json.load(open(os.path.join(DATA,"dataset_meta.json")))
    rows=[]
    for ds in meta.keys():
        if not is_classification(ds): 
            continue
        for model in MODELS:
            yv = load_y(ds,"valid")
            pv = load_pred(ds, model, "valid")
            if pv is None: 
                continue
            metric = MODEL_THRESH_METRIC.get(model, "bal_acc")
            th = best_thresholds(yv, pv, metric=metric)
            with open(os.path.join(OUT, f"{ds}_{model}_thresholds.json"), "w") as f:
                json.dump({"thresholds":th, "metric":metric}, f, indent=2)
            rows.append({"dataset":ds,"model":model,"metric":metric,"thresholds":th})
            print(f"{ds} {model} → thresholds (metric={metric}) {th}")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT,"thresholds_summary.csv"), index=False)

if __name__ == "__main__":
    main()

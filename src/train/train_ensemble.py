# src/train/train_ensemble.py
import os, json, numpy as np, pandas as pd
from itertools import product
from src.eval.metrics import is_classification, cls_metrics, reg_metrics

# Try modern RMSE; fall back if needed
try:
    from sklearn.metrics import root_mean_squared_error as sk_rmse
    def rmse(y_true, y_pred):
        return float(sk_rmse(y_true.reshape(-1), y_pred.reshape(-1)))
except Exception:
    from sklearn.metrics import mean_squared_error
    def rmse(y_true, y_pred):
        return float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1), squared=False))

DATA = "data"
PRED = "results/preds"
MET  = "results/metrics"
os.makedirs(MET, exist_ok=True)

# Candidate base models to consider
CANDS = ["rf","gnn","trf","hybrid"]

TOPK = 2          # pick the top-K models on validation
GRID_STEP = 0.10  # finer than 0.25

def load_y(ds, split):
    d = np.load(os.path.join(DATA, f"{ds}_{split}_ecfp.npz"))
    return d["y"].astype(np.float32)

def load_pred(ds, model, split):
    cal = os.path.join(PRED, f"{ds}_{model}_{split}_cal.npy")
    raw = os.path.join(PRED, f"{ds}_{model}_{split}.npy")
    path = cal if os.path.exists(cal) else raw
    return np.load(path) if os.path.exists(path) else None

def available_models(ds):
    avail = []
    for m in CANDS:
        if load_pred(ds, m, "valid") is not None and load_pred(ds, m, "test") is not None:
            avail.append(m)
    return avail

def model_score(ds, model, yv, cls):
    pv = load_pred(ds, model, "valid")
    if pv is None: return -np.inf
    if pv.ndim == 1: pv = pv[:,None]
    if cls:
        return cls_metrics(yv, pv)["auc"]  # higher is better
    else:
        return -rmse(yv, pv)               # higher is better (neg RMSE)

def normalised_weight_grid(n, step=0.10):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    if n == 1:
        yield np.array([1.0], dtype=float); return
    for w in product(vals, repeat=n):
        s = sum(w)
        if s == 0: continue
        if abs(s - 1.0) < 1e-9:
            yield np.array(w, dtype=float)

def stack_preds(w, arrs):
    out = np.zeros_like(arrs[0], dtype=float)
    for wi, Ai in zip(w, arrs):
        out += wi * Ai
    return out

def run_one(ds):
    models_all = available_models(ds)
    if len(models_all) == 0:
        print(ds, "ensemble skipped (no base models)."); return

    yv = load_y(ds, "valid")
    yt = load_y(ds, "test")
    cls = is_classification(ds)

    # Rank by validation performance and keep top-K
    scores = [(m, model_score(ds, m, yv, cls)) for m in models_all]
    scores.sort(key=lambda x: x[1], reverse=True)
    models = [m for m,_ in scores[:max(1, min(TOPK, len(scores)))]]

    pv_list = [load_pred(ds, m, "valid") for m in models]
    pt_list = [load_pred(ds, m, "test")  for m in models]
    if pv_list[0].ndim == 1:
        pv_list = [p[:,None] for p in pv_list]
        pt_list = [p[:,None] for p in pt_list]

    # Objective
    if cls:
        best_score = -np.inf; better = lambda a,b: a>b
    else:
        best_score =  np.inf; better = lambda a,b: a<b

    best_w, best_metric = None, None
    for w in normalised_weight_grid(len(models), step=GRID_STEP):
        Pv = stack_preds(w, pv_list)
        if cls:
            m = cls_metrics(yv, Pv)      # maximise AUC
            score = -m["auc"] * -1       # explicit float
            metric = m
        else:
            score = rmse(yv, Pv)         # minimise RMSE
            metric = {"rmse": score}
        if better(score, best_score):
            best_score, best_w, best_metric = score, w, metric

    if best_w is None:
        # Fallback to the single best model
        best_w = np.array([1.0], dtype=float)
        models = [scores[0][0]]
        pv_list = [load_pred(ds, models[0], "valid")]
        pt_list = [load_pred(ds, models[0], "test")]
        if pv_list[0].ndim == 1:
            pv_list = [pv_list[0][:,None]]
            pt_list = [pt_list[0][:,None]]

    Pv = stack_preds(best_w, pv_list)
    Pt = stack_preds(best_w, pt_list)

    # Save ensemble preds & weights
    np.save(os.path.join(PRED, f"{ds}_ens_valid.npy"), Pv)
    np.save(os.path.join(PRED, f"{ds}_ens_test.npy"),  Pt)
    with open(os.path.join(MET, f"{ds}_ens_weights.json"), "w") as f:
        json.dump({"models":models, "weights":best_w.tolist(), "grid_step":GRID_STEP, "topk":TOPK}, f, indent=2)

    # Log metrics
    if cls:
        mv = cls_metrics(yv, Pv); mt = cls_metrics(yt, Pt)
    else:
        mv = reg_metrics(yv, Pv);  mt = reg_metrics(yt, Pt)

    wdict = dict(zip(models, [round(float(x),2) for x in best_w]))
    print(ds, "ENS using", wdict, "| VALID:", mv, "| TEST:", mt)

def main():
    meta = json.load(open(os.path.join(DATA, "dataset_meta.json")))
    for ds in meta.keys():
        run_one(ds)

if __name__ == "__main__":
    main()

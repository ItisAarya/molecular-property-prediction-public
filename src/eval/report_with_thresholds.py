# src/eval/report_with_thresholds.py
import os, json, numpy as np, pandas as pd
from glob import glob
from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA="data"; PRED="results/preds"; MET="results/metrics"
os.makedirs(MET, exist_ok=True)

MODELS = ["rf","gnn","trf","hybrid","ens"]

def load_y(ds, split):
    d = np.load(os.path.join(DATA, f"{ds}_{split}_ecfp.npz"))
    return d["y"].astype(np.float32)

def load_pred(ds, model, split):
    # use calibrated if present (for rf/gnn/trf), raw for hybrid/ens
    cal = os.path.join(PRED, f"{ds}_{model}_{split}_cal.npy")
    raw = os.path.join(PRED, f"{ds}_{model}_{split}.npy")
    path = cal if os.path.exists(cal) else raw
    return np.load(path) if os.path.exists(path) else None

def load_thresholds(ds, model):
    path = os.path.join(MET, f"{ds}_{model}_thresholds.json")
    if not os.path.exists(path): return None
    return json.load(open(path))["thresholds"]

def apply_thresholds(p, th):
    if p.ndim == 1: p = p[:,None]
    T = p.shape[1]
    th = th if th is not None else [0.5]*T
    th = np.array(th, dtype=float)
    th = th[:T] if len(th)>=T else np.pad(th, (0, T-len(th)), constant_values=0.5)
    return (p >= th[None,:]).astype(int)

def macro_avg(res_list, keys):
    # average dicts (classification multi-task macro average)
    out = {}
    for k in keys:
        vals = [r[k] for r in res_list if k in r]
        if len(vals):
            out[k] = float(np.mean(vals))
    return out

def eval_one(ds, model):
    yv = load_y(ds, "valid"); yt = load_y(ds, "test")
    pv = load_pred(ds, model, "valid"); pt = load_pred(ds, model, "test")
    if pv is None or pt is None:
        return None, None
    cls = is_classification(ds)

    if cls:
        # thresholded metrics (per-task)
        th = load_thresholds(ds, model)
        yv_pred = apply_thresholds(pv, th)
        yt_pred = apply_thresholds(pt, th)

        # Compute per-task results then macro-average for multi-task datasets
        if pv.ndim == 1: pv = pv[:,None]; pt = pt[:,None]
        T = pv.shape[1]

        val_task_metrics = []
        test_task_metrics = []
        for t in range(T):
            m_v = ~np.isnan(yv[:,t])
            m_t = ~np.isnan(yt[:,t])
            val_task_metrics.append(cls_metrics(yv[m_v, t], pv[m_v, t], thr=th[t]))
            test_task_metrics.append(cls_metrics(yt[m_t, t], pt[m_t, t], thr=th[t]))

        KEYS = ["auc","ap","acc","bal_acc","f1"]
        mv = macro_avg(val_task_metrics, KEYS)
        mt = macro_avg(test_task_metrics, KEYS)

    else:
        mv = reg_metrics(yv, pv)
        mt = reg_metrics(yt, pt)

    return mv, mt

def main():
    meta = json.load(open(os.path.join(DATA,"dataset_meta.json")))
    rows=[]
    for ds in meta.keys():
        for model in MODELS:
            mv, mt = eval_one(ds, model)
            if mv is None: continue
            print(f"{ds} {model} VALID: {mv} | TEST: {mt}")
            rows.append({"dataset":ds, "model":model, **{f"val_{k}":v for k,v in mv.items()},
                         **{f"test_{k}":v for k,v in mt.items()}})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(MET, "final_report_thresholded.csv"), index=False)

if __name__ == "__main__":
    main()

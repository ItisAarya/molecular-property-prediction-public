# src/eval/calibration.py
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import expit  # stable sigmoid
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data"
PRED_DIR = "results/preds"
MET_DIR  = "results/metrics"
FIG_DIR  = "results/figs"
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_DS = ["tox21","bbbp","clintox"]
MODELS   = ["rf","gnn","trf","hybrid"]

# ✅ Skip calibrating the stacked Hybrid model (identity only)
CALIBRATE_MODELS = {"rf": True, "gnn": True, "trf": True, "hybrid": False}

def load_y(ds, split):
    d = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"))
    return d["y"].astype(np.float32)

def load_pred_if_exists(ds, model, split):
    path = os.path.join(PRED_DIR, f"{ds}_{model}_{split}.npy")
    return np.load(path) if os.path.exists(path) else None

def sigmoid(x): return expit(x)
def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p) - np.log1p(-p)
def nll(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -np.mean(y*np.log(p) + (1-y)*np.log1p(-p))

# ---------- Symmetric, adaptive-bin ECE ----------
def ece_symmetric(y_true, p_prob, bins=15, adaptive=True):
    mask = ~np.isnan(y_true)
    y = y_true[mask].astype(int)
    p = p_prob[mask]
    if y.size == 0: return np.nan
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat==1, p, 1-p)        # predicted-class confidence
    correct = (yhat == y).astype(int)

    if adaptive:
        qs = np.linspace(0, 1, bins+1)
        edges = np.unique(np.quantile(conf, qs))
        if edges.size < 2:
            return np.nan
    else:
        edges = np.linspace(0,1,bins+1)

    ece = 0.0
    N = conf.size
    for i in range(len(edges)-1):
        m = (conf >= edges[i]) & (conf < edges[i+1])
        if m.sum()==0: continue
        c_bar = conf[m].mean()
        a_bar = correct[m].mean()
        ece += (m.sum()/N) * abs(a_bar - c_bar)
    return float(ece)

def reliability_plot(ds, model, split, task_name, y_true, p_prob, suffix, bins=15):
    mask = ~np.isnan(y_true)
    y = y_true[mask].astype(int)
    p = p_prob[mask]
    if y.size == 0:
        return None, np.nan
    yhat = (p >= 0.5).astype(int)
    conf = np.where(yhat==1, p, 1-p)
    correct = (yhat == y).astype(int)

    qs = np.linspace(0,1,bins+1)
    edges = np.unique(np.quantile(conf, qs))
    xs, ys = [], []
    for i in range(len(edges)-1):
        m = (conf >= edges[i]) & (conf < edges[i+1])
        if m.sum()==0:
            xs.append(np.nan); ys.append(np.nan); continue
        xs.append(conf[m].mean()); ys.append(correct[m].mean())

    e = ece_symmetric(y, p, bins=bins, adaptive=True)
    plt.figure(figsize=(4.0,3.2))
    plt.plot([0,1],[0,1], linestyle="--")
    xs2 = [v for v in xs if not np.isnan(v)]
    ys2 = [v for i,v in enumerate(ys) if not np.isnan(xs[i])]
    plt.scatter(xs2, ys2, s=18)
    plt.xlabel("Confidence (predicted class)")
    plt.ylabel("Accuracy")
    plt.title(f"{ds}-{model}-{split} ({task_name}) ECE={e:.03f}")
    out = os.path.join(FIG_DIR, f"{ds}_{model}_{split}_{task_name}_reliability_{suffix}.png")
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    return out, e

# ---------- Calibrators ----------
def fit_temperature(p_val, y_val):
    L = logit(p_val); best_T, best_loss = 1.0, 1e9
    for T in np.logspace(-1.5,1.0,30):
        loss = nll(y_val, sigmoid(L/T))
        if loss < best_loss: best_T, best_loss = T, loss
    for _ in range(20):
        cands = best_T * np.array([0.9,0.95,1.0,1.05,1.1])
        vals = [nll(y_val, sigmoid(L/t)) for t in cands]
        i = int(np.argmin(vals))
        if cands[i] == best_T: break
        best_T = cands[i]
    return best_T

def fit_logistic(p_val, y_val):
    L = logit(p_val).reshape(-1,1)
    y = y_val.astype(int).reshape(-1)
    if len(np.unique(y)) < 2:
        # logistic regression cannot fit single-class targets
        return None
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    lr.fit(L, y)
    a = float(lr.coef_.reshape(-1)[0]); b = float(lr.intercept_[0])
    return a, b

def apply_calibration(p, method):
    k = method["kind"]
    if k == "identity":   return p
    if k == "temperature":return sigmoid(logit(p)/method["T"])
    if k == "logistic":   return sigmoid(method["a"]*logit(p) + method["b"])
    raise ValueError("Unknown calibration kind")

def choose_by_ece(pv, yv):
    """Pick {identity, temperature, logistic} by lowest VALID ECE (adaptive).
       Tie-breaker: lowest VALID NLL."""
    mask = ~np.isnan(yv)
    pv_m = pv[mask]; yv_m = yv[mask].astype(int)
    if pv_m.size == 0:
        return {"kind":"identity"}, pv

    # identity
    e_id  = ece_symmetric(yv, pv, adaptive=True)
    n_id  = nll(yv_m, pv_m)

    # temperature
    T = fit_temperature(pv_m, yv_m)
    pv_temp = sigmoid(logit(pv)/T)
    e_temp = ece_symmetric(yv, pv_temp, adaptive=True)
    n_temp = nll(yv_m, sigmoid(logit(pv_m)/T))

    # logistic (only if both classes present)
    res_log = fit_logistic(pv_m, yv_m)
    methods, eces, nlls = [], [], []
    # build candidates
    methods.append({"kind":"identity"});     eces.append(e_id);  nlls.append(n_id)
    methods.append({"kind":"temperature","T":float(T)}); eces.append(e_temp); nlls.append(n_temp)
    if res_log is not None:
        a,b = res_log
        pv_log = sigmoid(a*logit(pv) + b)
        e_log = ece_symmetric(yv, pv_log, adaptive=True)
        n_log = nll(yv_m, sigmoid(a*logit(pv_m) + b))
        methods.append({"kind":"logistic","a":float(a),"b":float(b)})
        eces.append(e_log); nlls.append(n_log)

    eces = np.array(eces); nlls = np.array(nlls)
    idxs = np.where(eces == np.nanmin(eces))[0]
    i = idxs[np.argmin(nlls[idxs])] if idxs.size>1 else int(idxs[0])
    chosen = methods[i]
    if chosen["kind"] == "temperature": pv_cal = pv_temp
    elif chosen["kind"] == "logistic":  pv_cal = sigmoid(chosen["a"]*logit(pv) + chosen["b"])
    else:                               pv_cal = pv
    return chosen, pv_cal

# ---------- Main ----------
def run():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    rows = []
    for ds in meta.keys():
        if ds not in CLASS_DS: continue

        yv = load_y(ds, "valid")
        yt = load_y(ds, "test")
        Ttasks = yv.shape[1] if yv.ndim==2 else 1

        for m in MODELS:
            pv = load_pred_if_exists(ds, m, "valid")
            pt = load_pred_if_exists(ds, m, "test")
            if pv is None or pt is None: 
                continue
            if pv.ndim==1: pv = pv[:,None]
            if pt.ndim==1: pt = pt[:,None]

            # If model is set to skip calibration, save identity and report
            if not CALIBRATE_MODELS.get(m, True):
                np.save(os.path.join(PRED_DIR, f"{ds}_{m}_valid_cal.npy"), pv)
                np.save(os.path.join(PRED_DIR, f"{ds}_{m}_test_cal.npy"),  pt)
                # summaries
                e_val_raw = float(np.nanmean([ece_symmetric(yv[:,t] if Ttasks>1 else yv.reshape(-1), pv[:,t]) for t in range(pv.shape[1])]))
                e_tst_raw = float(np.nanmean([ece_symmetric(yt[:,t] if Ttasks>1 else yt.reshape(-1), pt[:,t]) for t in range(pt.shape[1])]))
                # representative plots (raw only)
                counts = [(~np.isnan(yv[:,t] if Ttasks>1 else yv.reshape(-1))).sum() for t in range(pv.shape[1])]
                t_plot = int(np.argmax(counts))
                yv_tp = (yv[:,t_plot] if Ttasks>1 else yv.reshape(-1))
                yt_tp = (yt[:,t_plot] if Ttasks>1 else yt.reshape(-1))
                mv = ~np.isnan(yv_tp); mt = ~np.isnan(yt_tp)
                reliability_plot(ds, m, "valid", f"t{t_plot}", yv_tp[mv].astype(int), pv[mv, t_plot], "raw")
                reliability_plot(ds, m, "test",  f"t{t_plot}", yt_tp[mt].astype(int), pt[mt, t_plot], "raw")
                rows.append({"dataset":ds,"model":m,"ece_val_raw":e_val_raw,"ece_val_cal":e_val_raw,
                            "ece_test_raw":e_tst_raw,"ece_test_cal":e_tst_raw})
                print(f"{ds}-{m}: calibration skipped (identity). ECE val {e_val_raw:.3f} | test {e_tst_raw:.3f}")
                continue

            # per-task selection among {identity, temperature, logistic}
            pv_cal = np.zeros_like(pv)
            pt_cal = np.zeros_like(pt)
            choices = []
            for t in range(pv.shape[1]):
                yv_t = yv[:,t] if Ttasks>1 else yv.reshape(-1)
                choice, pv_t_cal = choose_by_ece(pv[:,t], yv_t)
                pv_cal[:,t] = pv_t_cal
                pt_cal[:,t] = apply_calibration(pt[:,t], choice)
                choices.append(choice)

            # ECE summaries (symmetric, adaptive)
            e_val_raw = float(np.nanmean([ece_symmetric(yv[:,t] if Ttasks>1 else yv.reshape(-1), pv[:,t]) for t in range(pv.shape[1])]))
            e_val_cal = float(np.nanmean([ece_symmetric(yv[:,t] if Ttasks>1 else yv.reshape(-1), pv_cal[:,t]) for t in range(pv.shape[1])]))
            e_tst_raw = float(np.nanmean([ece_symmetric(yt[:,t] if Ttasks>1 else yt.reshape(-1), pt[:,t]) for t in range(pt.shape[1])]))
            e_tst_cal = float(np.nanmean([ece_symmetric(yt[:,t] if Ttasks>1 else yt.reshape(-1), pt_cal[:,t]) for t in range(pt.shape[1])]))

            # representative plots (task with most labels)
            counts = [(~np.isnan(yv[:,t] if Ttasks>1 else yv.reshape(-1))).sum() for t in range(pv.shape[1])]
            t_plot = int(np.argmax(counts))
            yv_tp = (yv[:,t_plot] if Ttasks>1 else yv.reshape(-1))
            yt_tp = (yt[:,t_plot] if Ttasks>1 else yt.reshape(-1))
            mv = ~np.isnan(yv_tp); mt = ~np.isnan(yt_tp)
            reliability_plot(ds, m, "valid", f"t{t_plot}", yv_tp[mv].astype(int), pv[mv, t_plot], "raw")
            reliability_plot(ds, m, "valid", f"t{t_plot}", yv_tp[mv].astype(int), pv_cal[mv, t_plot], "cal")
            reliability_plot(ds, m, "test",  f"t{t_plot}", yt_tp[mt].astype(int), pt[mt, t_plot], "raw")
            reliability_plot(ds, m, "test",  f"t{t_plot}", yt_tp[mt].astype(int), pt_cal[mt, t_plot], "cal")

            # save outputs
            np.save(os.path.join(PRED_DIR, f"{ds}_{m}_valid_cal.npy"), pv_cal)
            np.save(os.path.join(PRED_DIR, f"{ds}_{m}_test_cal.npy"),  pt_cal)
            with open(os.path.join(MET_DIR, f"{ds}_{m}_calibration_methods.json"), "w") as f:
                json.dump({"choices":choices}, f, indent=2)

            rows.append({
                "dataset": ds, "model": m,
                "ece_val_raw": e_val_raw, "ece_val_cal": e_val_cal,
                "ece_test_raw": e_tst_raw, "ece_test_cal": e_tst_cal
            })
            print(f"{ds}-{m}: ECE (val) {e_val_raw:.3f}->{e_val_cal:.3f} | ECE (test) {e_tst_raw:.3f}->{e_tst_cal:.3f}")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(MET_DIR, "calibration_summary.csv"), index=False)

if __name__ == "__main__":
    run()

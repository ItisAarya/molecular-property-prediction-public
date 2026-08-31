# src/eval/metrics.py
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
    average_precision_score, r2_score, mean_absolute_error
)

# Prefer new sklearn API for RMSE; fallback if not available
try:
    from sklearn.metrics import root_mean_squared_error as _rmse
    def _rmse_value(y_true, y_pred): return float(_rmse(y_true, y_pred))
except Exception:
    from sklearn.metrics import mean_squared_error
    def _rmse_value(y_true, y_pred): return float(mean_squared_error(y_true, y_pred, squared=False))

def is_classification(dsname: str):
    return dsname.lower() in ["tox21","bbbp","clintox"]

def cls_metrics(y_true, y_prob, thr=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if y_true.ndim == 1: y_true = y_true.reshape(-1,1)
    if y_prob.ndim == 1: y_prob = y_prob.reshape(-1,1)
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    aucs, aps, accs, bals, f1s = [], [], [], [], []
    for t in range(y_true.shape[1]):
        yt = y_true[:,t]
        mask = ~np.isnan(yt)
        yt = yt[mask]; yp = y_prob[mask, t]; yb = y_pred[mask, t]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        aps.append(average_precision_score(yt, yp))
        accs.append(accuracy_score(yt, yb))
        bals.append(balanced_accuracy_score(yt, yb))
        f1s.append(f1_score(yt, yb))
    out["auc"] = float(np.nanmean(aucs)) if aucs else float("nan")
    out["ap"]  = float(np.nanmean(aps))  if aps  else float("nan")
    out["acc"] = float(np.nanmean(accs)) if accs else float("nan")
    out["bal_acc"] = float(np.nanmean(bals)) if bals else float("nan")
    out["f1"]  = float(np.nanmean(f1s))  if f1s  else float("nan")
    return out

def reg_metrics(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    return {
        "rmse": _rmse_value(y_true, y_pred),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred))
    }

"""
src/eval/calibration.py

Probability calibration for the classification models.

    python -m src.eval.calibration

For each (dataset, model, task) it picks the best of {identity, temperature scaling,
Platt/logistic scaling} and writes calibrated predictions alongside the raw ones.

WHAT CHANGED, AND WHY
---------------------
The inherited version fitted the calibrators on the validation split, chose between them
by validation ECE, and then reported the improvement on that same split -- so the reported
"ECE after calibration" was partly just the calibrator fitting the noise in those few
hundred rows. Unsurprisingly the gains did not survive to test: BBBP went 0.079 -> 0.053
on validation but 0.260 -> 0.202 on test.

Calibrators are now fitted on **out-of-fold predictions over `train`** and evaluated on
validation and test, which are never fitted on. For the hybrid and ensemble this means the
cross-fitted `*_oof.npy` files, so the calibrator sees meta-learner outputs that are
themselves out-of-sample; calibrating against in-sample meta predictions would be the same
leak one level up.

The inherited version also skipped calibrating the hybrid entirely. There is no longer a
reason to: with out-of-fold inputs it can be calibrated like anything else.

A NOTE ON WHAT CALIBRATION CANNOT FIX
-------------------------------------
BBBP's training split is 82% positive while its test split is 52%. A calibration map fitted
on the training distribution cannot correct a shift in the label marginal itself, so BBBP's
test ECE stays poor no matter how the map is fitted. That is a property of the benchmark
split, not a defect in the calibrator, and it is the motivating case for the conformal
prediction work planned in Phase 3.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # no display on a headless run
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.data.splits import enough_positives, load_y
from src.eval.metrics import is_classification

DATA_DIR = "data"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
FIG_DIR = "results/figs"

MODELS = ["rf", "gnn", "trf", "hybrid", "ens"]

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)


def load_pred(ds, model, split):
    path = os.path.join(PRED_DIR, f"{ds}_{model}_{split}.npy")
    if not os.path.exists(path):
        return None
    p = np.load(path)
    return p.reshape(-1, 1) if p.ndim == 1 else p


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def nll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))


def ece(y_true, p_prob, bins=15):
    """
    Symmetric, adaptive-bin expected calibration error.

    Symmetric: confidence is measured in the *predicted* class, so a confident negative
    counts the same as a confident positive. Adaptive bins (equal-count quantiles rather
    than equal-width) stop a handful of points in a sparse bin from dominating.
    """
    mask = ~np.isnan(y_true)
    y, p = y_true[mask].astype(int), p_prob[mask]
    if y.size == 0:
        return np.nan

    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(int)

    edges = np.unique(np.quantile(conf, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return np.nan

    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / conf.size) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


# --------------------------------------------------------------------------------------
# Calibration maps
# --------------------------------------------------------------------------------------
def fit_temperature(p, y):
    """Single-parameter scaling of the logits. Coarse sweep, then local refinement."""
    L = logit(p)
    best_T, best = 1.0, np.inf
    for T in np.logspace(-1.5, 1.0, 40):
        loss = nll(y, expit(L / T))
        if loss < best:
            best_T, best = T, loss
    for _ in range(20):
        cands = best_T * np.array([0.9, 0.95, 1.0, 1.05, 1.1])
        losses = [nll(y, expit(L / t)) for t in cands]
        pick = cands[int(np.argmin(losses))]
        if pick == best_T:
            break
        best_T = pick
    return float(best_T)


def fit_logistic(p, y):
    """Platt scaling: an affine map on the logits, so it can shift as well as sharpen."""
    if len(np.unique(y)) < 2:
        return None
    lr = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
    lr.fit(logit(p).reshape(-1, 1), y.astype(int))
    return float(lr.coef_.ravel()[0]), float(lr.intercept_[0])


def apply_map(p, method):
    kind = method["kind"]
    if kind == "identity":
        return p
    if kind == "temperature":
        return expit(logit(p) / method["T"])
    if kind == "logistic":
        return expit(method["a"] * logit(p) + method["b"])
    raise ValueError(f"unknown calibration kind: {kind}")


def choose_map(p_fit, y_fit):
    """Pick the map with the lowest ECE on the fitting data; break ties on NLL."""
    mask = ~np.isnan(y_fit)
    p, y = p_fit[mask], y_fit[mask].astype(int)
    if p.size == 0 or len(np.unique(y)) < 2:
        return {"kind": "identity"}

    candidates = [{"kind": "identity"}]
    candidates.append({"kind": "temperature", "T": fit_temperature(p, y)})
    platt = fit_logistic(p, y)
    if platt is not None:
        candidates.append({"kind": "logistic", "a": platt[0], "b": platt[1]})

    scores = [(ece(y.astype(float), apply_map(p, m)), nll(y, apply_map(p, m)), i)
              for i, m in enumerate(candidates)]
    scores.sort(key=lambda s: (s[0], s[1]))
    return candidates[scores[0][2]]


def reliability_plot(ds, model, split, y_true, p, suffix, bins=15):
    mask = ~np.isnan(y_true)
    y, pp = y_true[mask].astype(int), p[mask]
    if y.size == 0:
        return
    pred = (pp >= 0.5).astype(int)
    conf = np.where(pred == 1, pp, 1 - pp)
    correct = (pred == y).astype(int)

    edges = np.unique(np.quantile(conf, np.linspace(0, 1, bins + 1)))
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())

    plt.figure(figsize=(4.0, 3.2))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.scatter(xs, ys, s=18)
    plt.xlabel("Confidence (predicted class)")
    plt.ylabel("Accuracy")
    plt.title(f"{ds}-{model}-{split} ECE={ece(y_true, p):.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{ds}_{model}_{split}_reliability_{suffix}.png"), dpi=160)
    plt.close()


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    rows = []

    for ds in meta:
        if not is_classification(ds):
            continue

        y_tr, y_va, y_te = load_y(ds, "train"), load_y(ds, "valid"), load_y(ds, "test")

        for model in MODELS:
            p_oof = load_pred(ds, model, "oof")
            p_va = load_pred(ds, model, "valid")
            p_te = load_pred(ds, model, "test")
            if p_oof is None or p_va is None or p_te is None:
                continue

            cal_va = np.empty_like(p_va)
            cal_te = np.empty_like(p_te)
            choices = []

            for t in range(p_oof.shape[1]):
                # Too few positives to fit a map on: leave the probabilities alone rather
                # than fitting a correction to noise.
                if not enough_positives(y_tr, t):
                    choice = {"kind": "identity", "reason": "too few positives to fit"}
                else:
                    choice = choose_map(p_oof[:, t], y_tr[:, t])
                cal_va[:, t] = apply_map(p_va[:, t], choice)
                cal_te[:, t] = apply_map(p_te[:, t], choice)
                choices.append(choice)

            np.save(os.path.join(PRED_DIR, f"{ds}_{model}_valid_cal.npy"), cal_va)
            np.save(os.path.join(PRED_DIR, f"{ds}_{model}_test_cal.npy"), cal_te)
            with open(os.path.join(MET_DIR, f"{ds}_{model}_calibration_methods.json"), "w") as f:
                json.dump({"fitted_on": "out-of-fold predictions over train",
                           "choices": choices}, f, indent=2)

            def mean_ece(y, p):
                return float(np.nanmean([ece(y[:, t], p[:, t]) for t in range(p.shape[1])]))

            row = {
                "dataset": ds, "model": model,
                "ece_valid_raw": mean_ece(y_va, p_va), "ece_valid_cal": mean_ece(y_va, cal_va),
                "ece_test_raw": mean_ece(y_te, p_te), "ece_test_cal": mean_ece(y_te, cal_te),
            }
            rows.append(row)

            # Plot the task with the most labels, as a representative case.
            t_plot = int(np.argmax([(~np.isnan(y_va[:, t])).sum() for t in range(p_va.shape[1])]))
            reliability_plot(ds, model, "test", y_te[:, t_plot], p_te[:, t_plot], "raw")
            reliability_plot(ds, model, "test", y_te[:, t_plot], cal_te[:, t_plot], "cal")

            kinds = ", ".join(sorted({c["kind"] for c in choices}))
            print(f"  {ds}-{model:<7} ECE valid {row['ece_valid_raw']:.3f}->{row['ece_valid_cal']:.3f}"
                  f"  test {row['ece_test_raw']:.3f}->{row['ece_test_cal']:.3f}   [{kinds}]")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(MET_DIR, "calibration_summary.csv"), index=False)
        print("\nCalibrators fitted on out-of-fold train predictions; "
              "valid and test were never fitted on.")


if __name__ == "__main__":
    run()

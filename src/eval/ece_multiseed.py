"""
src/eval/ece_multiseed.py

Calibration error for any archived model, across the seeded scaffold splits.

    python -m src.eval.ece_multiseed --tags desc fuse_gated fuse_gated_nograph
    python -m src.eval.ece_multiseed --tags rf gnn ens --datasets tox21 bbbp clintox

WHY THIS EXISTS ALONGSIDE calibration.py
----------------------------------------
`src/eval/calibration.py` is the Phase 0 calibrator. It fits on **out-of-fold predictions
over train** -- the strongest protocol available here, because the calibrator then sees
probabilities the model never fitted on and the validation split stays reserved for early
stopping alone. It is also the reason it cannot be used for anything Phases 1-2 built: the
`*_oof.npy` files come from the inherited pipeline's cross-fitting step, and producing them
for a view or fusion model would mean k-fold retraining of every model on every split.

So Phase 3's calibration half reached five pipeline models on three datasets and none of
the five views or five fusion variants -- which is most of the models the paper is about.

This module closes that with the protocol the conformal work already uses: **fit the map on
`valid`, evaluate on `test`.** Test is never fitted on, so the reported test ECE is an
honest out-of-sample number.

WHAT IS WEAKER ABOUT IT, STATED PLAINLY
---------------------------------------
`valid` is also the early-stopping split. The epoch was chosen to look good on exactly the
rows the calibrator is then fitted to, so those probabilities are mildly optimistic and the
map is fitted to slightly the wrong distribution. This is second-order -- it biases the
*map*, not the evaluation, and test remains untouched -- but it is a real difference from
the out-of-fold protocol, and rows are labelled `valid-fit` against `oof-fit` so the two are
never averaged together as though they were the same measurement.

Both are reported. Where a model has OOF predictions the stronger protocol is available and
`calibration.py` is the number to quote; this module exists so that the models without them
are not simply absent from the calibration table.

WHY ACROSS SPLITS
-----------------
Phase 0 found that post-hoc calibration does not transfer reliably under scaffold shift --
BBBP improved on validation and got worse on test. A single split cannot distinguish "the
map helps" from "the map helped on that partition", which is the whole reason this project
reports five. ECE is therefore reported as mean +/- 95% CI over the seeded splits, and the
per-split spread is printed, because a calibration gain smaller than its own spread is not
a gain.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data.splits import enough_positives
from src.eval.calibration import apply_map, choose_map, ece, reliability_plot
from src.eval.metrics import is_classification

POOL_DIR = os.path.join("data", "pool")
SPLIT_DIR = os.path.join("data", "splits")
RUNS_DIR = os.path.join("results", "runs")
MET_DIR = os.path.join("results", "metrics")
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def load_labels(ds, variant, split):
    """Labels for one split of one variant, straight from the pool and the split index."""
    y = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)["y"]
    idx = json.load(open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")))
    return y[np.asarray(idx[split], dtype=int)]


def load_preds(ds, variant, tag, split):
    path = os.path.join(RUNS_DIR, variant, "preds", f"{ds}_{tag}_{split}.npy")
    if not os.path.exists(path):
        return None
    p = np.load(path)
    return p.reshape(-1, 1) if p.ndim == 1 else p


def one_split(ds, variant, tag, plot=False):
    """Raw and calibrated test ECE for one model on one split, or None if unavailable."""
    y_va, y_te = load_labels(ds, variant, "valid"), load_labels(ds, variant, "test")
    p_va, p_te = load_preds(ds, variant, tag, "valid"), load_preds(ds, variant, tag, "test")
    if p_va is None or p_te is None:
        return None
    if p_va.shape[1] != y_va.shape[1] or p_te.shape[1] != y_te.shape[1]:
        return None

    cal_te = np.empty_like(p_te)
    n_fitted = 0
    for t in range(p_va.shape[1]):
        # A task with too few positives in the calibration split gets the identity map.
        # Fitting a two-parameter correction to a handful of positives produces a map that
        # is mostly noise, and it would be applied to every test molecule.
        if not enough_positives(y_va, t):
            choice = {"kind": "identity", "reason": "too few positives in valid"}
        else:
            ok = np.isfinite(y_va[:, t]) & np.isfinite(p_va[:, t])
            choice = choose_map(p_va[ok, t], y_va[ok, t])
            n_fitted += choice["kind"] != "identity"
        cal_te[:, t] = apply_map(p_te[:, t], choice)

    def mean_ece(y, p):
        vals = []
        for t in range(p.shape[1]):
            ok = np.isfinite(y[:, t])
            if ok.sum() and len(np.unique(y[ok, t])) > 1:
                vals.append(ece(y[ok, t], p[ok, t]))
        return float(np.nanmean(vals)) if vals else np.nan

    if plot:
        # The task with the most measured labels, as the representative one. A reliability
        # curve on a task with 30 positives is mostly binning artefact.
        t = int(np.argmax([(~np.isnan(y_te[:, k])).sum() for k in range(p_te.shape[1])]))
        ok = np.isfinite(y_te[:, t])
        if ok.sum() and len(np.unique(y_te[ok, t])) > 1:
            reliability_plot(ds, f"{tag}_{variant}", "test", y_te[ok, t], p_te[ok, t], "raw")
            reliability_plot(ds, f"{tag}_{variant}", "test", y_te[ok, t], cal_te[ok, t], "cal")

    return {"ece_raw": mean_ece(y_te, p_te), "ece_cal": mean_ece(y_te, cal_te),
            "n_tasks": p_te.shape[1], "n_maps_fitted": n_fitted}


def ci95(v):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return (float(v[0]), 0.0) if v.size else (np.nan, np.nan)
    return float(v.mean()), float(T_CRIT.get(v.size, 1.96) * v.std(ddof=1) / np.sqrt(v.size))


def main():
    ap = argparse.ArgumentParser(
        description="Expected calibration error across seeded splits, fitted on valid.")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=[f"seed{i}" for i in range(5)])
    ap.add_argument("--plots", action="store_true",
                    help="also write reliability curves (raw and calibrated) to "
                         "results/figs/, for the first variant only -- one curve per "
                         "(tag, dataset) is a figure, six is a wall")
    args = ap.parse_args()

    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    datasets = [d for d in (args.datasets or list(pool_index)) if is_classification(d)]

    rows, missing = [], []
    for tag in args.tags:
        for ds in datasets:
            per = [one_split(ds, v, tag, plot=(args.plots and v == args.variants[0]))
                   for v in args.variants]
            per = [r for r in per if r]
            if not per:
                missing.append(f"{tag}/{ds}")
                continue
            raw_m, raw_h = ci95([r["ece_raw"] for r in per])
            cal_m, cal_h = ci95([r["ece_cal"] for r in per])
            delta = [r["ece_cal"] - r["ece_raw"] for r in per]
            d_m, d_h = ci95(delta)
            rows.append({
                "tag": tag, "dataset": ds, "n_splits": len(per),
                "protocol": "valid-fit",
                "n_tasks": per[0]["n_tasks"],
                "ece_raw": raw_m, "ece_raw_ci": raw_h,
                "ece_cal": cal_m, "ece_cal_ci": cal_h,
                "delta": d_m, "delta_ci": d_h,
                # Positive delta means calibration made test ECE worse.
                "helps": bool(d_m < 0 and abs(d_m) > d_h),
                "hurts": bool(d_m > 0 and abs(d_m) > d_h),
            })

    if not rows:
        raise SystemExit(
            "No archived predictions for those tags. Per-split predictions live in "
            f"{RUNS_DIR}/<variant>/preds/ and exist only for runs archived after the "
            "predictions fix.")

    df = pd.DataFrame(rows)
    os.makedirs(MET_DIR, exist_ok=True)
    out = os.path.join(MET_DIR, "ece_multiseed.csv")
    df.to_csv(out, index=False)

    print(f"Test ECE, mean +/- 95% CI over {len(args.variants)} seeded splits.")
    print("Calibration map fitted on valid; test never fitted on. Lower is better.\n")
    print(f"  {'tag':<20}{'dataset':<15}{'raw':>16}{'calibrated':>18}{'change':>18}")
    print("  " + "-" * 85)
    for r in df.itertuples():
        verdict = "helps" if r.helps else ("HURTS" if r.hurts else "")
        print(f"  {r.tag:<20}{r.dataset:<15}"
              f"{r.ece_raw:>9.4f} +/-{r.ece_raw_ci:<5.4f}"
              f"{r.ece_cal:>10.4f} +/-{r.ece_cal_ci:<5.4f}"
              f"{r.delta:>+10.4f} +/-{r.delta_ci:<5.4f}  {verdict}")

    n_h, n_x = int(df.helps.sum()), int(df.hurts.sum())
    print(f"\n  Calibration decisively helps on {n_h} of {len(df)} (tag, dataset) pairs, "
          f"decisively hurts on {n_x}, and is inside the interval on {len(df) - n_h - n_x}.")
    if missing:
        print(f"\n  No archived predictions for: {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

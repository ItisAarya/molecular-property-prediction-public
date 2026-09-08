"""
src/eval/conformal.py

Split conformal prediction: turn a point prediction into a set or interval that contains
the truth a specified fraction of the time.

    python -m src.eval.conformal --tags rf gnn trf hybrid ens --alpha 0.1

WHAT IT GIVES YOU THAT CALIBRATION DOES NOT
-------------------------------------------
Calibration (Phase 0) makes a predicted probability mean what it says on average. It says
nothing about any individual molecule, and Phase 0 found it does not even transfer
reliably: fitted out-of-fold, post-hoc calibration sometimes made test ECE *worse* under
scaffold shift.

Conformal prediction makes a different, stronger promise. Instead of "0.83" it returns a
*set* -- for a regression target an interval in chemical units, for a binary task a subset
of {inactive, active} -- with a distribution-free guarantee that the truth falls inside at
least 1-alpha of the time. The guarantee needs no assumption about the model being correct,
only that calibration and test molecules are exchangeable.

That last condition is exactly what a scaffold split breaks, and measuring the breakage is
the point of this module rather than a caveat on it.

THE EXCHANGEABILITY PROBLEM, STATED PLAINLY
-------------------------------------------
Split conformal needs a calibration set the model has not been fitted on. The obvious
candidate here is `valid` -- but `valid` chose the early-stopping epoch, so predictions on
it are the model at its best rather than a fresh draw. The calibration scores are therefore
optimistically small, the quantile too tight, and coverage on test should fall *below*
nominal.

Worse, `test` is scaffold-disjoint from everything: the molecules are structurally
different by construction, so calibration and test are not exchangeable no matter which
split is used. A conformal interval calibrated on in-distribution data and applied under
covariate shift loses its guarantee.

So the coverage number this module reports is not a formality to be confirmed. It is a
measurement of how much the guarantee degrades under scaffold shift -- and reporting a
90% method that delivers, say, 82% is a more useful result for this field than reporting
a nominal number and calling the box ticked.

WHY THE QUANTILE HAS THE +1
---------------------------
With n calibration points the threshold is the ceil((n+1)(1-alpha))-th smallest score, not
the plain (1-alpha) empirical quantile. The correction accounts for the test point itself
being one of the exchangeable draws; without it the interval is slightly too narrow and
under-covers even when everything else is right. When that index exceeds n the data cannot
support the requested confidence and the honest answer is an unbounded interval, which is
what `float("inf")` here means.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import is_classification

DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pool")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
RUNS_DIR = os.path.join("results", "runs")
MET_DIR = os.path.join("results", "metrics")


def conformal_quantile(scores, alpha):
    """
    The split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score.

    Returns inf when n is too small to support the requested confidence -- with 9
    calibration points you cannot promise 95%, and pretending otherwise is how a
    guarantee quietly becomes a decoration.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def regression_intervals(y_cal, p_cal, y_test, p_test, alpha=0.1):
    """
    Absolute-residual split conformal. Inputs and outputs are in chemical units.

    Returns coverage, mean interval width, and the half-width itself, which is constant
    across molecules for this score -- a deliberately simple baseline. A width that is the
    same for every molecule is uninformative about *which* predictions to distrust, which
    is the motivation for the normalised and quantile variants noted at the bottom.
    """
    ok_cal = np.isfinite(y_cal) & np.isfinite(p_cal)
    q = conformal_quantile(np.abs(y_cal[ok_cal] - p_cal[ok_cal]), alpha)

    ok = np.isfinite(y_test) & np.isfinite(p_test)
    if not ok.any():
        return {"coverage": np.nan, "mean_width": np.nan, "half_width": q, "n": 0}
    covered = np.abs(y_test[ok] - p_test[ok]) <= q
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(2 * q) if np.isfinite(q) else float("inf"),
        "half_width": q,
        "n": int(ok.sum()),
    }


def binary_sets(y_cal, p_cal, y_test, p_test, alpha=0.1):
    """
    Split conformal for one binary task.

    The score for a labelled molecule is 1 - p(its true class), so a confident correct
    prediction scores near 0 and a confident wrong one near 1. A class enters the
    prediction set when its own score falls at or below the threshold, which yields sets
    of size 0, 1 or 2.

    Size 2 means the model cannot rule either class out at this confidence -- the useful
    output, since it flags the molecules a screen should not act on unaided. Size 0 means
    both classes were ruled out, which is possible for a score threshold below both and
    signals a molecule unlike anything in calibration.
    """
    ok_cal = np.isfinite(y_cal) & np.isfinite(p_cal)
    yc, pc = y_cal[ok_cal], np.clip(p_cal[ok_cal], 0.0, 1.0)
    if yc.size == 0:
        return None
    # 1 - p(true class)
    q = conformal_quantile(np.where(yc == 1, 1.0 - pc, pc), alpha)

    ok = np.isfinite(y_test) & np.isfinite(p_test)
    if not ok.any():
        return None
    yt, pt = y_test[ok], np.clip(p_test[ok], 0.0, 1.0)

    in_pos = (1.0 - pt) <= q          # "active" admitted
    in_neg = pt <= q                  # "inactive" admitted
    sizes = in_pos.astype(int) + in_neg.astype(int)
    covered = np.where(yt == 1, in_pos, in_neg)

    return {
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "pct_ambiguous": float((sizes == 2).mean() * 100),
        "pct_empty": float((sizes == 0).mean() * 100),
        "n": int(ok.sum()),
    }


# --------------------------------------------------------------------------------------
# Loading labels and predictions for one (dataset, variant, tag)
# --------------------------------------------------------------------------------------
def load_labels(ds, variant, raw=True):
    """Labels for valid and test, sliced from the pool by this variant's indices."""
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")))
    key = "y_raw" if raw else "y"
    return {sp: pool[key][np.asarray(idx[sp], dtype=int)] for sp in ("valid", "test")}


def label_scale(ds):
    """(mean, std) per task, for putting normalised predictions back into chemical units."""
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    return np.asarray(pool["y_mean"], dtype=float), np.asarray(pool["y_std"], dtype=float)


def load_preds(ds, variant, tag):
    """Archived per-split predictions, or None if this model was not archived."""
    out = {}
    for sp in ("valid", "test"):
        path = os.path.join(RUNS_DIR, variant, "preds", f"{ds}_{tag}_{sp}.npy")
        if not os.path.exists(path):
            return None
        out[sp] = np.load(path)
    return out


def evaluate(ds, variant, tag, alpha):
    """Conformal coverage for one model on one split variant."""
    preds = load_preds(ds, variant, tag)
    if preds is None:
        return None
    cls = is_classification(ds)

    if cls:
        y = load_labels(ds, variant, raw=False)
        rows = []
        n_tasks = y["valid"].shape[1]
        for t in range(n_tasks):
            r = binary_sets(y["valid"][:, t], preds["valid"][:, t],
                            y["test"][:, t], preds["test"][:, t], alpha)
            if r:
                rows.append(r)
        if not rows:
            return None
        return {
            "coverage": float(np.mean([r["coverage"] for r in rows])),
            "mean_set_size": float(np.mean([r["mean_set_size"] for r in rows])),
            "pct_ambiguous": float(np.mean([r["pct_ambiguous"] for r in rows])),
            "pct_empty": float(np.mean([r["pct_empty"] for r in rows])),
            "n_tasks_scored": len(rows),
        }

    # Regression: work in chemical units so the width means something to a chemist.
    mean, std = label_scale(ds)
    y = load_labels(ds, variant, raw=True)
    conv = lambda p: p.reshape(len(p), -1)[:, 0] * std[0] + mean[0]
    r = regression_intervals(y["valid"][:, 0], conv(preds["valid"]),
                             y["test"][:, 0], conv(preds["test"]), alpha)
    return r


def main():
    ap = argparse.ArgumentParser(
        description="Split-conformal coverage for archived model predictions.")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+",
                    default=[f"seed{i}" for i in range(5)])
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="miss rate; 0.1 means a nominal 90%% coverage target")
    args = ap.parse_args()

    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    datasets = args.datasets or list(pool_index)
    nominal = 100 * (1 - args.alpha)

    rows, missing = [], []
    for tag in args.tags:
        for ds in datasets:
            per_split = [evaluate(ds, v, tag, args.alpha) for v in args.variants]
            per_split = [r for r in per_split if r]
            if not per_split:
                missing.append(f"{tag}/{ds}")
                continue
            agg = {"tag": tag, "dataset": ds, "n_splits": len(per_split),
                   "task_type": pool_index[ds]["task_type"]}
            for k in per_split[0]:
                agg[k] = float(np.mean([r[k] for r in per_split]))
            agg["coverage_gap"] = agg["coverage"] * 100 - nominal
            rows.append(agg)

    if not rows:
        raise SystemExit(
            "No archived predictions found for those tags. Per-split predictions live in "
            f"{RUNS_DIR}/<variant>/preds/ and are only written for runs archived after the "
            "predictions fix; earlier view and fusion runs saved metrics only.")

    df = pd.DataFrame(rows)
    print(f"Split conformal, nominal coverage {nominal:.0f}%, "
          f"mean over {len(args.variants)} seeded splits\n")
    for tag in args.tags:
        sub = df[df.tag == tag]
        if sub.empty:
            continue
        print(f"  {tag}")
        for _, r in sub.iterrows():
            if r.task_type == "classification":
                extra = (f"set size {r.mean_set_size:.2f}  "
                         f"ambiguous {r.pct_ambiguous:5.1f}%  empty {r.pct_empty:4.1f}%")
            else:
                extra = f"interval width {r.mean_width:.3f} (chemical units)"
            flag = "" if abs(r.coverage_gap) <= 2 else "   <- off nominal"
            print(f"    {r.dataset:<15} coverage {r.coverage * 100:5.1f}% "
                  f"({r.coverage_gap:+5.1f})  {extra}{flag}")
        print()

    os.makedirs(MET_DIR, exist_ok=True)
    out = os.path.join(MET_DIR, f"conformal_alpha{args.alpha:g}.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    if missing:
        print(f"\nNo archived predictions for: {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""))


if __name__ == "__main__":
    main()

"""
src/eval/conformal.py

Split conformal prediction: turn a point prediction into a set or interval that contains
the truth a specified fraction of the time.

    python -m src.eval.conformal --tags rf gnn trf hybrid ens
    python -m src.eval.conformal --tags ens --score normalized
    python -m src.eval.conformal --tags ens --conditional
    python -m src.eval.conformal --tags ens --by-distance

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

THREE WAYS THE GUARANTEE CAN BE TRUE AND USELESS
------------------------------------------------
Coverage hitting its target is necessary, not sufficient. Each option below exists because
the plain version hides a different failure:

1. **Constant width** (`--score absolute`). The absolute-residual score gives every
   molecule the same interval, so it cannot say which predictions to distrust. On ESOL that
   interval is +/-2 to +/-3 logS against a best-model RMSE near 0.77 -- correct and
   unusable. `--score normalized` divides the residual by a per-molecule difficulty
   estimate, so width varies with how hard the molecule is.

2. **Marginal coverage on imbalanced data** (`--conditional`). These datasets are ~6-9%
   positive. A marginal 90% guarantee can be met by covering the negatives, which are most
   of the data, while the actives -- the only class a toxicity screen cares about -- are
   covered far less. Per-class coverage is therefore always reported, and `--conditional`
   fits a separate quantile per class so the guarantee holds *within* each.

3. **Averaging over a shifted population** (`--by-distance`). A scaffold split deliberately
   puts structurally novel compounds in test. Coverage can be met on the familiar ones and
   missed on the rest, which inverts the point of the split. Splitting coverage by Tanimoto
   distance to the nearest training molecule turns that into a measurement.

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

from src.eval.calibration import apply_map, fit_logistic, fit_temperature
from src.eval.metrics import is_classification
from src.eval.similarity import distance_bins, nearest_train_similarity

DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pool")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
RUNS_DIR = os.path.join("results", "runs")
MET_DIR = os.path.join("results", "metrics")

# Models whose disagreement estimates per-molecule difficulty for the normalised score.
DISAGREEMENT_TAGS = ("rf", "gnn", "trf")


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


# --------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------
def regression_intervals(y_cal, p_cal, y_test, p_test, alpha=0.1,
                         sigma_cal=None, sigma_test=None, mask_test=None):
    """
    Split conformal for a regression target, in chemical units.

    With `sigma_*` supplied the score is the *normalised* residual |y - p| / sigma, so the
    interval becomes p +/- q*sigma(x) and its width varies per molecule: wide where the
    difficulty estimate is high, narrow where it is low. The marginal guarantee is
    unchanged -- what changes is that the width now carries information.
    """
    ok_cal = np.isfinite(y_cal) & np.isfinite(p_cal)
    res_cal = np.abs(y_cal[ok_cal] - p_cal[ok_cal])
    if sigma_cal is not None:
        res_cal = res_cal / sigma_cal[ok_cal]
    q = conformal_quantile(res_cal, alpha)

    ok = np.isfinite(y_test) & np.isfinite(p_test)
    if mask_test is not None:
        ok = ok & mask_test
    if not ok.any():
        return None

    half = q * sigma_test[ok] if sigma_test is not None else np.full(int(ok.sum()), q)
    covered = np.abs(y_test[ok] - p_test[ok]) <= half
    return {
        "coverage": float(covered.mean()),
        "mean_width": float(np.mean(2 * half)),
        "width_spread": float(np.std(2 * half)),
        "n": int(ok.sum()),
    }


def disagreement_sigma(ds, variant, split, n_rows):
    """
    Per-molecule difficulty, estimated from how much the base models disagree.

    Standard practice normalises the residual by a second model trained to predict error
    magnitude. That would mean another model per dataset per split. We already have three
    base models' predictions on every split, and their spread is a serviceable proxy: where
    a Random Forest, a GNN and a transformer agree, the molecule is easy; where they do
    not, it is not. Costs nothing beyond reading files that already exist.

    Returns None when fewer than two base models were archived, in which case the caller
    falls back to the constant-width score rather than inventing a difficulty estimate.
    """
    preds = []
    for tag in DISAGREEMENT_TAGS:
        path = os.path.join(RUNS_DIR, variant, "preds", f"{ds}_{tag}_{split}.npy")
        if os.path.exists(path):
            arr = np.load(path).reshape(n_rows, -1)[:, 0]
            preds.append(arr)
    if len(preds) < 2:
        return None
    sigma = np.std(np.stack(preds), axis=0)
    # A floor keeps the division finite where every model agrees exactly; without it a
    # single unanimous molecule produces an infinite normalised score and swallows the
    # quantile.
    return np.maximum(sigma, 1e-3)


# --------------------------------------------------------------------------------------
# Binary classification
# --------------------------------------------------------------------------------------
def binary_sets(y_cal, p_cal, y_test, p_test, alpha=0.1, conditional=False,
                mask_test=None):
    """
    Split conformal for one binary task, reporting coverage within each class.

    The score for a labelled molecule is 1 - p(its true class). A class enters the
    prediction set when its own score falls at or below the threshold, giving sets of size
    0, 1 or 2.

    `conditional=True` fits a separate threshold per class (Mondrian conformal). The
    marginal version guarantees coverage averaged over classes, which on data that is 6-9%
    positive is dominated by the negatives; the conditional version guarantees it for
    actives and inactives separately, at the cost of larger sets.
    """
    ok_cal = np.isfinite(y_cal) & np.isfinite(p_cal)
    yc, pc = y_cal[ok_cal], np.clip(p_cal[ok_cal], 0.0, 1.0)
    if yc.size == 0:
        return None
    score_cal = np.where(yc == 1, 1.0 - pc, pc)

    if conditional:
        # Too few of a class to calibrate it is a real possibility here -- SIDER's rarest
        # task has 22 positives in 1,427 molecules -- and an infinite threshold is the
        # correct answer: that class is then always admitted, which is honest rather than
        # silently under-covered.
        q_pos = conformal_quantile(score_cal[yc == 1], alpha)
        q_neg = conformal_quantile(score_cal[yc == 0], alpha)
    else:
        q_pos = q_neg = conformal_quantile(score_cal, alpha)

    ok = np.isfinite(y_test) & np.isfinite(p_test)
    if mask_test is not None:
        ok = ok & mask_test
    if not ok.any():
        return None
    yt, pt = y_test[ok], np.clip(p_test[ok], 0.0, 1.0)

    in_pos = (1.0 - pt) <= q_pos
    in_neg = pt <= q_neg
    sizes = in_pos.astype(int) + in_neg.astype(int)
    covered = np.where(yt == 1, in_pos, in_neg)

    is_pos, is_neg = yt == 1, yt == 0
    return {
        "coverage": float(covered.mean()),
        "coverage_pos": float(covered[is_pos].mean()) if is_pos.any() else np.nan,
        "coverage_neg": float(covered[is_neg].mean()) if is_neg.any() else np.nan,
        "mean_set_size": float(sizes.mean()),
        "pct_ambiguous": float((sizes == 2).mean() * 100),
        "pct_empty": float((sizes == 0).mean() * 100),
        "n": int(ok.sum()),
        "n_pos": int(is_pos.sum()),
    }


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_labels(ds, variant, raw=True):
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")))
    key = "y_raw" if raw else "y"
    return {sp: pool[key][np.asarray(idx[sp], dtype=int)] for sp in ("valid", "test")}


def label_scale(ds):
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    return np.asarray(pool["y_mean"], dtype=float), np.asarray(pool["y_std"], dtype=float)


def load_preds(ds, variant, tag):
    out = {}
    for sp in ("valid", "test"):
        path = os.path.join(RUNS_DIR, variant, "preds", f"{ds}_{tag}_{sp}.npy")
        if not os.path.exists(path):
            return None
        out[sp] = np.load(path)
    return out


def recalibrate(y_cal, p_cal, p_test, kind):
    """
    Fit a probability calibration map on the calibration split, apply it to both.

    TEMPERATURE SCALING CANNOT CHANGE A CONFORMAL SET. Verified numerically and true by
    construction: the conformal score is 1-p for an active and p for an inactive, and the
    sigmoid temperature map is symmetric about 0.5, g(1-p) = 1-g(p) to machine precision.
    So both classes' scores pass through the *same* strictly increasing map, the quantile
    moves with them, and every `score <= q` comparison is preserved. Split conformal is
    invariant to monotone transformations of its score, so the sets come back bit for bit
    identical even though the probabilities themselves changed by up to 0.02.

    It is kept as an option precisely to make that visible: "calibrate, then conformalise"
    is a natural thing to try and, for binary tasks with this score, it is wasted effort.

    Platt/logistic scaling is different. Its intercept breaks the symmetry, so the two
    classes' scores move differently and the sets genuinely change -- substantially, in
    favour of the minority class, because `fit_logistic` is fitted with balanced class
    weights. That is a real effect but an accidental one: `--conditional` reaches the same
    place deliberately and keeps the guarantee.
    """
    ok = np.isfinite(y_cal) & np.isfinite(p_cal)
    if kind == "none" or ok.sum() < 10 or len(np.unique(y_cal[ok])) < 2:
        return p_cal, p_test
    try:
        if kind == "temperature":
            method = {"kind": "temperature", "T": fit_temperature(p_cal[ok], y_cal[ok])}
        else:
            fit = fit_logistic(p_cal[ok], y_cal[ok])
            if fit is None:
                return p_cal, p_test
            method = {"kind": "logistic", "a": fit[0], "b": fit[1]}
        return apply_map(p_cal, method), apply_map(p_test, method)
    except Exception:
        # A fit that will not converge is not worth failing the run over; the
        # uncalibrated probabilities are a valid conformal input either way.
        return p_cal, p_test


def evaluate(ds, variant, tag, alpha, score="absolute", conditional=False, mask_test=None,
             calibrate="none"):
    """Conformal behaviour for one model on one split variant."""
    preds = load_preds(ds, variant, tag)
    if preds is None:
        return None

    if is_classification(ds):
        y = load_labels(ds, variant, raw=False)
        rows = []
        for t in range(y["valid"].shape[1]):
            pc, pt = preds["valid"][:, t], preds["test"][:, t]
            if calibrate != "none":
                pc, pt = recalibrate(y["valid"][:, t], pc, pt, calibrate)
            r = binary_sets(y["valid"][:, t], pc, y["test"][:, t], pt,
                            alpha, conditional=conditional, mask_test=mask_test)
            if r:
                rows.append(r)
        if not rows:
            return None
        keys = ["coverage", "coverage_pos", "coverage_neg", "mean_set_size",
                "pct_ambiguous", "pct_empty"]
        out = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        out["n_tasks_scored"] = len(rows)
        out["n"] = rows[0]["n"]
        return out

    mean, std = label_scale(ds)
    y = load_labels(ds, variant, raw=True)
    conv = lambda p: p.reshape(len(p), -1)[:, 0] * std[0] + mean[0]
    sig_cal = sig_test = None
    if score == "normalized":
        sc = disagreement_sigma(ds, variant, "valid", preds["valid"].shape[0])
        st = disagreement_sigma(ds, variant, "test", preds["test"].shape[0])
        if sc is not None and st is not None:
            sig_cal, sig_test = sc * std[0], st * std[0]
    return regression_intervals(y["valid"][:, 0], conv(preds["valid"]),
                                y["test"][:, 0], conv(preds["test"]),
                                alpha, sig_cal, sig_test, mask_test)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def by_distance(ds, variant, tag, alpha, score, conditional, calibrate="none"):
    """Coverage split by Tanimoto distance from the nearest training molecule."""
    sim = nearest_train_similarity(ds, variant, "test")
    out = []
    for label, mask in distance_bins(sim):
        if mask.sum() < 10:
            continue
        r = evaluate(ds, variant, tag, alpha, score, conditional, mask_test=mask,
                     calibrate=calibrate)
        if r:
            r["band"] = label
            r["n_band"] = int(mask.sum())
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Split-conformal coverage for archived model predictions.")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=[f"seed{i}" for i in range(5)])
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="miss rate; 0.1 means a nominal 90%% coverage target")
    ap.add_argument("--score", default="absolute", choices=["absolute", "normalized"],
                    help="regression score: constant width, or width scaled by a "
                         "per-molecule difficulty estimate from base-model disagreement")
    ap.add_argument("--conditional", action="store_true",
                    help="fit a separate quantile per class, so coverage holds within "
                         "each class rather than only on average")
    ap.add_argument("--calibrate", default="none",
                    choices=["none", "temperature", "logistic"],
                    help="classification only: recalibrate probabilities on the "
                         "calibration split before conformal. `temperature` provably "
                         "changes nothing (see recalibrate); `logistic` does.")
    ap.add_argument("--by-distance", action="store_true",
                    help="report coverage by Tanimoto similarity to the nearest "
                         "training molecule")
    args = ap.parse_args()

    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    datasets = args.datasets or list(pool_index)
    nominal = 100 * (1 - args.alpha)

    rows, missing = [], []
    for tag in args.tags:
        for ds in datasets:
            cls = pool_index[ds]["task_type"] == "classification"
            if args.by_distance:
                per = []
                for v in args.variants:
                    per += by_distance(ds, v, tag, args.alpha, args.score,
                                       args.conditional, args.calibrate)
                if not per:
                    missing.append(f"{tag}/{ds}")
                    continue
                for band in sorted({r["band"] for r in per}):
                    sub = [r for r in per if r["band"] == band]
                    agg = {"tag": tag, "dataset": ds, "band": band,
                           "task_type": pool_index[ds]["task_type"],
                           "n_band": float(np.mean([r["n_band"] for r in sub]))}
                    for k in sub[0]:
                        if k not in ("band",):
                            agg[k] = float(np.nanmean([r[k] for r in sub]))
                    agg["coverage_gap"] = agg["coverage"] * 100 - nominal
                    rows.append(agg)
                continue

            per = [evaluate(ds, v, tag, args.alpha, args.score, args.conditional,
                            calibrate=args.calibrate) for v in args.variants]
            per = [r for r in per if r]
            if not per:
                missing.append(f"{tag}/{ds}")
                continue
            agg = {"tag": tag, "dataset": ds, "n_splits": len(per),
                   "task_type": pool_index[ds]["task_type"]}
            for k in per[0]:
                agg[k] = float(np.nanmean([r[k] for r in per]))
            agg["coverage_gap"] = agg["coverage"] * 100 - nominal
            rows.append(agg)

    if not rows:
        raise SystemExit(
            "No archived predictions found for those tags. Per-split predictions live in "
            f"{RUNS_DIR}/<variant>/preds/ and are only written for runs archived after the "
            "predictions fix; earlier view and fusion runs saved metrics only.")

    df = pd.DataFrame(rows)
    mode = ("by distance" if args.by_distance
            else ("class-conditional" if args.conditional else f"{args.score} score"))
    print(f"Split conformal [{mode}], nominal {nominal:.0f}%, "
          f"mean over {len(args.variants)} seeded splits\n")

    for tag in args.tags:
        sub = df[df.tag == tag]
        if sub.empty:
            continue
        print(f"  {tag}")
        for _, r in sub.iterrows():
            head = f"{r.dataset:<15}"
            if args.by_distance:
                head = f"{r.dataset:<15} sim {r.band:<8} n={r.n_band:>5.0f}  "
            if r.task_type == "classification":
                extra = (f"actives {r.coverage_pos * 100:5.1f}%  "
                         f"inactives {r.coverage_neg * 100:5.1f}%  "
                         f"set {r.mean_set_size:.2f}")
            else:
                extra = f"width {r.mean_width:.3f} (sd {r.width_spread:.3f})"
            print(f"    {head}coverage {r.coverage * 100:5.1f}% "
                  f"({r.coverage_gap:+5.1f})  {extra}")
        print()

    os.makedirs(MET_DIR, exist_ok=True)
    suffix = (f"_{args.calibrate}" if args.calibrate != "none" else "") + (
             "_bydistance" if args.by_distance
             else ("_conditional" if args.conditional else f"_{args.score}"))
    out = os.path.join(MET_DIR, f"conformal_alpha{args.alpha:g}{suffix}.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    if missing:
        print(f"\nNo archived predictions for: {', '.join(missing[:8])}"
              + (" ..." if len(missing) > 8 else ""))


if __name__ == "__main__":
    main()

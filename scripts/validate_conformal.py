"""
scripts/validate_conformal.py

Two checks on `src/eval/conformal.py` that section 6 of the paper quotes.

    python -m scripts.validate_conformal
    python -m scripts.validate_conformal --tags rf gnn ens --datasets tox21 bbbp clintox

1. SYNTHETIC DATA. Before trusting a coverage number on MoleculeNet, the implementation has
   to hit its target where the guarantee is known to hold: calibration and test points drawn
   from one distribution. Regression (absolute residual), marginal binary LAC and
   class-conditional (Mondrian) binary LAC are run at a 90% target on data with 8% positives,
   the imbalance that matters on Tox21, and averaged over repeated draws.

2. TEMPERATURE SCALING DOES NOT CHANGE A BINARY LAC SET. For every model, dataset, seeded split
   and task with archived predictions, fit a temperature on the calibration split, build the
   sets with and without it, and compare them molecule by molecule. Reports how far the
   probabilities themselves moved, so "identical sets" cannot be mistaken for "nothing
   happened". Missing predictions are skipped and reported: predictions are git-ignored, so
   a fresh clone has only the models it has re-trained.

Writes `results/metrics/conformal_validation.csv`.
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.eval.calibration import apply_map, fit_temperature
from src.eval.conformal import binary_sets, load_labels, load_preds, regression_intervals

MET = os.path.join("results", "metrics")
SEEDS = [f"seed{i}" for i in range(5)]


def synthetic(n_cal=1000, n_test=1000, reps=500, alpha=0.1, pos_rate=0.08, seed=0):
    """Mean coverage over `reps` exchangeable draws, for the three procedures section 6 uses."""
    rng = np.random.default_rng(seed)
    reg, marg, cond_pos, cond_neg = [], [], [], []
    for _ in range(reps):
        # Regression: y = f(x) + noise, predictions from an imperfect model.
        x = rng.normal(size=n_cal + n_test)
        y = 2 * x + rng.normal(scale=1 + 0.5 * np.abs(x))
        p = 1.8 * x
        reg.append(regression_intervals(y[:n_cal], p[:n_cal], y[n_cal:], p[n_cal:], alpha)["coverage"])
        # Binary: a label with 8% positives and a noisy but informative probability.
        lab = (rng.uniform(size=n_cal + n_test) < pos_rate).astype(float)
        prob = 1 / (1 + np.exp(-(rng.normal(size=lab.size) + 2.0 * lab - 2.5)))
        m = binary_sets(lab[:n_cal], prob[:n_cal], lab[n_cal:], prob[n_cal:], alpha)
        c = binary_sets(lab[:n_cal], prob[:n_cal], lab[n_cal:], prob[n_cal:], alpha, conditional=True)
        marg.append(m["coverage"])
        cond_pos.append(c["coverage_pos"])
        cond_neg.append(c["coverage_neg"])
    return {"regression": np.mean(reg), "binary_marginal": np.mean(marg),
            "binary_conditional_actives": np.nanmean(cond_pos),
            "binary_conditional_inactives": np.nanmean(cond_neg)}


def temperature_invariance(tag, ds, alpha=0.1):
    """(sets identical everywhere?, max |p_T - p| on test, task-splits compared) or None."""
    same, moved, n = True, 0.0, 0
    for v in SEEDS:
        preds = load_preds(ds, v, tag)
        if preds is None:
            return None
        y = load_labels(ds, v, raw=False)
        for t in range(y["valid"].shape[1]):
            yc, pc = y["valid"][:, t], np.clip(preds["valid"][:, t], 0, 1)
            yt, pt = y["test"][:, t], np.clip(preds["test"][:, t], 0, 1)
            ok = np.isfinite(yc)
            if ok.sum() < 10 or len(np.unique(yc[ok])) < 2:
                continue
            m = {"kind": "temperature", "T": fit_temperature(pc[ok], yc[ok])}
            pc_t, pt_t = apply_map(pc, m), apply_map(pt, m)
            a = binary_sets(yc, pc, yt, pt, alpha)
            b = binary_sets(yc, pc_t, yt, pt_t, alpha)
            if a is None or b is None:
                continue
            # Same coverage and size can hide different sets; compare membership directly.
            keep = np.isfinite(yt)
            q0 = np.sort(np.where(yc[ok] == 1, 1 - pc[ok], pc[ok]))
            q1 = np.sort(np.where(yc[ok] == 1, 1 - pc_t[ok], pc_t[ok]))
            k = int(np.ceil((ok.sum() + 1) * (1 - alpha))) - 1
            s0 = np.stack([1 - pt[keep] <= q0[k], pt[keep] <= q0[k]])
            s1 = np.stack([1 - pt_t[keep] <= q1[k], pt_t[keep] <= q1[k]])
            same &= bool((s0 == s1).all())
            moved = max(moved, float(np.abs(pt_t[keep] - pt[keep]).max()))
            n += 1
    return same, moved, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["rf", "gnn", "ens"])
    ap.add_argument("--datasets", nargs="+", default=["tox21", "bbbp", "clintox"])
    args = ap.parse_args()

    rows = []
    syn = synthetic()
    print("Synthetic exchangeable data, 90% target, 8% positives (mean over 500 draws):")
    for k, v in syn.items():
        print(f"  {k:<30} {100 * v:.1f}%")
        rows.append({"check": "synthetic", "item": k, "value": v})

    print("\nTemperature scaling before binary LAC conformal, per model and dataset:")
    for tag in args.tags:
        for ds in args.datasets:
            r = temperature_invariance(tag, ds)
            if r is None:
                print(f"  {tag:<6} {ds:<8} predictions not found -- skipped")
                continue
            same, moved, n = r
            print(f"  {tag:<6} {ds:<8} sets identical: {same}   max |p_T - p| = {moved:.3f}   "
                  f"({n} task-splits)")
            rows.append({"check": "temperature", "item": f"{tag}/{ds}", "value": moved,
                         "identical": same, "task_splits": n})
    os.makedirs(MET, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(MET, "conformal_validation.csv"), index=False)
    print(f"\nWrote {MET}/conformal_validation.csv")


if __name__ == "__main__":
    main()

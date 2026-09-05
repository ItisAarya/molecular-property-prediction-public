"""
src/eval/view_stats.py

Compare single-view encoders across the seeded scaffold splits.

    python -m src.eval.view_stats --a gine --b gin_ref

Reads results/runs/<variant>/metrics/<ds>_<tag>_test.csv written by
scripts/run_view_multiseed.py and reports, per dataset, each encoder's mean with a 95%
confidence interval plus a paired comparison across matched splits.

WHY PAIRED, AND WHY THIS SEPARATE FROM stats.py
-----------------------------------------------
`src/eval/stats.py` aggregates the full pipeline (rf / gnn / trf / hybrid / ens) from
`final_report_thresholded.csv`. Single-view encoders are trained outside that pipeline --
they are candidate *components*, not pipeline models -- and write their own per-tag CSVs.
Same statistics, different input, so it lives in its own module rather than complicating
the pipeline one.

Splits differ in difficulty far more than encoders differ from each other: a hard partition
drags both models down together. Comparing them *within* each split removes that shared
variance, which is the only reason a five-point comparison has any power at all.

READING THE OUTPUT
------------------
With n = 5 the Wilcoxon signed-rank test cannot produce a p-value below 0.0625, so a clean
5-0 sweep still does not reach 0.05. The paired t-test and Cohen's dz are printed alongside,
and the practical question -- is the mean difference larger than the interval? -- is
answered explicitly, because an architecture that costs 5x the parameters needs to earn its
place, not merely avoid being disproven.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.eval.metrics import is_classification

RESULTS = "results"
RUNS_DIR = os.path.join(RESULTS, "runs")
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def _is_cls(ds):
    return is_classification(ds)


def seeded_variants():
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted(v for v in os.listdir(RUNS_DIR) if v.startswith("seed"))


def load_tag(variant, ds, tag):
    """Test-set metric for one encoder on one dataset in one split, or None if absent."""
    path = os.path.join(RUNS_DIR, variant, "metrics", f"{ds}_{tag}_test.csv")
    if not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0]
    key = "auc" if _is_cls(ds) else "rmse"
    return float(row[key])


def ci95(v):
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 2:
        return float(v.mean()) if n else np.nan, 0.0, np.nan
    sd = float(v.std(ddof=1))
    return float(v.mean()), sd, T_CRIT.get(n, 1.96) * sd / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="gine", help="candidate encoder")
    ap.add_argument("--b", default="gin_ref", help="reference encoder")
    ap.add_argument("--datasets", nargs="+",
                    default=["tox21", "bbbp", "clintox", "esol", "lipophilicity"])
    args = ap.parse_args()

    variants = seeded_variants()
    if not variants:
        raise SystemExit("No seeded runs. Run: python -m scripts.run_view_multiseed")
    print(f"Seeded splits: {variants}\n")

    print(f"{'=' * 84}")
    print(f"{args.a}  vs  {args.b}   (paired across {len(variants)} seeded scaffold splits)")
    print(f"{'=' * 84}\n")

    rows = []
    for ds in args.datasets:
        cls = _is_cls(ds)
        key, higher = ("AUC", True) if cls else ("RMSE", False)

        pairs = [(load_tag(v, ds, args.a), load_tag(v, ds, args.b)) for v in variants]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if len(pairs) < 3:
            print(f"  {ds}: only {len(pairs)} paired split(s) -- skipping\n")
            continue

        va = np.array([p[0] for p in pairs])
        vb = np.array([p[1] for p in pairs])
        ma, sa, ha = ci95(va)
        mb, sb, hb = ci95(vb)

        diff = (va - vb) if higher else (vb - va)   # positive = candidate is better
        wins = int((diff > 0).sum())
        p_w = float(sps.wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else 1.0
        p_t = float(sps.ttest_rel(va, vb).pvalue) if not np.allclose(diff, 0) else 1.0
        sd = diff.std(ddof=1)
        dz = float(diff.mean() / sd) if sd > 0 else np.nan

        # Is the *paired difference* distinguishable from zero?
        #
        # This used to compare the mean difference against `hb`, the reference model's own
        # across-split spread. That is the wrong yardstick. Pairing is the entire point of
        # running both encoders on the same five splits: the split-to-split variation is
        # shared, so it cancels in the difference, and the difference has a far smaller
        # standard error than either model's scores do. Testing against `hb` made the
        # verdict depend on how noisy the *reference* happened to be on that dataset --
        # seq_frozen lost tox21 0-5 with a paired p of 0.018 and was called "inside noise",
        # while a weaker sider result (p=0.027) was called "degrades", purely because
        # gin_ref was more stable on sider.
        #
        # The correct comparison is the mean difference against its own 95% interval,
        # which is equivalent to the paired t-test at the same alpha.
        md, sd_diff, hd = ci95(diff)
        decisive = abs(md) > hd
        verdict = ("improves" if md > 0 else "degrades") if decisive else "inside noise"

        # Statistical significance is not the same as mattering. Phase 0 measured the
        # baseline's own across-seed intervals at roughly +/-0.02 AUC and +/-0.10 RMSE, so
        # an effect below that is real but small relative to how much the benchmark moves
        # between splits. Flagged rather than folded into the verdict: where to draw that
        # line is a reporting decision, not a statistical one.
        mde = 0.02 if cls else 0.10
        below_mde = decisive and abs(md) < mde

        print(f"  {ds}  ({key})")
        print(f"      {args.a:<10}{ma:.4f}  +/- {ha:.4f}")
        print(f"      {args.b:<10}{mb:.4f}  +/- {hb:.4f}")
        print(f"      mean diff {md:+.4f} +/- {hd:.4f}   wins {wins}-{len(pairs) - wins}   "
              f"p(wilcox)={p_w:.4f}  p(t)={p_t:.4f}  dz={dz:+.2f}")
        note = f"  (below the +/-{mde} practical threshold)" if below_mde else ""
        print(f"      -> {verdict}{note}\n")

        rows.append({
            "dataset": ds, "metric": key.lower(), "a": args.a, "b": args.b,
            "n_splits": len(pairs), "mean_a": ma, "ci_a": ha, "mean_b": mb, "ci_b": hb,
            "mean_diff": md, "ci_diff": hd, "a_wins": wins,
            "p_wilcoxon": p_w, "p_ttest": p_t, "cohens_dz": dz,
            "verdict": verdict, "below_mde": below_mde,
        })

    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(RESULTS, "metrics", f"view_compare_{args.a}_vs_{args.b}.csv")
        df.to_csv(out, index=False)

        n_better = int((df.verdict == "improves").sum())
        n_worse = int((df.verdict == "degrades").sum())
        n_noise = int((df.verdict == "inside noise").sum())
        print(f"{'=' * 84}")
        print(f"  {args.a} improves on {n_better}/{len(df)} datasets, "
              f"degrades on {n_worse}, indistinguishable on {n_noise}.")
        n_small = int(df.below_mde.sum())
        print(f"  With n={len(variants)} the Wilcoxon floor is p=0.0625; a 5-0 sweep is not p<0.05.")
        print("  Verdicts come from the paired difference's own 95% interval (= paired t).")
        if n_small:
            print(f"  {n_small} of those are statistically clear but below the practical "
                  f"threshold (+/-0.02 AUC, +/-0.10 RMSE).")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

"""
src/eval/stats.py

Aggregate the per-split runs into means, confidence intervals and paired significance
tests.

    python -m src.eval.stats                      # summary table + pairwise tests
    python -m src.eval.stats --compare ens hybrid # one specific comparison

Reads results/runs/<variant>/metrics/final_report_thresholded.csv for every seeded variant
and writes:
    results/metrics/multiseed_summary.csv   mean, std, 95% CI per dataset/model/metric
    results/metrics/multiseed_tests.csv     paired comparisons between models

WHY THIS IS THE POINT OF PHASE 0
--------------------------------
Everything before this produced single numbers. A single number cannot answer the only
question that matters for the paper: when the fusion model reports a higher AUC than the
baseline, is that a real effect or the luck of one partition?

Five seeded scaffold splits turn each metric into a small sample, which supports both an
interval and a test.

WHY A t-INTERVAL AND NOT +/- 1.96 SIGMA
---------------------------------------
With n = 5 the normal approximation is too tight. The interval uses the Student t critical
value for n - 1 = 4 degrees of freedom (t = 2.776 at 95%), which is roughly 40% wider than
1.96 and is the honest width at this sample size. Reporting the narrower one would overstate
precision -- exactly the failure mode this phase exists to remove.

WHY PAIRED TESTS
----------------
Split difficulty varies far more than the gap between two models: a hard partition drags
every model down together. An unpaired test would attribute that shared variance to noise
and lose almost all its power. Comparing models *within* each split and testing the
differences removes the split effect entirely.

The Wilcoxon signed-rank test is used because five points say nothing useful about
normality, and it is what the plan specifies. With n = 5 its smallest achievable p-value is
0.0625 -- so a perfect 5-0 sweep still cannot reach p < 0.05. That is a real limit of five
seeds, not a quirk of the code, and it is reported rather than hidden: the paired t-test
and the effect size are printed alongside so a conclusion never rests on one statistic.
Reaching p < 0.05 by this route needs more seeds, or aggregation across datasets.
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.eval.metrics import is_classification
from src.eval.view_stats import across_datasets, holm

warnings.filterwarnings("ignore")

RESULTS = "results"
RUNS_DIR = os.path.join(RESULTS, "runs")
MET_DIR = os.path.join(RESULTS, "metrics")

# Which metric decides each task type, and whether larger is better.
PRIMARY = {"classification": ("test_auc", True), "regression": ("test_rmse", False)}
MODELS = ["rf", "gnn", "trf", "hybrid", "ens"]

T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
          8: 2.365, 9: 2.306, 10: 2.262}


def is_cls(ds):
    return is_classification(ds)


def seeded_variants():
    """Only the seeded scaffold splits enter the statistics."""
    if not os.path.isdir(RUNS_DIR):
        return []
    return sorted(v for v in os.listdir(RUNS_DIR) if v.startswith("seed"))


def load_runs(variants):
    """One long dataframe of every model's metrics on every seeded split."""
    frames = []
    for v in variants:
        path = os.path.join(RUNS_DIR, v, "metrics", "final_report_thresholded.csv")
        if not os.path.exists(path):
            print(f"  [skip] {v}: no final_report_thresholded.csv")
            continue
        df = pd.read_csv(path)
        df["variant"] = v
        frames.append(df)
    if not frames:
        raise SystemExit("No seeded runs found. Run: python -m scripts.run_split")
    return pd.concat(frames, ignore_index=True)


def ci95(values):
    """Mean and 95% t-interval half-width. Half-width is NaN for a single observation."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return mean, 0.0, np.nan, n
    t = T_CRIT.get(n, 1.96)
    return mean, sd, t * sd / np.sqrt(n), n


def summarise(df):
    rows = []
    for ds in sorted(df.dataset.unique()):
        metric, _ = PRIMARY["classification" if is_cls(ds) else "regression"]
        extra = ["test_ap", "test_f1", "test_bal_acc"] if is_cls(ds) else ["test_mae", "test_r2"]
        for model in MODELS:
            sub = df[(df.dataset == ds) & (df.model == model)]
            if sub.empty:
                continue
            row = {"dataset": ds, "model": model, "n_splits": len(sub)}
            for col in [metric] + extra:
                if col not in sub.columns:
                    continue
                mean, sd, half, n = ci95(sub[col].values)
                key = col.replace("test_", "")
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = sd
                row[f"{key}_ci95"] = half
            rows.append(row)
    return pd.DataFrame(rows)


def paired_compare(df, ds, model_a, model_b):
    """
    Compare two models across the splits they both ran on.

    Returns None when fewer than three paired splits are available -- with one or two
    points a test statistic is meaningless and printing one would invite over-reading it.
    """
    metric, higher_better = PRIMARY["classification" if is_cls(ds) else "regression"]
    a = df[(df.dataset == ds) & (df.model == model_a)].set_index("variant")[metric]
    b = df[(df.dataset == ds) & (df.model == model_b)].set_index("variant")[metric]
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return None

    va, vb = a.loc[common].values, b.loc[common].values
    diff = (va - vb) if higher_better else (vb - va)  # positive = model_a is better
    wins = int((diff > 0).sum())

    if np.allclose(diff, 0):
        p_w = p_t = 1.0
    else:
        p_w = float(sps.wilcoxon(diff).pvalue)
        p_t = float(sps.ttest_rel(va, vb).pvalue)

    sd = diff.std(ddof=1)
    return {
        "dataset": ds, "metric": metric, "model_a": model_a, "model_b": model_b,
        "n_splits": len(common),
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "mean_diff": float(diff.mean()),
        "a_wins": wins, "b_wins": len(common) - wins,
        "p_wilcoxon": p_w, "p_ttest": p_t,
        # Cohen's dz for paired samples: mean difference in units of its own SD.
        "cohens_dz": float(diff.mean() / sd) if sd > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), default=None)
    args = ap.parse_args()

    variants = seeded_variants()
    print(f"Seeded splits found: {variants or 'none'}")
    df = load_runs(variants)

    summary = summarise(df)
    os.makedirs(MET_DIR, exist_ok=True)
    summary.to_csv(os.path.join(MET_DIR, "multiseed_summary.csv"), index=False)

    print(f"\n{'=' * 78}\nMEAN +/- 95% CI ACROSS {len(variants)} SEEDED SCAFFOLD SPLITS\n{'=' * 78}")
    for ds in sorted(summary.dataset.unique()):
        metric, higher = PRIMARY["classification" if is_cls(ds) else "regression"]
        key = metric.replace("test_", "")
        unit = "AUC" if is_cls(ds) else "RMSE"
        sub = summary[summary.dataset == ds].sort_values(
            f"{key}_mean", ascending=not higher
        )
        print(f"\n  {ds}  ({unit}, {'higher' if higher else 'lower'} is better)")
        for _, r in sub.iterrows():
            print(f"      {r.model:<8}{r[f'{key}_mean']:.4f}  +/- {r[f'{key}_ci95']:.4f}"
                  f"   (sd {r[f'{key}_std']:.4f}, n={int(r.n_splits)})")

    pairs = [tuple(args.compare)] if args.compare else [
        ("ens", "hybrid"), ("hybrid", "rf"), ("hybrid", "gnn"), ("ens", "rf"),
    ]
    tests = []
    for ds in sorted(df.dataset.unique()):
        for a, b in pairs:
            r = paired_compare(df, ds, a, b)
            if r:
                tests.append(r)

    if tests:
        t = pd.DataFrame(tests)
        t.to_csv(os.path.join(MET_DIR, "multiseed_tests.csv"), index=False)
        print(f"\n{'=' * 78}\nPAIRED COMPARISONS (positive diff = first model better)\n{'=' * 78}")
        print(f"\n  {'dataset':<15}{'comparison':<16}{'mean diff':>11}{'wins':>7}"
              f"{'p(wilcox)':>11}{'p(t)':>9}{'dz':>8}")
        for _, r in t.iterrows():
            print(f"  {r.dataset:<15}{r.model_a + ' vs ' + r.model_b:<16}"
                  f"{r.mean_diff:>+11.4f}{str(r.a_wins) + '-' + str(r.b_wins):>7}"
                  f"{r.p_wilcoxon:>11.4f}{r.p_ttest:>9.4f}{r.cohens_dz:>8.2f}")

        n = int(t.n_splits.max())
        print(f"\n  With n={n} paired splits the Wilcoxon test cannot go below "
              f"p={2 ** -(n - 1) if n <= 6 else 0.03:.4f};")
        print("  a clean sweep is therefore not the same as statistical significance at 0.05.")
        print("  The paired t-test and Cohen's dz are shown so no conclusion rests on one number.")

        # Family-wise correction and the across-dataset test, using the same
        # implementation the view/fusion comparisons use -- one standard, not two.
        for (a, b), grp in t.groupby(["model_a", "model_b"], sort=False):
            grp = grp.copy()
            adj = holm(grp.p_ttest.values)
            ad = across_datasets(grp.mean_diff.values, grp.cohens_dz.values)
            survives = [ds for ds, p in zip(grp.dataset, adj) if p < 0.05]
            raw = [ds for ds, p in zip(grp.dataset, grp.p_ttest) if p < 0.05]
            print(f"\n  {a} vs {b}, over {ad['n_datasets']} dataset(s)")
            print(f"      per-dataset p<0.05: {', '.join(raw) or 'none'}"
                  f"   after Holm: {', '.join(survives) or 'none'}")
            print(f"      across datasets: favoured on {ad['wins']}/{ad['n_datasets']}, "
                  f"sign p={ad['p_sign']:.4f}, Wilcoxon(dz) p={ad['p_wilcoxon_dz']:.4f}, "
                  f"Wilcoxon(raw) p={ad['p_wilcoxon_raw']:.4f}")

    print(f"\nWrote {MET_DIR}/multiseed_summary.csv and multiseed_tests.csv")


if __name__ == "__main__":
    main()

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

The summary then reports two things that the per-dataset rows cannot say on their own:

**Holm-corrected per-dataset verdicts.** Eight datasets are eight hypotheses; at alpha=0.05
each, the family produces a false positive 34% of the time. Several conclusions in this
project rest on exactly one decisive dataset out of eight, which is the size of effect that
noise generates. The corrected column says which of those survive. Both columns are printed
because the uncorrected one is what earlier sessions reported and the change has to stay
visible.

**The across-dataset test.** "Does A beat B in general?" is one hypothesis over eight paired
observations, not eight hypotheses -- and it is the criterion `02_ENHANCEMENT_PLAN.md` §7
actually names, the reason eight datasets were staged rather than five. It is reported three
ways (sign test, Wilcoxon on standardised differences, Wilcoxon on raw differences) because
the raw version silently assumes an AUC point and a logS point are comparable quantities.
Where the three disagree, that disagreement is the result.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.eval.intervals import ci95
from src.eval.metrics import is_classification

RESULTS = "results"
RUNS_DIR = os.path.join(RESULTS, "runs")


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


def holm(pvals):
    """
    Holm-Bonferroni adjusted p-values, in the input order.

    Eight datasets are eight hypotheses. At alpha=0.05 apiece, the chance that at least one
    fires on pure noise is 1-0.95^8 = 34%, which is exactly the size of the effects this
    project reports: several conclusions rest on a single decisive dataset out of eight
    (`fuse_gated_nograph` on Lipophilicity at p=0.025, `proposed` over `gated` on BACE at
    p=0.017). Without a correction there is no way to tell those from the one false
    positive the family is expected to produce.

    Holm rather than plain Bonferroni because it is uniformly more powerful and needs the
    same (no) assumption about dependence between the tests -- the datasets are different
    molecules and different tasks, but the same five splits and the same two models, so
    independence is not safe to assume and Benjamini-Hochberg is not the right tool.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def across_datasets(diffs, dzs):
    """
    The across-dataset test: one p-value for "does A beat B in general?"

    THIS IS THE TEST THE PLAN'S SUCCESS CRITERION NAMES, and it is not the same question as
    the per-dataset table above. That table asks, eight times, "is the difference on *this*
    dataset larger than its own noise?" This asks once, over eight paired observations,
    whether the differences are centred on zero -- which is what "improves over the best
    single view, Wilcoxon p < 0.05" in `02_ENHANCEMENT_PLAN.md` §7 means, and the reason
    eight datasets were staged rather than five (the signed-rank floor falls from p=0.0625
    to p=0.0078).

    Three statistics, because one of them is not trustworthy on its own:

    `wilcoxon_raw` ranks the mean differences directly. Those differences are in mixed
    units -- AUC on five datasets, logS/logD/kcal-mol on three -- so a 0.02 AUC change and
    a 0.02 logS change get ranked against each other as though they were comparable. They
    are not, and Demsar (2006) is explicit that the signed-rank test over datasets assumes
    commensurable differences.

    `wilcoxon_dz` fixes that by ranking each dataset's Cohen's dz instead: the mean
    difference divided by its own across-split standard deviation, which is unitless.

    `sign` ranks nothing at all -- it counts how many datasets favour A. It is the weakest
    and the only one that needs no commensurability assumption whatsoever. With n=8 its
    floor is p=0.0078 (8-0) and 7-1 gives p=0.070, so it cannot reach 0.05 on a 7-1 split.

    Report all three. Where they disagree, the disagreement is the finding.
    """
    d = np.asarray(diffs, dtype=float)
    z = np.asarray(dzs, dtype=float)
    z = z[np.isfinite(z)]
    n = d.size
    wins = int((d > 0).sum())
    out = {
        "n_datasets": n,
        "wins": wins,
        "median_diff": float(np.median(d)),
        "p_sign": float(sps.binomtest(wins, n, 0.5).pvalue),
        "p_wilcoxon_raw": (float(sps.wilcoxon(d).pvalue)
                           if n >= 3 and not np.allclose(d, 0) else np.nan),
        "p_wilcoxon_dz": (float(sps.wilcoxon(z).pvalue)
                          if z.size >= 3 and not np.allclose(z, 0) else np.nan),
    }
    return out


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
        ma, sa, ha, _ = ci95(va)
        mb, sb, hb, _ = ci95(vb)

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
        md, sd_diff, hd, _ = ci95(diff)
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
        w = max(10, len(args.a), len(args.b)) + 2
        print(f"      {args.a:<{w}}{ma:.4f}  +/- {ha:.4f}")
        print(f"      {args.b:<{w}}{mb:.4f}  +/- {hb:.4f}")
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

    if not rows:
        return

    df = pd.DataFrame(rows)

    # ---- Family-wise correction over the per-dataset tests -----------------------------
    df["p_holm"] = holm(df.p_ttest.values)
    df["verdict_holm"] = np.where(
        df.p_holm >= 0.05, "inside noise",
        np.where(df.mean_diff > 0, "improves", "degrades"))

    # ---- The across-dataset test ------------------------------------------------------
    ad = across_datasets(df.mean_diff.values, df.cohens_dz.values)
    for k, v in ad.items():
        df[f"across_{k}"] = v

    out = os.path.join(RESULTS, "metrics", f"view_compare_{args.a}_vs_{args.b}.csv")
    df.to_csv(out, index=False)

    def tally(col):
        return (int((df[col] == "improves").sum()),
                int((df[col] == "degrades").sum()),
                int((df[col] == "inside noise").sum()))

    n_better, n_worse, n_noise = tally("verdict")
    h_better, h_worse, h_noise = tally("verdict_holm")

    print(f"{'=' * 84}")
    print(f"PER-DATASET  ({len(df)} independent tests)")
    print(f"  uncorrected      {args.a} improves {n_better}, degrades {n_worse}, "
          f"indistinguishable {n_noise}")
    print(f"  Holm-corrected   {args.a} improves {h_better}, degrades {h_worse}, "
          f"indistinguishable {h_noise}")
    n_small = int(df.below_mde.sum())
    if n_small:
        print(f"  {n_small} uncorrected verdict(s) are statistically clear but below the "
              f"practical threshold (+/-0.02 AUC, +/-0.10 RMSE).")
    lost = df[(df.verdict != "inside noise") & (df.verdict_holm == "inside noise")]
    if len(lost):
        print(f"  Does not survive the correction: {', '.join(lost.dataset)} "
              f"(raw p {', '.join(f'{p:.3f}' for p in lost.p_ttest)} -> "
              f"{', '.join(f'{p:.3f}' for p in lost.p_holm)})")
    print(f"  With n={len(variants)} splits the per-dataset Wilcoxon floor is p=0.0625; "
          f"a 5-0 sweep is not p<0.05.")
    print("  Verdicts come from the paired difference's own 95% interval (= paired t).")

    print()
    print(f"ACROSS-DATASET  (n={ad['n_datasets']} datasets, the criterion in "
          f"02_ENHANCEMENT_PLAN.md section 7)")
    print(f"  {args.a} favoured on {ad['wins']}/{ad['n_datasets']} datasets by mean; "
          f"median difference {ad['median_diff']:+.4f}")
    print(f"  sign test        p={ad['p_sign']:.4f}   (unit-free, weakest, floor p=0.0078)")
    print(f"  Wilcoxon on dz   p={ad['p_wilcoxon_dz']:.4f}   (unit-free, standardised)")
    print(f"  Wilcoxon on raw  p={ad['p_wilcoxon_raw']:.4f}   (mixes AUC with logS/logD/"
          f"kcal-mol -- see Demsar 2006)")
    ps = [ad["p_sign"], ad["p_wilcoxon_dz"], ad["p_wilcoxon_raw"]]
    ps = [p for p in ps if np.isfinite(p)]
    if all(p < 0.05 for p in ps):
        print(f"  -> {args.a} beats {args.b} across datasets on all three statistics.")
    elif any(p < 0.05 for p in ps):
        print("  -> mixed: significant on some statistics and not others. The claim is "
              "unit-dependent and must be reported as such.")
    else:
        print("  -> no across-dataset difference. Any per-dataset win above is a local "
              "result, not a general one.")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

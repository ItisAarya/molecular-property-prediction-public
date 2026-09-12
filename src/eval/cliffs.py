"""
src/eval/cliffs.py

Activity cliffs: pairs of near-identical molecules whose measured activity is not.

    python -m src.eval.cliffs --tags desc fuse_gated
    python -m src.eval.cliffs --tags rf gnn ens --sim 0.85

WHAT AN ACTIVITY CLIFF IS AND WHY IT BREAKS THINGS
--------------------------------------------------
Two molecules that differ by one substituent, with a ten-fold difference in potency, or one
active and one not. Chemically this is the interesting case -- it is where the structure-
activity relationship is steep and where a medicinal chemist actually needs help.

Statistically it is the case every similarity-based model gets wrong by construction. A
fingerprint model predicts similar things for similar fingerprints; a graph network smooths
over local structure; a scaffold split does not separate cliff partners, because they share
a scaffold. So a model can score well overall while being systematically wrong exactly where
the chemistry is hard, and the aggregate metric will not show it.

`02_ENHANCEMENT_PLAN.md` §7 lists this under Analyses, tied to SCAGE's activity-cliff claim.

HOW A CLIFF IS DEFINED HERE
---------------------------
A test molecule is **on a cliff** if some other molecule in the same dataset (train or test)
has ECFP Tanimoto similarity >= `--sim` to it and a sufficiently different label:

* classification -- the neighbour carries the opposite label on the same task;
* regression -- the neighbour's target differs by at least `--gap` standard deviations.

This is the matched-molecular-pair idea approximated with fingerprints rather than an MMP
fragmentation, which is the standard cheap substitute and is what the fingerprints already
in the pool support. It over-counts slightly: two molecules with identical ECFP bits are not
necessarily a matched pair. That biases the cliff set toward *harder* cases, not easier ones,
so a null result here would be the conservative direction.

The comparison is cliff molecules against non-cliff molecules **within the same test set and
the same model**, so dataset difficulty and model quality both cancel.

WHAT TO EXPECT, AND WHAT WOULD BE A FINDING
-------------------------------------------
Every model should be worse on cliffs. That is not the interesting question. The interesting
question is *which* model degrades least -- if fusion buys anything chemically real rather
than statistically marginal, cliffs are where it should show, because that is where one view
(a fingerprint) is guaranteed to mislead and another (a learned graph or a pretrained
sequence model) might not.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import cls_metrics, is_classification, reg_metrics

POOL_DIR = os.path.join("data", "pool")
SPLIT_DIR = os.path.join("data", "splits")
RUNS_DIR = os.path.join("results", "runs")
MET_DIR = os.path.join("results", "metrics")
BLOCK = 512


def cliff_mask(ds, variant, sim_threshold=0.9, gap=1.0):
    """
    Which test molecules sit on an activity cliff.

    Returns a boolean array over the test split. Neighbours are searched over the whole
    dataset (train and test), because a cliff is a property of the chemistry, not of the
    partition -- a test molecule whose near-twin is in the training set with the opposite
    label is precisely the case worth isolating.
    """
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")))
    test = np.asarray(idx["test"], dtype=int)
    X, y = pool["X"].astype(np.float32), pool["y"]
    cls = is_classification(ds)

    q, r = X[test], X
    r_sum = r.sum(axis=1)
    on_cliff = np.zeros(len(test), dtype=bool)

    for start in range(0, len(test), BLOCK):
        block = q[start:start + BLOCK]
        rows = test[start:start + BLOCK]
        inter = block @ r.T
        union = block.sum(axis=1)[:, None] + r_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(union > 0, inter / union, 0.0)
        # A molecule is its own nearest neighbour at similarity 1.0; exclude the diagonal.
        s[np.arange(len(rows)), rows] = 0.0
        close = s >= sim_threshold

        for i, row in enumerate(rows):
            nbrs = np.flatnonzero(close[i])
            if nbrs.size == 0:
                continue
            for t in range(y.shape[1]):
                mine = y[row, t]
                if not np.isfinite(mine):
                    continue
                theirs = y[nbrs, t]
                theirs = theirs[np.isfinite(theirs)]
                if theirs.size == 0:
                    continue
                differs = (np.any(theirs != mine) if cls
                           else np.any(np.abs(theirs - mine) >= gap))
                if differs:
                    on_cliff[start + i] = True
                    break
    return on_cliff


def score(ds, variant, tag, mask):
    """Test metrics for one model restricted to the molecules `mask` selects."""
    path = os.path.join(RUNS_DIR, variant, "preds", f"{ds}_{tag}_test.npy")
    if not os.path.exists(path) or not mask.any():
        return None
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")))
    y = pool["y"][np.asarray(idx["test"], dtype=int)]
    p = np.load(path).reshape(len(y), -1)
    if p.shape[1] != y.shape[1]:
        return None
    m = (cls_metrics(y[mask], p[mask]) if is_classification(ds)
         else reg_metrics(y[mask], p[mask], ds))
    return m["auc"] if is_classification(ds) else m["rmse"]


def main():
    ap = argparse.ArgumentParser(description="Model accuracy on activity cliffs.")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=[f"seed{i}" for i in range(5)])
    ap.add_argument("--sim", type=float, default=0.9,
                    help="ECFP Tanimoto at or above which two molecules count as a pair")
    ap.add_argument("--gap", type=float, default=1.0,
                    help="regression: label difference in SDs that makes a pair a cliff")
    args = ap.parse_args()

    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    datasets = args.datasets or list(pool_index)

    rows = []
    for ds in datasets:
        cls = is_classification(ds)
        for v in args.variants:
            try:
                on = cliff_mask(ds, v, args.sim, args.gap)
            except FileNotFoundError:
                continue
            if on.sum() < 10 or (~on).sum() < 10:
                continue
            for tag in args.tags:
                a, b = score(ds, v, tag, on), score(ds, v, tag, ~on)
                if a is None or b is None:
                    continue
                rows.append({"dataset": ds, "variant": v, "tag": tag,
                             "metric": "auc" if cls else "rmse",
                             "n_cliff": int(on.sum()), "n_other": int((~on).sum()),
                             "cliff": a, "other": b,
                             # Positive = worse on cliffs, whichever direction is better.
                             "penalty": (b - a) if cls else (a - b)})

    if not rows:
        raise SystemExit(
            "No dataset had both enough cliff and enough non-cliff molecules with archived "
            f"predictions at --sim {args.sim}. Try a lower threshold, or check that "
            f"{RUNS_DIR}/<variant>/preds/ holds predictions for those tags.")

    df = pd.DataFrame(rows)
    os.makedirs(MET_DIR, exist_ok=True)
    out = os.path.join(MET_DIR, f"activity_cliffs_sim{args.sim:g}.csv")
    df.to_csv(out, index=False)

    print(f"Activity cliffs at ECFP Tanimoto >= {args.sim}, mean over "
          f"{len(args.variants)} seeded splits")
    print("penalty = how much worse the model is on cliff molecules (higher = worse)\n")
    agg = (df.groupby(["dataset", "tag", "metric"])
             .agg(n_cliff=("n_cliff", "mean"), n_other=("n_other", "mean"),
                  cliff=("cliff", "mean"), other=("other", "mean"),
                  penalty=("penalty", "mean"))
             .reset_index())
    for ds in agg.dataset.unique():
        sub = agg[agg.dataset == ds].sort_values("penalty")
        n_c, n_o = sub.n_cliff.iloc[0], sub.n_other.iloc[0]
        print(f"  {ds}  ({sub.metric.iloc[0]}, {n_c:.0f} cliff / {n_o:.0f} other)")
        for r in sub.itertuples():
            print(f"      {r.tag:<20}cliff {r.cliff:.4f}   other {r.other:.4f}   "
                  f"penalty {r.penalty:+.4f}")
        print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

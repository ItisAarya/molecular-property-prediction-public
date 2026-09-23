"""
src/eval/gate_analysis.py

What does the fusion gate actually rely on?

    python -m src.eval.gate_analysis --tag fuse_gated
    python -m src.eval.gate_analysis --tag fuse_gated --by-distance

The gate assigns each molecule a softmax weight over the three views, so it is the one
component of this project that produces an interpretable quantity for free. This module
reads the weights saved during training and asks three questions:

1. **Which view does each dataset lean on?** Averaged over molecules and splits.
2. **Does that agree with which view actually won as a single model in Phase 1?** If the
   gate leans on the graph view for a dataset where the descriptor view was measurably
   better alone, the gate is not learning what it appears to be learning.
3. **Does reliance shift for unfamiliar molecules?** Weights split by Tanimoto distance to
   the nearest training molecule.

A WARNING THIS MODULE IS BUILT AROUND
-------------------------------------
Attention and gate weights are the most over-read quantity in this literature. A high
weight means the gate routed more of that view's vector into the fused representation on
that molecule. It does **not** establish that the view caused the prediction, that the
information was unavailable elsewhere, or that a chemist would find the attribution
meaningful. Views are correlated -- a fingerprint, a graph and a SMILES string describe the
same molecule -- so weight can move between them with little effect on the output.

Question 2 above is the guard against over-reading. An attribution that disagrees with the
measured single-view ranking is evidence the weights are not tracking usefulness, and that
is worth reporting either way.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import is_classification
from src.eval.similarity import distance_bins, nearest_train_similarity

POOL_DIR = os.path.join("data", "pool")
RUNS_DIR = os.path.join("results", "runs")
MET_DIR = os.path.join("results", "metrics")

VIEW_NAMES = ("graph", "seq", "desc")
# The single-view tags each fused view corresponds to, for the agreement check.
VIEW_TO_TAG = {"graph": "gine", "seq": "seq_frozen", "desc": "desc"}


def load_gate(ds, variant, tag, split="test"):
    path = os.path.join(RUNS_DIR, variant, "preds", f"{ds}_{tag}_{split}_gate.npy")
    return np.load(path) if os.path.exists(path) else None


def single_view_scores(ds, variant):
    """Test score for each single view on this split, for the agreement check."""
    key = "auc" if is_classification(ds) else "rmse"
    out = {}
    from src.eval.leakage import excluded
    for view, tag in VIEW_TO_TAG.items():
        if excluded(ds, tag):
            continue
        path = os.path.join(RUNS_DIR, variant, "metrics", f"{ds}_{tag}_test.csv")
        if os.path.exists(path):
            out[view] = float(pd.read_csv(path).iloc[0][key])
    return out


def best_single_view(ds, variants):
    """Which view wins this dataset on its own, averaged over splits."""
    per = [single_view_scores(ds, v) for v in variants]
    # Views with a valid score on every split. On ClinTox and BBBP the sequence view's
    # archived runs read raw SMILES and are excluded (src/eval/leakage.py), so the agreement
    # check there is between the graph and descriptor views only.
    usable = [v for v in VIEW_TO_TAG if per and all(v in p for p in per)]
    if not usable:
        return None
    mean = {v: float(np.mean([p[v] for p in per])) for v in usable}
    return (max if is_classification(ds) else min)(mean, key=lambda v: mean[v])


def main():
    ap = argparse.ArgumentParser(description="Aggregate fusion gate attributions.")
    ap.add_argument("--tag", default="fuse_gated")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=[f"seed{i}" for i in range(5)])
    ap.add_argument("--views", nargs="+", default=list(VIEW_NAMES), choices=list(VIEW_NAMES),
                    help="which views this tag was trained with, in order. A "
                         "leave-one-view-out run has fewer gate columns, and labelling "
                         "them wrong would silently mislabel the whole attribution.")
    ap.add_argument("--by-distance", action="store_true")
    args = ap.parse_args()

    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    datasets = args.datasets or list(pool_index)
    views = tuple(v for v in VIEW_NAMES if v in args.views)

    rows, missing = [], []
    for ds in datasets:
        if args.by_distance:
            for variant in args.variants:
                w = load_gate(ds, variant, args.tag)
                if w is None:
                    continue
                sim = nearest_train_similarity(ds, variant, "test")
                for label, mask in distance_bins(sim):
                    if mask.sum() < 10:
                        continue
                    rows.append({"dataset": ds, "band": label, "n": int(mask.sum()),
                                 **{v: float(w[mask, i].mean())
                                    for i, v in enumerate(views)}})
            continue

        per = [load_gate(ds, v, args.tag) for v in args.variants]
        per = [w for w in per if w is not None]
        if not per:
            missing.append(ds)
            continue
        means = np.stack([w.mean(axis=0) for w in per])          # (splits, views)
        row = {"dataset": ds, "n_splits": len(per)}
        for i, v in enumerate(views):
            row[v] = float(means[:, i].mean())
            row[f"{v}_ci"] = float(2.776 * means[:, i].std(ddof=1) / np.sqrt(len(per))) \
                if len(per) > 1 else np.nan
        row["gate_prefers"] = views[int(np.argmax(means.mean(axis=0)))]
        best = best_single_view(ds, args.variants)
        # Only compare against views this run actually had.
        row["best_alone"] = best if best in views else f"{best} (excluded)"
        row["agrees"] = row["gate_prefers"] == best
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No gate weights found for {args.tag}. They are written during training as "
            f"<dataset>_<tag>_<split>_gate.npy and archived under {RUNS_DIR}/<variant>/preds/; "
            "runs from before that was added saved predictions only.")

    df = pd.DataFrame(rows)

    if args.by_distance:
        print(f"Gate attribution by distance to the training set ({args.tag})\n")
        print(f"{'dataset':<15}{'similarity':<12}{'n':>6}" +
              "".join(f"{v:>9}" for v in views))
        for ds in df.dataset.unique():
            sub = df[df.dataset == ds]
            for band in sorted(sub.band.unique()):
                b = sub[sub.band == band]
                print(f"{ds:<15}{band:<12}{b.n.mean():>6.0f}" +
                      "".join(f"{b[v].mean():>9.3f}" for v in views))
    else:
        print(f"Mean gate weight per view ({args.tag}), "
              f"mean +/- 95% CI over {len(args.variants)} seeded splits\n")
        print(f"{'dataset':<15}" + "".join(f"{v:>18}" for v in views) +
              f"{'gate prefers':>14}{'best alone':>12}{'':>4}")
        for _, r in df.iterrows():
            cells = "".join(f"{r[v]:>11.3f}+-{r[f'{v}_ci']:.3f}" for v in views)
            mark = "" if r.agrees else "  <- disagrees"
            print(f"{r.dataset:<15}{cells}{r.gate_prefers:>14}{str(r.best_alone):>12}{mark}")
        n_agree = int(df.agrees.sum())
        print(f"\n  Gate's preferred view matches the best single view on "
              f"{n_agree} of {len(df)} datasets.")
        print("  A high weight means the gate routed more of that view through, not that "
              "the view caused the prediction.")

    os.makedirs(MET_DIR, exist_ok=True)
    suffix = "_bydistance" if args.by_distance else ""
    out = os.path.join(MET_DIR, f"gate_attribution_{args.tag}{suffix}.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    if missing:
        print(f"No gate weights for: {', '.join(missing)}")


if __name__ == "__main__":
    main()

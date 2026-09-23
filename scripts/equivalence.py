"""
scripts/equivalence.py

Is "no significant difference" also "no difference that matters"?

    python -m scripts.equivalence

Writes `results/metrics/equivalence.csv`.

WHY
---
Failing to find a difference with five splits is not evidence that two models are equivalent:
the test may simply lack the power to see one. Equivalence has to be tested directly. This is
the two one-sided tests procedure (TOST) in its confidence-interval form: on each dataset, two
models are declared equivalent at alpha = 0.05 when the 90% t-interval of their paired
difference over the five seeded splits lies entirely inside the practical threshold of section
3.5 (+/-0.02 AUC, +/-0.10 RMSE). A dataset whose interval crosses that margin is *not shown
equivalent*, which is a different verdict from "different".
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.leakage import excluded

RUNS = os.path.join("results", "runs")
MET = os.path.join("results", "metrics")
CLS = {"tox21", "bbbp", "clintox", "bace", "sider"}
DATASETS = ["tox21", "bbbp", "clintox", "bace", "sider", "esol", "lipophilicity", "freesolv"]
SEEDS = [f"seed{i}" for i in range(5)]

PAIRS = [("fuse_proposed", "fuse_gated"), ("fuse_gated_nograph", "fuse_gated"),
         ("fuse_proposed", "desc"), ("fuse_gated", "desc"),
         ("fuse_proposed_e2e", "fuse_gated_e2e")]


def metric(variant, ds, tag):
    if excluded(ds, tag):
        return None
    path = os.path.join(RUNS, variant, "metrics", f"{ds}_{tag}_test.csv")
    if not os.path.exists(path):
        return None
    return float(pd.read_csv(path).iloc[0]["auc" if ds in CLS else "rmse"])


def main():
    rows = []
    for a, b in PAIRS:
        for ds in DATASETS:
            x = [metric(v, ds, a) for v in SEEDS]
            y = [metric(v, ds, b) for v in SEEDS]
            if any(v is None for v in x + y):
                continue
            d = np.array(x) - np.array(y)
            if ds not in CLS:
                d = -d                        # positive = first model better, as everywhere
            margin = 0.02 if ds in CLS else 0.10
            half = stats.t.ppf(0.95, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = d.mean() - half, d.mean() + half
            rows.append({"a": a, "b": b, "dataset": ds, "mean_diff": d.mean(),
                         "ci90_low": lo, "ci90_high": hi, "margin": margin,
                         "equivalent": bool(-margin < lo and hi < margin)})
    df = pd.DataFrame(rows)
    os.makedirs(MET, exist_ok=True)
    df.to_csv(os.path.join(MET, "equivalence.csv"), index=False)
    for (a, b), g in df.groupby(["a", "b"], sort=False):
        shown = ", ".join(g.dataset[g.equivalent]) or "none"
        print(f"{a} vs {b}: equivalent within the threshold on {int(g.equivalent.sum())} of "
              f"{len(g)} datasets ({shown})")
    print(f"\nWrote {MET}/equivalence.csv")


if __name__ == "__main__":
    main()

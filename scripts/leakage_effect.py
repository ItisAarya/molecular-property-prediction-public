"""
scripts/leakage_effect.py

How much of each sequence-reading model's ClinTox and BBBP score came from SMILES notation?

    python -m scripts.leakage_effect

Writes `results/metrics/leakage_effect.csv`: for every model re-run on canonical SMILES, its
test AUC on raw strings (`<tag>_rawsmiles`) and on canonical strings (`<tag>`), on the canonical
DeepChem split and as the mean over the five seeded splits, with a paired t-test over those five.

Everything else about the two runs is identical -- code, seed, splits, device, and the graph and
descriptor views -- so the difference is the notation (`src/data/smiles.py`). Two models that
never read SMILES characters are listed for reference; their scores do not depend on notation.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

RUNS = os.path.join("results", "runs")
MET = os.path.join("results", "metrics")
SEEDS = [f"seed{i}" for i in range(5)]
TAGS = ["fuse_seqonly", "fuse_gated_nograph", "fuse_concat", "fuse_gated", "fuse_xattn",
        "fuse_bilinear", "fuse_proposed"]
REFERENCE = ["gin_ref", "desc"]


def auc(variant, ds, tag):
    path = os.path.join(RUNS, variant, "metrics", f"{ds}_{tag}_test.csv")
    return float(pd.read_csv(path).iloc[0]["auc"]) if os.path.exists(path) else np.nan


def main():
    rows = []
    for ds in ("clintox", "bbbp"):
        for tag in TAGS:
            raw = np.array([auc(v, ds, f"{tag}_rawsmiles") for v in SEEDS])
            can = np.array([auc(v, ds, tag) for v in SEEDS])
            if np.isnan(raw).any() or np.isnan(can).any():
                print(f"  {ds} {tag}: incomplete, skipped")
                continue
            rows.append({"dataset": ds, "model": tag,
                         "canonical_split_raw": auc("deepchem", ds, f"{tag}_rawsmiles"),
                         "canonical_split_canonical": auc("deepchem", ds, tag),
                         "seeded_mean_raw": raw.mean(), "seeded_mean_canonical": can.mean(),
                         "seeded_mean_change": (can - raw).mean(),
                         "p_paired_t": float(stats.ttest_rel(raw, can).pvalue)})
        for tag in REFERENCE:
            vals = np.array([auc(v, ds, tag) for v in SEEDS])
            rows.append({"dataset": ds, "model": f"{tag} (reads no SMILES characters)",
                         "canonical_split_raw": auc("deepchem", ds, tag),
                         "canonical_split_canonical": auc("deepchem", ds, tag),
                         "seeded_mean_raw": vals.mean(), "seeded_mean_canonical": vals.mean(),
                         "seeded_mean_change": 0.0, "p_paired_t": np.nan})
    df = pd.DataFrame(rows)
    os.makedirs(MET, exist_ok=True)
    df.to_csv(os.path.join(MET, "leakage_effect.csv"), index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nWrote {MET}/leakage_effect.csv")


if __name__ == "__main__":
    main()

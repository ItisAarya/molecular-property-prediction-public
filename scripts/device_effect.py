"""
scripts/device_effect.py

How much does changing the accelerator move a result -- and how does that compare with changing
the seed on the same accelerator?

    python -m scripts.device_effect

Writes `results/metrics/device_effect.csv` (one row per model x dataset x split) and
`results/metrics/device_effect_summary.csv` (the numbers section 7 of the paper quotes).

TWO EXPERIMENTS, ONE YARDSTICK
------------------------------
*Device.* Five models were trained twice with identical code, seed (42) and split indices: once
on a laptop CPU, once on a Colab T4 (`<tag>` against `<tag>_gpu`). On ClinTox and BBBP the
comparison uses the raw-SMILES runs of both (`<tag>_rawsmiles`), because what is being measured
is the device, and both devices saw the same inputs.

*Seed.* Two models were trained twice on the same CPU with seeds 42 and 43 (`<tag>` against
`<tag>_seed43`). A device change on its own changes which random numbers the dropout masks draw,
so without this control a "hardware effect" cannot be told apart from an ordinary change of seed.

Both are scaled by the practical threshold of section 3.5 (0.02 AUC, 0.10 RMSE).
"""

import os

import numpy as np
import pandas as pd

RUNS = os.path.join("results", "runs")
MET = os.path.join("results", "metrics")
CLS = {"tox21", "bbbp", "clintox", "bace", "sider"}
DATASETS = ["tox21", "bbbp", "clintox", "bace", "sider", "esol", "lipophilicity", "freesolv"]
SEEDS = [f"seed{i}" for i in range(5)]
VARIANTS = ["deepchem"] + SEEDS
LEAKY = {"clintox", "bbbp"}

DEVICE_PAIRS = [("gin_ref", "gin_ref_gpu"), ("fuse_concat", "fuse_concat_gpu"),
                ("fuse_xattn", "fuse_xattn_gpu"), ("fuse_bilinear", "fuse_bilinear_gpu"),
                ("fuse_proposed", "fuse_proposed_gpu")]
SEED_PAIRS = [("desc", "desc_seed43"), ("fuse_gated_nograph", "fuse_gated_nograph_seed43")]


def metric(variant, ds, tag):
    path = os.path.join(RUNS, variant, "metrics", f"{ds}_{tag}_test.csv")
    if not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0]
    return float(row["auc" if ds in CLS else "rmse"])


def raw_tag(ds, tag, rerun_on_leaky):
    """The archived name holding the raw-SMILES run of `tag` on `ds`, if that is what to compare."""
    if ds in LEAKY and rerun_on_leaky and not tag.startswith(("gin_ref", "desc")):
        return f"{tag}_rawsmiles"
    return tag


def rows_for(pairs, kind):
    rows = []
    for a, b in pairs:
        for ds in DATASETS:
            ta, tb = raw_tag(ds, a, True), raw_tag(ds, b, True)
            for v in VARIANTS:
                x, y = metric(v, ds, ta), metric(v, ds, tb)
                if x is None or y is None:
                    continue
                rows.append({"experiment": kind, "model": a, "dataset": ds, "variant": v,
                             "first": x, "second": y, "abs_diff": abs(x - y),
                             "threshold": 0.02 if ds in CLS else 0.10})
    return pd.DataFrame(rows)


def summarise(df, kind):
    df = df[df.experiment == kind]
    if df.empty:
        return None
    scaled = df.abs_diff / df.threshold
    canon = df[df.variant == "deepchem"]
    seeded = df[df.variant != "deepchem"]
    means = (seeded.groupby(["model", "dataset"])
             .agg(first=("first", "mean"), second=("second", "mean"), threshold=("threshold", "first")))
    mean_scaled = (means["first"] - means["second"]).abs() / means["threshold"]
    return {
        "experiment": kind,
        "models": df.model.nunique(),
        "metrics_compared": len(df),
        "identical": int((df.abs_diff == 0).sum()),
        "median_abs_diff": float(df.abs_diff.median()),
        "mean_abs_diff": float(df.abs_diff.mean()),
        "max_abs_diff": float(df.abs_diff.max()),
        "single_split_over_threshold": int((scaled > 1).sum()),
        "single_split_fraction_over": float((scaled > 1).mean()),
        "canonical_over_threshold": int((canon.abs_diff / canon.threshold > 1).sum()),
        "canonical_n": len(canon),
        "canonical_median_x_threshold": float((canon.abs_diff / canon.threshold).median()),
        "canonical_max_x_threshold": float((canon.abs_diff / canon.threshold).max()),
        "seeded_split_over_threshold": int((seeded.abs_diff / seeded.threshold > 1).sum()),
        "seeded_split_n": len(seeded),
        "five_split_mean_over_threshold": int((mean_scaled > 1).sum()),
        "five_split_mean_n": int(len(mean_scaled)),
        "five_split_mean_median_x_threshold": float(mean_scaled.median()),
        "five_split_mean_max_x_threshold": float(mean_scaled.max()),
    }


def main():
    df = pd.concat([rows_for(DEVICE_PAIRS, "device: CPU vs T4"),
                    rows_for(SEED_PAIRS, "seed: 42 vs 43, same CPU")], ignore_index=True)
    os.makedirs(MET, exist_ok=True)
    df.to_csv(os.path.join(MET, "device_effect.csv"), index=False)
    summary = pd.DataFrame([s for s in (summarise(df, "device: CPU vs T4"),
                                        summarise(df, "seed: 42 vs 43, same CPU")) if s])
    summary.to_csv(os.path.join(MET, "device_effect_summary.csv"), index=False)
    print(summary.T.to_string())
    print(f"\nWrote {MET}/device_effect.csv and device_effect_summary.csv")


if __name__ == "__main__":
    main()

"""
scripts/calibration_summary.py

The post-hoc calibration counts of section 6.1, with notation-leaking results handled.

    python -m src.eval.ece_multiseed --tags fuse_gated fuse_gated_nograph \\
        --datasets clintox bbbp --out results/metrics/ece_multiseed_canonical.csv
    python -m scripts.calibration_summary

Writes `results/metrics/ece_multiseed_leakfree.csv`: the archived `ece_multiseed.csv` rows,
minus every (model, dataset) pair that read raw SMILES on ClinTox or BBBP, with the CPU fusion
models' ClinTox and BBBP rows replaced by their canonical-SMILES re-runs.
"""

import os

import pandas as pd

from src.eval.leakage import RERUN_CANONICAL, excluded

MET = os.path.join("results", "metrics")


def main():
    arch = pd.read_csv(os.path.join(MET, "ece_multiseed.csv"))
    canon_path = os.path.join(MET, "ece_multiseed_canonical.csv")
    canon = pd.read_csv(canon_path) if os.path.exists(canon_path) else arch.iloc[0:0]

    leaky = arch.dataset.isin(["clintox", "bbbp"]) & (
        arch.tag.isin(RERUN_CANONICAL) | arch.apply(lambda r: excluded(r.dataset, r.tag), axis=1))
    kept = arch[~leaky]
    df = pd.concat([kept, canon], ignore_index=True)
    df.to_csv(os.path.join(MET, "ece_multiseed_leakfree.csv"), index=False)

    helps = df[df.helps]
    hurts = df[df.hurts]
    print(f"{len(df)} (model, dataset) pairs: calibration helps on {len(helps)}, hurts on "
          f"{len(hurts)}, inside the interval on {len(df) - len(helps) - len(hurts)}")
    if len(hurts):
        print(hurts[["tag", "dataset", "ece_raw", "ece_cal"]].to_string(index=False))
    print(f"median raw ECE where it helped {helps.ece_raw.median():.3f}; where it did not "
          f"{df[~df.helps].ece_raw.median():.3f}")
    print(f"removed {int(leaky.sum())} raw-SMILES rows, added {len(canon)} canonical rows")


if __name__ == "__main__":
    main()

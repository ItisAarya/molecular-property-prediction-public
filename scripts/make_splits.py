"""
scripts/make_splits.py

Generate the seeded scaffold splits used for multi-seed evaluation.

    python -m scripts.make_splits              # seeds 0-4
    python -m scripts.make_splits --seeds 0 1  # a subset

Writes data/splits/<ds>_seed<k>.json holding train / valid / test index arrays into the
pool built by scripts/build_pool.py, plus data/splits/<ds>_deepchem.json which reproduces
the original DeepChem partition for continuity with the Phase 0 Part 1 and 2 results.

WHY
---
Every number in the project so far comes from one partition. A single split gives a point
estimate with no error bar, so there is no way to say whether one model beating another by
0.01 AUC means anything. Five seeded scaffold splits let each metric be reported as
mean +/- 95% CI, and let two models be compared with a paired test over matched splits --
which is what the fusion model in Phase 2 will have to clear.

The splits are stored as indices, not as copied feature files: the features live once in
data/pool/ and never depend on the partition. Adding a sixth seed later costs training
time and nothing else.

WHAT TO CHECK IN THE OUTPUT
---------------------------
Two things decide whether a split is usable:

  * scaffold leakage must be 0.0% -- no scaffold may appear on both sides of the
    train/test boundary, or the benchmark stops measuring generalisation to new chemistry.
  * for classification, every task needs positives in all three parts. Tox21 assays run
    as low as 2.5% positive, so a split leaving a task with none makes it unscorable.
    (Regression datasets are checked on target mean/std per part instead.)
"""

import argparse
import json
import os

import numpy as np

from scripts.dataset_select import add_datasets_arg, resolve

from src.data.splits import _murcko_scaffold, random_scaffold_split

DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pool")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

os.makedirs(SPLIT_DIR, exist_ok=True)


def load_pool(ds):
    d = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"))
    y = d["y"].astype(np.float32)
    return [str(x) for x in d["smiles"]], (y.reshape(-1, 1) if y.ndim == 1 else y)


def audit(ds, smiles, y, idx, is_cls):
    """
    Scaffold leakage plus a label-health check appropriate to the task type.

    For classification the risk is a task landing with no positives in some part, which
    makes it unscorable. For regression that check is meaningless -- the labels are
    continuous, so counting `y == 1` would report every regression split as broken. There
    the useful check is that each part spans a comparable range of the target.
    """
    tr, va, te = idx["train"], idx["valid"], idx["test"]
    scaf = np.array([_murcko_scaffold(s) for s in smiles])

    # Acyclic molecules all share the empty scaffold and cannot leak, so exclude them.
    train_scaf = {s for s in scaf[tr] if s}
    leak_va = sum(1 for s in scaf[va] if s and s in train_scaf)
    leak_te = sum(1 for s in scaf[te] if s and s in train_scaf)

    out = {
        "sizes": [len(tr), len(va), len(te)],
        "scaffold_leak_valid_pct": round(100 * leak_va / max(len(va), 1), 2),
        "scaffold_leak_test_pct": round(100 * leak_te / max(len(te), 1), 2),
    }

    if is_cls:
        pos = {t: [int(np.nansum(y[part, t] == 1)) for part in (tr, va, te)]
               for t in range(y.shape[1])}
        out["positives_per_task"] = pos
        out["tasks_missing_positives"] = [t for t, c in pos.items() if min(c) == 0]
    else:
        col = y[:, 0]
        out["target_mean_per_part"] = [
            round(float(np.nanmean(col[part])), 3) for part in (tr, va, te)
        ]
        out["target_std_per_part"] = [
            round(float(np.nanstd(col[part])), 3) for part in (tr, va, te)
        ]
        out["tasks_missing_positives"] = []

    return out


def deepchem_split(ds, pool_index):
    """The original partition, recovered from where each split's rows sit in the pool."""
    off = pool_index[ds]["deepchem_split"]
    return {k: np.arange(off[k][0], off[k][1]) for k in ("train", "valid", "test")}


def write_split(ds, name, idx, meta):
    path = os.path.join(SPLIT_DIR, f"{ds}_{name}.json")
    with open(path, "w") as f:
        json.dump({
            "dataset": ds, "split": name,
            "train": [int(i) for i in idx["train"]],
            "valid": [int(i) for i in idx["valid"]],
            "test": [int(i) for i in idx["test"]],
            "audit": meta,
        }, f)
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Write the DeepChem and seeded scaffold splits as index arrays."
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    add_datasets_arg(ap)
    args = ap.parse_args()

    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    problems = []

    for ds in resolve(args.datasets, pool_index):
        smiles, y = load_pool(ds)
        is_cls = pool_index[ds]["task_type"] == "classification"
        kind = "classification" if is_cls else "regression"
        print(f"\n=== {ds} ({kind}, n={len(smiles)}, {y.shape[1]} task(s)) ===")

        variants = {"deepchem": deepchem_split(ds, pool_index)}
        for seed in args.seeds:
            tr, va, te = random_scaffold_split(ds, smiles, seed)
            variants[f"seed{seed}"] = {"train": tr, "valid": va, "test": te}

        for name, idx in variants.items():
            a = audit(ds, smiles, y, idx, is_cls)
            write_split(ds, name, idx, a)
            flag = ""
            if a["scaffold_leak_test_pct"] > 0:
                flag += f"  LEAK test={a['scaffold_leak_test_pct']}%"
                problems.append(f"{ds}/{name}: scaffold leak")
            if a["tasks_missing_positives"]:
                flag += f"  tasks with no positives: {a['tasks_missing_positives']}"
                problems.append(f"{ds}/{name}: tasks {a['tasks_missing_positives']} unscorable")
            extra = ""
            if not is_cls:
                extra = f"  target mean/part={a['target_mean_per_part']}"
            print(f"  {name:<10} sizes={a['sizes']}  "
                  f"scaffold leak valid/test = {a['scaffold_leak_valid_pct']}%/"
                  f"{a['scaffold_leak_test_pct']}%{extra}{flag}")

    print(f"\nWrote splits to {SPLIT_DIR}/")
    if problems:
        print("\nISSUES:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("No scaffold leakage; every classification task has positives in all three parts.")


if __name__ == "__main__":
    main()

"""
scripts/audit_duplicates.py

Find molecules that appear more than once in a dataset, and check whether the copies agree.

    python -m scripts.audit_duplicates
    python -m scripts.audit_duplicates --datasets bbbp clintox

Writes `results/metrics/duplicate_audit.csv` (the canonical split) and
`results/metrics/duplicate_audit_by_split.csv` (every split variant).

WHY THIS EXISTS
---------------
Raw SMILES strings hide duplicates: the same molecule has many valid spellings, so two rows
can be the same compound and compare unequal. Canonicalising first reveals them, and in three
of these eight benchmarks it reveals that the same molecule is present twice **with different
measured answers**. Aspirin is in BBBP twice, labelled both permeable and not.

That matters for a paper in two ways, and they pull in opposite directions.

Whether it is **leakage** depends on the split, and the audit is what establishes which.
On the canonical DeepChem split no duplicate group crosses train/valid/test: DeepChem groups
every acyclic molecule under the one empty scaffold, so identical molecules always share a
group. The five seeded splits (`src/data/splits.py`) treat each acyclic molecule as its own
group instead, so an acyclic duplicate *can* land on both sides. It does, rarely: 1-2 groups
per split in BBBP (chloroform, dichloromethane, divinyl ether, 2-chloro-1,1,1-trifluoroethane)
and 0-1 in ESOL (a hexitol recorded twice with different solubilities). The by-split archive
records every such case, so the size of that leak is a measurement, not an assumption.

It **is** a ceiling on achievable accuracy. A model cannot be right about both copies of a
molecule the benchmark labels two ways, so some fraction of every reported error on these
datasets is not the model's to fix.

ClinTox needs a caveat. All 19 of its conflicting groups are one drug entered twice, once from
each of the benchmark's source lists: once in aromatic notation labelled approved and
non-toxic, once in Kekule notation labelled failed for toxicity. They are an artefact of how
the benchmark was assembled, and the same artefact makes SMILES notation predict the ClinTox
labels (`scripts/audit_notation.py`).

WHAT COUNTS AS A CONFLICT
-------------------------
Two copies conflict when some task has finite labels on more than one copy and those labels
differ. Missing measurements are not conflicts -- Tox21 and SIDER are mostly missing, and
treating an unmeasured task as a disagreement would report noise as a finding. Regression
labels are compared with a tolerance, since two runs of the same experiment do not produce
bit-identical floats.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data.materialize import dataset_names
# Shared with the app's provenance lookup on purpose: if the two canonicalisations ever
# disagreed, the app would call a molecule novel that this audit counts as a duplicate.
from src.deploy.lookup import canonical

POOL_DIR = os.path.join("data", "pool")
SPLIT_DIR = os.path.join("data", "splits")
MET_DIR = os.path.join("results", "metrics")
OUT = os.path.join(MET_DIR, "duplicate_audit.csv")
TOL = 1e-6


def audit(ds, variant="deepchem"):
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    y = np.asarray(pool["y_raw"], dtype=float)

    groups = {}
    for row, smi in enumerate(pool["smiles"]):
        c = canonical(str(smi))
        if c is not None:
            groups.setdefault(c, []).append(row)

    with open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")) as f:
        splits = json.load(f)
    where = {}
    for name in ("train", "valid", "test"):
        for row in splits[name]:
            where[int(row)] = name

    dup = {c: rows for c, rows in groups.items() if len(rows) > 1}
    conflicting, cross_split, n_rows_conflicting = 0, 0, 0
    for rows in dup.values():
        ys = y[rows]
        bad = False
        for t in range(ys.shape[1]):
            col = ys[:, t]
            col = col[np.isfinite(col)]
            if col.size > 1 and float(np.ptp(col)) > TOL:
                bad = True
                break
        if bad:
            conflicting += 1
            n_rows_conflicting += len(rows)
        if len({where.get(r) for r in rows}) > 1:
            cross_split += 1

    return {
        "dataset": ds,
        "rows": int(len(pool["smiles"])),
        "unique_molecules": int(len(groups)),
        "duplicate_groups": int(len(dup)),
        "conflicting_groups": int(conflicting),
        "rows_in_conflicting_groups": int(n_rows_conflicting),
        "groups_spanning_splits": int(cross_split),
    }


VARIANTS = ["deepchem", "seed0", "seed1", "seed2", "seed3", "seed4"]
BY_SPLIT_OUT = os.path.join(MET_DIR, "duplicate_audit_by_split.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variant", default="deepchem")
    ap.add_argument("--out", default=OUT,
                    help="where to write. Defaults to the full-set archive, "
                         "which a narrowed --datasets run refuses to touch.")
    args = ap.parse_args()

    everything = dataset_names()
    datasets = args.datasets or everything
    rows = [audit(ds, args.variant) for ds in datasets]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # A narrowed run does NOT overwrite the archive. This project has been caught by that
    # twice already -- a re-run with fewer `--datasets` silently replaced a full result file
    # with a partial one, and the partial version then disagreed with the paper. `--out`
    # exists for anyone who genuinely wants a subset on disk.
    partial = set(datasets) != set(everything)
    if partial and args.out == OUT:
        print(f"\nNOT written: this run covers {len(datasets)} of {len(everything)} datasets, "
              f"and {OUT} is the full-set archive that scripts/check_paper.py asserts "
              f"against.\nRe-run without --datasets to refresh it, or pass --out to write "
              f"this subset somewhere else.")
        return

    os.makedirs(MET_DIR, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Every split variant, because the seeded splits group acyclic molecules differently from
    # the canonical one and can separate identical molecules. Written only for the full set.
    if args.out == OUT:
        by_split = pd.DataFrame([dict(audit(ds, v), variant=v)
                                 for v in VARIANTS for ds in datasets])
        by_split.to_csv(BY_SPLIT_OUT, index=False)
        crossing = by_split[by_split.groups_spanning_splits > 0]
        print("\nDuplicate groups that cross train/valid/test, by split variant:")
        print(crossing[["variant", "dataset", "groups_spanning_splits"]].to_string(index=False)
              if len(crossing) else "  none")
        print(f"Wrote {BY_SPLIT_OUT}")
    print()
    total_cross = int(df.groups_spanning_splits.sum())
    if total_cross:
        print(f"WARNING: {total_cross} duplicate group(s) span more than one split. That is "
              f"leakage: the same molecule is being trained on and tested on.")
    else:
        print(f"No duplicate group spans more than one split in any dataset on the "
              f"{args.variant} split.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

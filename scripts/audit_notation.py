"""
scripts/audit_notation.py

Does the way a SMILES string is *written* predict the label?

    python -m scripts.audit_notation

Writes `results/metrics/notation_audit.csv`, one row per (dataset, task).

WHY THIS EXISTS
---------------
A molecule with an aromatic ring can be written with lower-case aromatic atoms (`c1ccccc1`) or
in Kekule form (`C1=CC=CC=C1`). Chemistry-aware featurisers cannot tell the two apart; a
language model reads the characters and can. If a benchmark was assembled from sources that
used different conventions, and the sources differ in their labels, the notation becomes a
label proxy that only sequence models can exploit.

For every dataset this script reports, over molecules containing at least one aromatic atom
(the only ones that can be written both ways):

  kekule_fraction   share written without any lower-case aromatic atom
  notation_auc      for classification, the AUC of that single notation bit as a predictor
                    of each task (folded so that 0.5 means no information and 1.0 perfect)
  notation_spearman for regression, |Spearman rho| between the notation bit and the target

A dataset whose aromatic molecules are all written one way cannot leak through notation.
`src/data/smiles.py` canonicalises exactly the datasets this audit flags.
"""

import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.data.materialize import dataset_names
from src.data.smiles import NOTATION_CANONICALISED
from src.eval.metrics import is_classification

RDLogger.DisableLog("rdApp.*")
POOL_DIR = os.path.join("data", "pool")
OUT = os.path.join("results", "metrics", "notation_audit.csv")

# A lower-case aromatic atom, bare or bracketed ("c", "n", "[nH]", "[se]"), not part of an
# element symbol such as "Cl" or "Sc".
AROMATIC_TOKEN = re.compile(r"(?<![A-Z])[bcnops](?![a-z])|\[(?:se|as|te|[bcnops])")


def main():
    rows = []
    for ds in dataset_names():
        pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
        smiles = [str(s) for s in pool["smiles"]]
        y = np.asarray(pool["y_raw"], dtype=float).reshape(len(smiles), -1)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        aromatic = np.array([m is not None and any(a.GetIsAromatic() for a in m.GetAtoms())
                             for m in mols])
        written_aromatic = np.array([bool(AROMATIC_TOKEN.search(s)) for s in smiles], dtype=float)
        kekule_fraction = float(1 - written_aromatic[aromatic].mean()) if aromatic.any() else np.nan

        for t in range(y.shape[1]):
            ok = aromatic & np.isfinite(y[:, t])
            bit, target = written_aromatic[ok], y[ok, t]
            row = {"dataset": ds, "task": t, "aromatic_molecules": int(ok.sum()),
                   "kekule_fraction": kekule_fraction,
                   "canonicalised": ds in NOTATION_CANONICALISED,
                   "notation_auc": np.nan, "notation_spearman": np.nan}
            if 0 < bit.mean() < 1:
                if is_classification(ds):
                    if len(np.unique(target)) == 2:
                        auc = roc_auc_score(target, bit)
                        row["notation_auc"] = float(max(auc, 1 - auc))
                else:
                    row["notation_spearman"] = float(abs(stats.spearmanr(bit, target)[0]))
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)

    summary = df.groupby("dataset", sort=False).agg(
        kekule_fraction=("kekule_fraction", "first"),
        max_notation_auc=("notation_auc", "max"),
        max_notation_spearman=("notation_spearman", "max"),
        canonicalised=("canonicalised", "first"))
    print(summary.round(3).to_string())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

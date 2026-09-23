"""
src/deploy/lookup.py

Was this molecule in the data, and if so, which part of it, and what was the true answer?

    from src.deploy.lookup import find
    find("CC(=O)Oc1ccccc1C(=O)O", "bbbp")
    # {'found': True, 'split': 'train', 'y': array([1.])}

WHY A DEMO NEEDS THIS
---------------------
Every molecule a person thinks to type is famous, and famous molecules are in these
datasets. Aspirin is in the *training* set of five of the eight. So the natural demo --
type in aspirin, admire the prediction -- is not a demonstration of prediction at all. It
is a memory test, and presenting it as a prediction would be a quiet lie to whoever is
watching.

Worse, it can be a memory test the model *fails*: this project's deployed BBBP model puts
aspirin at 0.4% probability of crossing the blood-brain barrier, and the measured answer in
its own training set is yes. Hiding that would be dishonest; showing it, next to the split
the molecule came from and the model's measured accuracy, is the whole point.

So the app looks every input up and says plainly which of three situations it is in:

  train        the model learned from this molecule. Not a prediction.
  valid/test   held out. This is a real prediction, and the true answer is known, so the
               viewer can check the model rather than take its word.
  not found    genuinely novel to this project. A real prediction with no answer key.

MATCHING IS BY CANONICAL SMILES
-------------------------------
The same molecule has many valid SMILES spellings, so a string comparison would miss most
matches and report "novel" for molecules sitting in the training set. RDKit's canonical
form is the same for every spelling of one structure, which is what makes the lookup
honest. It does not resolve stereochemistry differences or salt forms, so a near-miss can
still read as novel -- the failure direction is "we did not notice it was in the data",
which is the safe one for a claim of novelty only if the caller words it carefully. The app
says "not found in this dataset" rather than "never seen before".

THE BENCHMARKS CONTAIN THE SAME MOLECULE TWICE, WITH DIFFERENT ANSWERS
---------------------------------------------------------------------
Canonicalising the pools turns up duplicates that raw SMILES strings hide, and some of them
disagree with themselves:

    dataset    rows   unique   duplicate groups   groups whose labels conflict
    bbbp       2039     1975                 60                             10
    clintox    1480     1461                 19                             19
    esol       1128     1117                 11                              6

Aspirin is one of them: BBBP holds it twice, once labelled permeable and once not. So
"the measured answer" is not always a single thing, and a lookup that silently returned the
first match would show one of two contradictory answers with no hint that the other exists.
`find` returns every match and flags the disagreement.

On the canonical DeepChem split, which is what the deployed models were trained on, no
duplicate group spans two splits. The seeded splits can separate a few acyclic duplicates
(`scripts/audit_duplicates.py` records them per split), so this lookup reports every match
rather than assuming a molecule appears in one split only.

The index is built once per dataset and cached: canonicalising a few thousand SMILES costs
a few seconds, and a Streamlit rerun would otherwise pay it on every keystroke.
"""

import json
import os

import numpy as np

POOL_DIR = os.path.join("data", "pool")
SPLIT_DIR = os.path.join("data", "splits")

_INDEX = {}


def canonical(smiles):
    """RDKit's canonical SMILES, or None if the string is not a molecule."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def index(ds, variant="deepchem"):
    """{canonical smiles -> [(row, split), ...]} for one dataset, built once per process.

    A list rather than a single entry because these pools genuinely contain the same
    molecule more than once -- see the module docstring. Keeping every match is what lets
    `find` notice when the duplicates disagree.
    """
    key = (ds, variant)
    if key in _INDEX:
        return _INDEX[key]

    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    with open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")) as f:
        splits = json.load(f)

    where = {}
    for name in ("train", "valid", "test"):
        for row in splits[name]:
            where[int(row)] = name

    out = {}
    for row, smi in enumerate(pool["smiles"]):
        c = canonical(str(smi))
        if c is not None:
            out.setdefault(c, []).append((row, where.get(row, "?")))
    _INDEX[key] = out
    return out


def find(smiles, ds, variant="deepchem"):
    """
    Where this molecule sits in one dataset, and its measured answer if it is there.

    `y` is the raw measured label -- chemical units for regression, 0/1 for
    classification, NaN for a task this molecule was never measured on (Tox21 and SIDER
    are both full of those, and treating a missing measurement as a zero is the bug this
    project's Phase 0 was largely about).

    When the molecule appears more than once, `y` is the *first* occurrence and `conflict`
    is True if any task's finite labels disagree across the copies. A caller that shows
    `y` without checking `conflict` is presenting one of several contradictory measurements
    as though it were the answer. `split` is a single name when every copy sits in the same
    split, which -- measured across all eight datasets -- is always the case here.
    """
    c = canonical(smiles)
    if c is None:
        return {"found": False, "reason": "unparsable"}

    hits = index(ds, variant).get(c)
    if not hits:
        return {"found": False, "reason": "not in this dataset"}

    rows = [r for r, _ in hits]
    splits = sorted({s for _, s in hits})
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    ys = np.asarray(pool["y_raw"][rows], dtype=float)

    conflict = False
    for t in range(ys.shape[1]):
        col = ys[:, t]
        col = col[np.isfinite(col)]
        if col.size > 1 and np.ptp(col) > 1e-6:
            conflict = True
            break

    return {"found": True,
            "split": splits[0] if len(splits) == 1 else "/".join(splits),
            "rows": rows, "n_copies": len(rows), "conflict": conflict,
            "y": ys[0], "all_y": ys}

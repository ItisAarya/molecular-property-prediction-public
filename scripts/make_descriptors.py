"""
scripts/make_descriptors.py

Compute RDKit 2-D physicochemical descriptors for every molecule in each pool.

    python -m scripts.make_descriptors
    python -m scripts.make_descriptors --datasets bace sider freesolv

Writes, per dataset:
    data/pool/<ds>_desc.npz    X (n x n_desc, float32, RAW), names, smiles

This is the third view's raw material. The inherited pipeline used only a 1024-bit ECFP
fingerprint, and only inside a Random Forest. A fingerprint records *which substructures
are present*; it says nothing about molecular weight, lipophilicity, polar surface area or
flexibility -- the bulk properties that actually drive solubility, permeability and
hydration free energy. Those are exactly what these descriptors carry, so the descriptor
view is complementary to the graph and sequence views rather than a rewording of them.


WHY THE VALUES ARE STORED RAW
-----------------------------
A descriptor is a property of a molecule, so like every other feature it is
split-independent and belongs in the pool, computed once.

Its *normalisation* is not. Z-scoring needs a mean and a standard deviation, and those are
fitted quantities. Fitting them over the whole pool would let the test molecules' weights
and logP values set the constants the model trains against -- a quiet distributional leak
of exactly the shape Phase 0 removed when it moved the meta-learner, the ensemble weights,
the calibrators and the thresholds off the validation split.

So this script stores raw values and nothing else. The scaler is fitted on the active
split's training rows only, at training time, in `src/models/encoders/descriptor.py`.


TWO NUMERICAL TRAPS, BOTH HANDLED HERE
--------------------------------------
1. `Ipc` (information content) grows exponentially with molecule size. On SIDER's largest
   peptide it reaches 3.6e198, which is not merely large -- it overflows float32 to
   infinity, and would dominate any scaler fitted alongside it. RDKit offers an averaged
   form, `Ipc(mol, avg=True)`, which is the same quantity per atom and stays bounded. That
   is what we use.

2. A handful of descriptors are undefined for particular molecules (the partial-charge
   family returns NaN when a molecule contains an atom outside its parameter set). Those
   are stored as NaN rather than zero, because zero is a real, plausible descriptor value
   and would be indistinguishable from a measured one. The training-time scaler imputes
   them with the training median -- the same reasoning as the `w` mask on Tox21's labels:
   a value that does not exist must not be written as one that does.
"""

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import argparse
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors

from scripts.dataset_select import add_datasets_arg, resolve

DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pool")

# Above this magnitude a value is treated as unusable rather than merely large. float32
# tops out near 3.4e38, and nothing chemically meaningful here approaches 1e30.
MAX_MAGNITUDE = 1e30


def descriptor_functions():
    """
    The RDKit 2-D descriptor list, with `Ipc` swapped for its averaged form.

    Returns a list of (name, callable). Order is fixed by RDKit, so the columns mean the
    same thing on every dataset and across re-runs.
    """
    out = []
    for name, fn in Descriptors.descList:
        if name == "Ipc":
            # Bounded per-atom form; see module docstring.
            out.append((name, lambda m: GraphDescriptors.Ipc(m, avg=True)))
        else:
            out.append((name, fn))
    return out


def compute_one(mol, funcs):
    """All descriptors for one molecule. Anything undefined or absurd becomes NaN."""
    vals = np.empty(len(funcs), dtype=np.float64)
    for i, (_, fn) in enumerate(funcs):
        try:
            vals[i] = fn(mol)
        except Exception:
            # A descriptor that raises on this molecule is missing, not zero.
            vals[i] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    vals[np.abs(vals) > MAX_MAGNITUDE] = np.nan
    return vals


def build_for_dataset(ds, funcs):
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    smiles = [str(s) for s in pool["smiles"]]

    X = np.empty((len(smiles), len(funcs)), dtype=np.float64)
    unparsed = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Should not happen -- verify_prep asserts every SMILES parses -- but an
            # all-NaN row is imputed rather than misaligning the pool.
            X[i] = np.nan
            unparsed += 1
        else:
            X[i] = compute_one(mol, funcs)

    names = np.array([n for n, _ in funcs])
    n_missing = int(np.isnan(X).sum())
    cols_with_missing = int((np.isnan(X).any(axis=0)).sum())
    # Constant columns carry no signal; reported here, dropped at fit time on train rows.
    with np.errstate(invalid="ignore"):
        constant = int(np.sum(np.nanstd(X, axis=0) < 1e-8))

    np.savez_compressed(
        os.path.join(POOL_DIR, f"{ds}_desc.npz"),
        X=X.astype(np.float32),
        names=names,
        smiles=pool["smiles"],
    )

    print(f"  {ds:<15} n={len(smiles):<6} descriptors={len(funcs)}  "
          f"missing={n_missing} in {cols_with_missing} column(s)  "
          f"constant={constant}" + (f"  UNPARSED={unparsed}" if unparsed else ""))
    return names, X


def main():
    ap = argparse.ArgumentParser(
        description="Compute RDKit 2-D descriptors into the pool, unnormalised."
    )
    add_datasets_arg(ap)
    args = ap.parse_args()

    import json
    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    selected = resolve(args.datasets, pool_index)

    funcs = descriptor_functions()
    print(f"Computing {len(funcs)} RDKit 2-D descriptors per molecule.")

    worst = {}
    for ds in selected:
        names, X = build_for_dataset(ds, funcs)
        with np.errstate(invalid="ignore"):
            frac = np.isnan(X).mean(axis=0)
        for i in np.where(frac > 0)[0]:
            worst[names[i]] = max(worst.get(names[i], 0.0), float(frac[i]))

    if worst:
        print("\nDescriptors undefined for some molecules (worst rate across datasets):")
        for name, f in sorted(worst.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {name:<28} {f * 100:5.2f}% of molecules")
        print("These are imputed with the training median at fit time, never with zero.")

    print(f"\nWrote data/pool/<ds>_desc.npz for {len(selected)} dataset(s).")


if __name__ == "__main__":
    main()

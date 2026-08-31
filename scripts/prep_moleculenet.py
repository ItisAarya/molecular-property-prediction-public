from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# scripts/prep_moleculenet.py
import os, json, numpy as np, pandas as pd
import deepchem as dc
from joblib import dump

DATASETS = ["tox21", "bbbp", "clintox", "esol", "lipophilicity"]
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def load_dataset(name, featurizer="ECFP", split="scaffold"):
    if name == "tox21":
        return dc.molnet.load_tox21(featurizer=featurizer, split=split)
    if name == "bbbp":
        return dc.molnet.load_bbbp(featurizer=featurizer, split=split)
    if name == "clintox":
        return dc.molnet.load_clintox(featurizer=featurizer, split=split)
    if name == "esol":
        return dc.molnet.load_delaney(featurizer=featurizer, split=split)
    if name == "lipophilicity":
        return dc.molnet.load_lipo(featurizer=featurizer, split=split)
    raise ValueError(f"Unknown dataset: {name}")

def save_split_csv(name, tasks, dset, split_tag):
    # Save SMILES + labels CSV for traceability
    smiles = np.array(dset.ids, dtype=object)
    y = dset.y
    df = pd.DataFrame({"smiles": smiles})
    if y.ndim == 1:
        df[name] = y
    else:
        for i, t in enumerate(tasks):
            df[t] = y[:, i]
    csv_path = os.path.join(OUT_DIR, f"{name}_{split_tag}.csv")
    df.to_csv(csv_path, index=False)
    return csv_path

#def save_ecfp_npz(name, dset, split_tag):
#    # Save ECFP features for ML baselines
#    npz_path = os.path.join(OUT_DIR, f"{name}_{split_tag}_ecfp.npz")
#    np.savez_compressed(npz_path, X=dset.X, y=dset.y, smiles=np.array(dset.ids, dtype=object))
#    return npz_path

def save_ecfp_npz(name, dset, split_tag):
    import numpy as np

    def to_dense_numeric(X):
        # Handle scipy sparse
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                return X.toarray().astype(np.float32, copy=False)
        except Exception:
            pass
        # Handle object arrays / lists of bitvectors
        arr = np.asarray(X, dtype=object)
        if arr.dtype == object:
            arr = np.vstack([np.asarray(row).ravel() for row in arr])
        return arr.astype(np.float32, copy=False)

    X = to_dense_numeric(dset.X)
    y = np.asarray(dset.y, dtype=np.float32, order="C")

    # Save SMILES as fixed-width unicode to avoid object dtype
    smiles = np.array(dset.ids, dtype="U200")

    npz_path = os.path.join(OUT_DIR, f"{name}_{split_tag}_ecfp.npz")
    np.savez_compressed(npz_path, X=X, y=y, smiles=smiles)
    return npz_path


def main():
    meta = {}
    for ds in DATASETS:
        print(f"\n=== Processing {ds} ===")
        tasks, (train, valid, test), transformers = load_dataset(ds, featurizer="ECFP", split="scaffold")

        # Save splits to CSV (SMILES + labels)
        tr_csv = save_split_csv(ds, tasks, train, "train")
        va_csv = save_split_csv(ds, tasks, valid, "valid")
        te_csv = save_split_csv(ds, tasks, test,  "test")

        # Save ECFP features to NPZ for ML baselines
        tr_npz = save_ecfp_npz(ds, train, "train")
        va_npz = save_ecfp_npz(ds, valid, "valid")
        te_npz = save_ecfp_npz(ds, test,  "test")

        meta[ds] = {
            "tasks": tasks,
            "sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
            "csv": {"train": tr_csv, "valid": va_csv, "test": te_csv},
            "ecfp": {"train": tr_npz, "valid": va_npz, "test": te_npz},
            "split": "scaffold"
        }
        print(f"{ds} → train/valid/test = {len(train)}/{len(valid)}/{len(test)}")
    with open(os.path.join(OUT_DIR, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved metadata → data/dataset_meta.json")

if __name__ == "__main__":
    main()

"""
scripts/prep_moleculenet.py

Downloads the MoleculeNet datasets through DeepChem, applies a scaffold split, and
writes them to disk in a form the rest of the pipeline can consume.

Outputs (one set per dataset, per split):
    data/<ds>_<split>.csv        SMILES + labels. A missing label is an empty cell (NaN).
    data/<ds>_<split>_ecfp.npz   X, y, y_raw, w, smiles, y_mean, y_std
    data/dataset_meta.json       task names, split sizes, missing-label stats, y scaling


WHY WE KEEP THE `w` MATRIX
--------------------------
DeepChem returns three arrays per dataset: features `X`, labels `y`, and a weight
matrix `w`. For sparsely measured datasets, `w` carries essential information:

    w[i, t] == 0   ->   molecule i was NEVER TESTED in assay t

For those entries DeepChem stores a placeholder `0.0` in `y`. If `w` is discarded,
every untested (molecule, assay) pair silently becomes a confirmed *negative*.

On Tox21 that is ~15% of training labels and ~24% of validation/test labels (up to
~39% for individual assays). Since only ~6% of Tox21 labels are positive, those
fabricated negatives badly distort both the training signal and every reported metric.

So: wherever `w == 0`, we write `y = NaN`. Downstream code masks NaN explicitly and
therefore never trains on, or scores against, a label that does not exist.

Note on the values inside `w`: DeepChem's BalancingTransformer also stores per-class
balancing weights in `w` (Tox21 has 25 distinct values). We deliberately keep only the
binary "is this label present?" signal, because our training code computes its own
class weighting (`pos_weight`); carrying DeepChem's weights through as well would
apply class balancing twice.


WHY WE KEEP `y_raw`, `y_mean`, `y_std`
--------------------------------------
For the regression datasets (ESOL, Lipophilicity), DeepChem applies a
NormalizationTransformer that z-scores the labels to mean 0 / std 1. Training on
normalized targets is fine and usually helps optimization — but *reporting* an RMSE
in normalized units is meaningless to a chemist and not comparable to published
numbers. Example: an ESOL RMSE of 0.513 normalized is 0.513 x 2.067 = ~1.06 logS.

We therefore store both:
    y      - normalized (use this for training)
    y_raw  - original chemical units, e.g. logS or logD (use this for reporting)
plus the constants (`y_mean`, `y_std`) needed to convert between them:
    y_raw = y * y_std + y_mean
For classification datasets no scaling is applied, so y_raw == y, y_mean = 0, y_std = 1.
"""

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import os
import json
import argparse

import numpy as np
import pandas as pd
import deepchem as dc

DATASETS = [
    "tox21", "bbbp", "clintox", "esol", "lipophilicity",   # original five
    "bace", "sider", "freesolv",                            # added for statistical power
]
CLASSIFICATION = {"tox21", "bbbp", "clintox", "bace", "sider"}
OUT_DIR = "data"

os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_dataset(name, featurizer="ECFP", splitter="scaffold"):
    """Return (tasks, (train, valid, test), transformers) for one MoleculeNet dataset."""
    loaders = {
        "tox21": dc.molnet.load_tox21,
        "bbbp": dc.molnet.load_bbbp,
        "clintox": dc.molnet.load_clintox,
        "esol": dc.molnet.load_delaney,
        "lipophilicity": dc.molnet.load_lipo,
        "bace": dc.molnet.load_bace_classification,
        "sider": dc.molnet.load_sider,
        # FreeSolv via load_sampl, NOT load_freesolv. `load_freesolv` serves a file
        # whose target is already z-scored (task literally named "y", mean 0 / std 1,
        # no transformer applied), so its labels carry no physical unit and our own
        # normalisation would z-score an already-z-scored target. `load_sampl` is the
        # standard MoleculeNet FreeSolv: task "expt", hydration free energy in kcal/mol
        # (-25.47 to 3.43). Reporting RMSE against the former would be meaningless and
        # not comparable to published numbers -- the same class of error as the
        # normalised-units bug fixed in Phase 0.
        "freesolv": dc.molnet.load_sampl,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}")
    return loaders[name](featurizer=featurizer, splitter=splitter)


def normalization_constants(transformers, n_tasks):
    """
    Pull (mean, std) out of DeepChem's NormalizationTransformer, if one was applied.

    Returns two arrays of shape (n_tasks,). If the labels were not normalized we
    return mean=0 / std=1, which makes `y_raw = y * std + mean` an identity.
    """
    for t in transformers:
        # NormalizationTransformer stores these only when it transforms y.
        if hasattr(t, "y_means") and hasattr(t, "y_stds") and getattr(t, "transform_y", False):
            mean = np.ravel(np.asarray(t.y_means, dtype=np.float64))
            std = np.ravel(np.asarray(t.y_stds, dtype=np.float64))
            # Broadcast a single shared constant out to every task, if needed.
            if mean.size == 1 and n_tasks > 1:
                mean = np.repeat(mean, n_tasks)
                std = np.repeat(std, n_tasks)
            return mean.astype(np.float32), std.astype(np.float32)

    return np.zeros(n_tasks, dtype=np.float32), np.ones(n_tasks, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Reshaping DeepChem's arrays
# --------------------------------------------------------------------------------------
def to_dense_2d(a, n_rows):
    """Coerce X (which may be sparse, or an object array of bit vectors) to a dense 2-D float32 array."""
    try:
        import scipy.sparse as sp

        if sp.issparse(a):
            return a.toarray().astype(np.float32, copy=False)
    except Exception:
        pass

    arr = np.asarray(a, dtype=object) if np.asarray(a).dtype == object else np.asarray(a)
    if arr.dtype == object:
        arr = np.vstack([np.asarray(row).ravel() for row in arr])
    return arr.astype(np.float32, copy=False).reshape(n_rows, -1)


def extract_arrays(dset, y_mean, y_std):
    """
    Turn one DeepChem split into the arrays we save.

    Returns X, y (NaN where missing), y_raw (NaN where missing), w (binary), smiles.
    """
    n = len(dset)

    X = to_dense_2d(dset.X, n)

    y = np.asarray(dset.y, dtype=np.float32).reshape(n, -1)
    w = np.asarray(dset.w, dtype=np.float32).reshape(n, -1)

    # Binary presence mask: 1.0 = label measured, 0.0 = never measured.
    # (Discards DeepChem's class-balancing magnitudes on purpose — see module docstring.)
    mask = (w != 0).astype(np.float32)

    # Undo the z-scoring to recover chemical units. Identity for classification.
    y_raw = y * y_std[None, :] + y_mean[None, :]

    # The core fix: a label that was never measured is NaN, not 0.
    missing = mask == 0.0
    y = np.where(missing, np.nan, y).astype(np.float32)
    y_raw = np.where(missing, np.nan, y_raw).astype(np.float32)

    # Let numpy size the string dtype to the longest SMILES present. A fixed width
    # (e.g. "U200") silently truncates longer SMILES -- the longest here is 339 chars --
    # and a truncated SMILES then fails to parse everywhere downstream.
    smiles = np.array([str(s) for s in dset.ids])

    return X, y, y_raw, mask, smiles


# --------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------
def save_split(ds, split_tag, tasks, X, y, y_raw, w, smiles, y_mean, y_std):
    """Write the CSV and the NPZ for one split. Returns (csv_path, npz_path)."""
    # CSV: human-readable, for traceability. Missing labels become empty cells,
    # which pandas reads back as NaN.
    df = pd.DataFrame({"smiles": smiles})
    for i, task in enumerate(tasks):
        df[task] = y[:, i]
    csv_path = os.path.join(OUT_DIR, f"{ds}_{split_tag}.csv")
    df.to_csv(csv_path, index=False)

    npz_path = os.path.join(OUT_DIR, f"{ds}_{split_tag}_ecfp.npz")
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,              # normalized (regression) / 0-1 (classification); NaN = missing
        y_raw=y_raw,      # chemical units; NaN = missing
        w=w,              # 1.0 = label present, 0.0 = missing
        smiles=smiles,
        y_mean=y_mean,
        y_std=y_std,
    )
    return csv_path, npz_path


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Download MoleculeNet datasets and write CSV + NPZ per split."
    )
    ap.add_argument(
        "--datasets", nargs="+", default=DATASETS, choices=DATASETS,
        help="Which datasets to prepare. Default: all. Naming a subset leaves the "
             "other datasets' files untouched and merges into the existing metadata.",
    )
    args = ap.parse_args()
    selected = list(args.datasets)

    # Preparing a subset must not delete the metadata of the datasets we are not
    # touching: the whole point of a subset run is that the files already on disk --
    # and the results computed from them -- stay exactly as they were.
    meta_path = os.path.join(OUT_DIR, "dataset_meta.json")
    meta = {}
    if len(selected) < len(DATASETS) and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        kept = [d for d in meta if d not in selected]
        print(f"Merging into existing metadata; leaving untouched: {', '.join(kept) or 'none'}")

    for ds in selected:
        print(f"\n=== Processing {ds} ===")
        tasks, (train, valid, test), transformers = load_dataset(ds)
        tasks = list(tasks)
        n_tasks = len(tasks)

        y_mean, y_std = normalization_constants(transformers, n_tasks)
        scaled = bool(np.any(y_std != 1.0) or np.any(y_mean != 0.0))

        split_info, csv_paths, npz_paths, missing_stats = {}, {}, {}, {}

        for tag, dset in [("train", train), ("valid", valid), ("test", test)]:
            X, y, y_raw, w, smiles = extract_arrays(dset, y_mean, y_std)
            csv_path, npz_path = save_split(
                ds, tag, tasks, X, y, y_raw, w, smiles, y_mean, y_std
            )

            split_info[tag] = int(len(dset))
            csv_paths[tag] = csv_path
            npz_paths[tag] = npz_path
            missing_stats[tag] = {
                "overall_pct": round(float((w == 0).mean() * 100), 2),
                "per_task_pct": [round(float(v * 100), 2) for v in (w == 0).mean(axis=0)],
            }

            n_missing = int((w == 0).sum())
            print(
                f"  {tag:<5} n={len(dset):<5} tasks={n_tasks:<3} "
                f"features={X.shape[1]:<5} missing labels={n_missing} "
                f"({missing_stats[tag]['overall_pct']}%)"
            )

        meta[ds] = {
            "tasks": tasks,
            "task_type": "classification" if ds in CLASSIFICATION else "regression",
            "sizes": split_info,
            "csv": csv_paths,
            "ecfp": npz_paths,
            "split": "scaffold",
            "n_features": int(X.shape[1]),
            "transformers": [type(t).__name__ for t in transformers],
            "label_scaled": scaled,
            "y_mean": [float(v) for v in y_mean],
            "y_std": [float(v) for v in y_std],
            "missing_labels": missing_stats,
        }

        if scaled:
            print(
                f"  labels are z-scored: y_raw = y * {y_std[0]:.4f} + {y_mean[0]:.4f} "
                f"(multiply RMSE/MAE by {y_std[0]:.4f} to report in chemical units)"
            )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata -> {meta_path}")


if __name__ == "__main__":
    main()

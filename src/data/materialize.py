"""
src/data/materialize.py

Write the per-split data files for one split variant, sliced out of the pool.

    python -m src.data.materialize --variant seed0

WHY THIS EXISTS
---------------
Every training and evaluation script in the pipeline reads fixed paths:
`data/<ds>_train_ecfp.npz`, `data/<ds>_valid_graphs.pt`, and so on. Multi-seed evaluation
needs the same pipeline run against five different partitions.

Two ways to get there. Parameterise every path in every script -- eight files, and a
refactor that Phase 1 is going to redo anyway when the training layer is rewritten around
`configs/`. Or keep the paths fixed and swap what sits behind them. This module does the
second: it slices the pooled features by a split's index arrays and writes them to the
paths the pipeline already expects.

That is deliberately an interim mechanism, not the final architecture. It is safe here
because the pipeline is a batch process with no concurrent readers, and it costs seconds
per variant because nothing is recomputed -- only sliced and written.

THE OBVIOUS HAZARD, AND THE GUARD
---------------------------------
`data/` becomes stateful: its contents depend on which variant was materialised last. If a
run dies midway, the next person to look at `data/` sees some arbitrary seed and has no way
to know. Worse, results already in `results/` would silently belong to a different split
than the data sitting next to them.

So every materialise writes `data/ACTIVE_SPLIT.json` recording the variant, and
`active_variant()` reads it back. The orchestrator prints it before each stage and restores
the `deepchem` variant when it finishes.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pool")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")
ACTIVE_FILE = os.path.join(DATA_DIR, "ACTIVE_SPLIT.json")
SPLITS = ["train", "valid", "test"]

# Everything a variant can write. A run that only trains the sequence view needs `tok`
# for the inputs and `ecfp` for the labels and scaling constants, and nothing else --
# which is what makes a Colab bundle 48 MB instead of several hundred.
ALL_ARTIFACTS = ("ecfp", "chemberta", "graphs", "tok", "desc", "csv")


def active_variant():
    """Which split variant currently sits in data/, or None if never materialised."""
    if not os.path.exists(ACTIVE_FILE):
        return None
    return json.load(open(ACTIVE_FILE))["variant"]


def dataset_names():
    """
    Every dataset the prep pipeline has staged, in `dataset_meta.json` order.

    Lives here because this module already owns `dataset_meta.json`. It was previously
    copied verbatim into four scripts, which is three more places for the path to go stale.
    """
    meta_path = os.path.join(DATA_DIR, "dataset_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} -- run: python -m scripts.prep_moleculenet")
    with open(meta_path, encoding="utf-8") as f:
        return list(json.load(f).keys())


def load_split(ds, variant):
    path = os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} -- run: python -m scripts.make_splits")
    d = json.load(open(path))
    return {s: np.asarray(d[s], dtype=int) for s in SPLITS}


def materialize_dataset(ds, variant, tasks, artifacts=None):
    """
    Slice the pooled artifacts by this variant's indices and write the split files.

    `artifacts` selects which views to write (default: all of them). Restricting it lets
    an environment that only holds some of the pools -- a Colab bundle carrying just the
    tokens and fingerprints, say -- materialise a split without the missing ones.
    """
    want = set(ALL_ARTIFACTS if artifacts is None else artifacts)
    idx = load_split(ds, variant)

    # `ecfp` is always read: it carries the labels and the y scaling constants that the
    # CSV and every metric depend on, whichever view is being trained.
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"))
    emb = np.load(os.path.join(POOL_DIR, f"{ds}_chemberta.npz")) if "chemberta" in want else None
    desc_path = os.path.join(POOL_DIR, f"{ds}_desc.npz")
    desc = (np.load(desc_path, allow_pickle=True)
            if "desc" in want and os.path.exists(desc_path) else None)
    graphs = (torch.load(os.path.join(POOL_DIR, f"{ds}_graphs.pt"),
                         weights_only=False)["graphs"] if "graphs" in want else None)
    tok = (torch.load(os.path.join(POOL_DIR, f"{ds}_tok.pt"), weights_only=False)
           if "tok" in want else None)

    sizes = {}
    for sp in SPLITS:
        i = idx[sp]
        sizes[sp] = int(len(i))

        if "ecfp" in want:
            np.savez_compressed(
                os.path.join(DATA_DIR, f"{ds}_{sp}_ecfp.npz"),
                X=pool["X"][i], y=pool["y"][i], y_raw=pool["y_raw"][i], w=pool["w"][i],
                smiles=pool["smiles"][i], y_mean=pool["y_mean"], y_std=pool["y_std"],
            )
        if emb is not None:
            np.savez_compressed(
                os.path.join(DATA_DIR, f"{ds}_{sp}_chemberta.npz"),
                cls=emb["cls"][i], mean=emb["mean"][i], model=emb["model"],
            )
        if desc is not None:
            # Raw values, as stored in the pool. The scaler is fitted on the training
            # rows at training time -- see src/models/encoders/descriptor.py.
            np.savez_compressed(
                os.path.join(DATA_DIR, f"{ds}_{sp}_desc.npz"),
                X=desc["X"][i], names=desc["names"], smiles=desc["smiles"][i],
            )
        if graphs is not None:
            torch.save(
                {"graphs": [graphs[j] for j in i], "tasks": tasks, "split": sp, "dataset": ds},
                os.path.join(DATA_DIR, f"{ds}_{sp}_graphs.pt"),
            )
        if tok is not None:
            torch.save(
                {"input_ids": tok["input_ids"][i], "attention_mask": tok["attention_mask"][i],
                 "y": tok["y"][i], "smiles": [tok["smiles"][j] for j in i], "tasks": tasks},
                os.path.join(DATA_DIR, f"{ds}_{sp}_tok.pt"),
            )

        if "csv" in want:
            df = pd.DataFrame({"smiles": pool["smiles"][i]})
            for c, task in enumerate(tasks):
                df[task] = pool["y"][i][:, c]
            df.to_csv(os.path.join(DATA_DIR, f"{ds}_{sp}.csv"), index=False)

    return sizes


def materialize(variant, verbose=True, artifacts=None):
    """Materialise every dataset for one variant and update dataset_meta.json."""
    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    meta_path = os.path.join(DATA_DIR, "dataset_meta.json")
    meta = json.load(open(meta_path))

    for ds in pool_index:
        tasks = pool_index[ds]["tasks"]
        sizes = materialize_dataset(ds, variant, tasks, artifacts)
        # Downstream code reads sizes from the metadata, so it has to track the swap.
        meta[ds]["sizes"] = sizes
        meta[ds]["split"] = variant
        if verbose:
            print(f"    {ds:<15} train/valid/test = {sizes['train']}/{sizes['valid']}/{sizes['test']}")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    with open(ACTIVE_FILE, "w") as f:
        json.dump({"variant": variant}, f, indent=2)

    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="e.g. deepchem, seed0, seed1")
    ap.add_argument("--artifacts", nargs="+", default=None, choices=list(ALL_ARTIFACTS),
                    help="which views to write (default: all)")
    args = ap.parse_args()
    print(f"Materialising split variant '{args.variant}' into {DATA_DIR}/")
    materialize(args.variant, artifacts=args.artifacts)
    print(f"\nActive split is now: {active_variant()}")


if __name__ == "__main__":
    main()

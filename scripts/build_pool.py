"""
scripts/build_pool.py

Collapse the per-split artifacts into one pool per dataset.

    python -m scripts.build_pool

Writes, for each dataset:
    data/pool/<ds>_ecfp.npz        X, y, y_raw, w, smiles
    data/pool/<ds>_chemberta.npz   cls, mean
    data/pool/<ds>_graphs.pt       graphs
    data/pool/<ds>_tok.pt          input_ids, attention_mask, y

WHY
---
Everything so far is stored per split: data/<ds>_train_ecfp.npz, _valid_, _test_. That is
fine while there is one split, but Phase 0 Part 3 needs five, so every number can be
reported as mean +/- CI rather than a single unrepeatable point estimate.

Re-running the featurisation five times would be absurd, because **none of these features
depend on the split**. An ECFP fingerprint, a molecular graph and a frozen ChemBERTa
embedding are properties of a molecule, not of the partition it happened to land in. In
particular the embedding cache costs ~21 minutes per pass; five passes would be ~1.7 hours
of recomputing identical vectors.

So the features are concatenated once into a pool, and a split becomes nothing more than
three index arrays into that pool. Adding a sixth seed later costs the training time and
nothing else.

Verified before building: all four artifact types are row-aligned per split, and no SMILES
appears in more than one split, so concatenation neither reorders nor duplicates anything.

TOKENS ARE REGENERATED, NOT CONCATENATED
----------------------------------------
The saved token tensors were padded to each split's own longest sequence (ESOL: 79 in
train, 53 in valid, 55 in test), so they cannot simply be stacked. Re-tokenising the pool
is cheap -- tokenisation is string processing, unlike the encoder forward pass -- so it is
done here to keep one consistent padding width. Phase 1's LoRA fine-tuning needs these.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import argparse

from scripts.dataset_select import add_datasets_arg, resolve

DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pool")
SPLITS = ["train", "valid", "test"]
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MAX_LEN = 128

os.makedirs(POOL_DIR, exist_ok=True)


def pool_path(ds, kind):
    return os.path.join(POOL_DIR, f"{ds}_{kind}")


def build_ecfp(ds):
    """Concatenate X / y / y_raw / w / smiles across the three splits."""
    parts = {k: [] for k in ("X", "y", "y_raw", "w", "smiles")}
    for sp in SPLITS:
        d = np.load(os.path.join(DATA_DIR, f"{ds}_{sp}_ecfp.npz"))
        for k in parts:
            parts[k].append(d[k])

    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    # y_mean / y_std are per-dataset constants, identical in every split.
    d0 = np.load(os.path.join(DATA_DIR, f"{ds}_train_ecfp.npz"))
    out["y_mean"], out["y_std"] = d0["y_mean"], d0["y_std"]

    np.savez_compressed(pool_path(ds, "ecfp.npz"), **out)
    return out["smiles"], out["X"].shape


def build_embeddings(ds):
    cls, mean = [], []
    for sp in SPLITS:
        d = np.load(os.path.join(DATA_DIR, f"{ds}_{sp}_chemberta.npz"))
        cls.append(d["cls"])
        mean.append(d["mean"])
    np.savez_compressed(
        pool_path(ds, "chemberta.npz"),
        cls=np.concatenate(cls), mean=np.concatenate(mean),
        model=np.array(MODEL_NAME),
    )
    return np.concatenate(cls).shape


def build_graphs(ds):
    graphs, tasks = [], None
    for sp in SPLITS:
        obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{sp}_graphs.pt"), weights_only=False)
        graphs.extend(obj["graphs"])
        tasks = obj["tasks"]
    torch.save({"graphs": graphs, "tasks": tasks, "dataset": ds}, pool_path(ds, "graphs.pt"))
    return len(graphs)


def build_tokens(ds, tokenizer, smiles, y):
    """Re-tokenise the pooled SMILES so padding width is consistent across the whole pool."""
    enc = tokenizer(list(smiles), padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
    torch.save({
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "y": torch.tensor(y, dtype=torch.float),
        "smiles": list(smiles),
    }, pool_path(ds, "tok.pt"))
    return tuple(enc["input_ids"].shape)


def main():
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(
        description="Concatenate the per-split artifacts into one pool per dataset."
    )
    add_datasets_arg(ap)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    selected = resolve(args.datasets, meta)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Pooling a subset must keep the entries for the datasets we are not rebuilding:
    # their pool files are still on disk and every existing split indexes into them.
    index_path = os.path.join(POOL_DIR, "pool_index.json")
    index = {}
    if len(selected) < len(meta) and os.path.exists(index_path):
        index = json.load(open(index_path))

    for ds in selected:
        smiles, x_shape = build_ecfp(ds)
        pool = np.load(pool_path(ds, "ecfp.npz"))

        emb_shape = build_embeddings(ds)
        n_graphs = build_graphs(ds)
        tok_shape = build_tokens(ds, tokenizer, smiles, pool["y"])

        n = len(smiles)
        assert emb_shape[0] == n and n_graphs == n and tok_shape[0] == n, \
            f"{ds}: pool artifacts disagree on row count"

        # Where each original split's rows ended up, so the canonical DeepChem split
        # stays reproducible as "seed: deepchem" alongside the randomised ones.
        offsets, cursor = {}, 0
        for sp in SPLITS:
            k = meta[ds]["sizes"][sp]
            offsets[sp] = [cursor, cursor + k]
            cursor += k

        index[ds] = {
            "n": n,
            "n_tasks": len(meta[ds]["tasks"]),
            "tasks": meta[ds]["tasks"],
            "task_type": meta[ds]["task_type"],
            "deepchem_split": offsets,
            "shapes": {"X": list(x_shape), "emb": list(emb_shape), "tokens": list(tok_shape)},
        }
        print(f"  {ds:<15} n={n:<6} X={x_shape}  emb={emb_shape}  tokens={tok_shape}")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"pool_index.json now covers {len(index)} dataset(s): {', '.join(index)}")
    print(f"\nWrote {os.path.join(POOL_DIR, 'pool_index.json')}")
    print("A split is now just index arrays into these pools; features are never rebuilt.")


if __name__ == "__main__":
    main()

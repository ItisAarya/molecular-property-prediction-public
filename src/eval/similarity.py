"""
src/eval/similarity.py

How far is each test molecule from anything the model was trained on?

    from src.eval.similarity import nearest_train_similarity
    sim = nearest_train_similarity("tox21", "seed0")   # one value per test molecule

WHY THIS MATTERS HERE
---------------------
Every guarantee in this project's evaluation rests on the same assumption: that the
molecules used to fit something and the molecules it is scored on are exchangeable. A
scaffold split is designed to violate that. It puts structurally novel compounds in the
test set on purpose, because that is the situation a screening model actually faces.

So "coverage is 90%" is an average over a population that is not homogeneous. It can be met
by covering the familiar molecules comfortably and the unfamiliar ones not at all -- which
is the opposite of useful, since the unfamiliar ones are why the split exists. Splitting
coverage by distance to the training set turns that suspicion into a measurement.

THE MEASURE
-----------
Tanimoto similarity on the 1024-bit ECFP fingerprints already in the pool: for two bit
vectors, the size of their intersection over the size of their union. 1.0 means an
identical fingerprint, 0.0 means no shared substructure. For each test molecule we take the
*maximum* over all training molecules -- its nearest neighbour -- because what limits a
prediction is the closest thing the model has seen, not the average thing.

Computed from the pool and the split index arrays directly, so it never depends on which
variant happens to be materialised in `data/`.
"""

import json
import os

import numpy as np

DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pool")
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

# Rows per block when comparing against the training set. Tox21 is 783 x 6258 which is
# fine in one go, but blocking keeps memory flat if a larger dataset is ever added.
BLOCK = 512


def _tanimoto_max(query, reference):
    """
    Largest Tanimoto similarity between each `query` row and any `reference` row.

    Both are binary matrices. The intersection of two bit vectors is their dot product,
    and the union is |a| + |b| - |a and b|, so the whole thing is one matrix product per
    block rather than a Python loop over pairs.
    """
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    r_sum = r.sum(axis=1)

    out = np.empty(q.shape[0], dtype=np.float32)
    for start in range(0, q.shape[0], BLOCK):
        block = q[start:start + BLOCK]
        inter = block @ r.T
        union = block.sum(axis=1)[:, None] + r_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[start:start + BLOCK] = sim.max(axis=1)
    return out


def nearest_train_similarity(ds, variant, split="test"):
    """Per-molecule nearest-neighbour Tanimoto similarity to the training set."""
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")))
    X = pool["X"]
    return _tanimoto_max(X[np.asarray(idx[split], dtype=int)],
                         X[np.asarray(idx["train"], dtype=int)])


def distance_bins(sim, edges=(0.0, 0.3, 0.4, 0.5, 0.7, 1.01)):
    """
    Assign molecules to similarity bands, returned as (label, mask) pairs.

    The default edges follow the convention that Tanimoto similarity below about 0.3-0.4
    on ECFP means the two molecules share little meaningful substructure, while above 0.7
    they are close analogues. The bands are reported rather than assumed: if a split puts
    almost everything in one band, that itself is worth seeing.
    """
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (sim >= lo) & (sim < hi)
        out.append((f"{lo:.1f}-{min(hi, 1.0):.1f}", mask))
    return out

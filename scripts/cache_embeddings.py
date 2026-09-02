"""
scripts/cache_embeddings.py

Precompute and cache ChemBERTa embeddings for every molecule.

    python -m scripts.cache_embeddings

Writes data/<ds>_<split>_chemberta.npz containing:
    cls   (N, 768)  the [CLS] token vector -- what the baseline head consumed
    mean  (N, 768)  attention-masked mean over real tokens (padding excluded)

WHY CACHE
---------
The transformer baseline freezes the encoder and trains only a small MLP head. So for any
given molecule the encoder produces the same vector every single time -- across epochs,
across cross-validation folds, across every experiment we will ever run. Recomputing it is
pure waste.

That waste is the difference between a feasible and an infeasible protocol. Correct
stacking needs out-of-fold predictions: the meta-learner must be fitted on base-model
predictions for molecules those base models did not train on. Done naively over 5 folds
that means running ChemBERTa five more times, roughly six hours on this CPU. With the
encoder output cached, each fold only has to train a two-layer MLP on precomputed vectors,
which takes seconds -- about twenty minutes in total.

The cache is also what Phase 1 needs: the fusion model consumes a sequence-view embedding
alongside the graph and descriptor views, and re-encoding on every training step would
dominate its runtime too.

WHY BOTH POOLINGS
-----------------
The baseline used the [CLS] token only. Mean pooling over the real (non-padding) tokens is
frequently stronger for sentence-level regression and is a cheap ablation to have available
later, so both are stored now while the forward passes are being paid for anyway.

INVALIDATION
------------
The cache is keyed to MODEL_NAME and to the tokenised inputs in data/<ds>_<split>_tok.pt.
Re-run this script after re-running scripts/tokenize_smiles.py, or the embeddings will
describe molecules that are no longer there. verify_prep-style row-count checks at the end
of this script catch that.
"""

import json
import os
import time

import numpy as np
import torch
from transformers import AutoModel

DATA_DIR = "data"
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
SPLITS = ["train", "valid", "test"]
BATCH = 64


def cache_path(ds, split):
    return os.path.join(DATA_DIR, f"{ds}_{split}_chemberta.npz")


@torch.no_grad()
def encode_split(model, ds, split, device="cpu"):
    """Run the frozen encoder over one split and return (cls, mean) embedding matrices."""
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_tok.pt"), weights_only=False)
    ids, attn = obj["input_ids"], obj["attention_mask"]
    n = ids.size(0)

    cls_out, mean_out = [], []
    for i in range(0, n, BATCH):
        b_ids = ids[i:i + BATCH].to(device)
        b_att = attn[i:i + BATCH].to(device)

        hidden = model(input_ids=b_ids, attention_mask=b_att).last_hidden_state

        cls_out.append(hidden[:, 0, :].cpu().numpy())

        # Mean over real tokens only. Without the mask, padding would drag every
        # short molecule's embedding toward the padding vector.
        mask = b_att.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_out.append((summed / counts).cpu().numpy())

    return (
        np.concatenate(cls_out).astype(np.float32),
        np.concatenate(mean_out).astype(np.float32),
    )


def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))

    print(f"Loading frozen encoder: {MODEL_NAME}")
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    dim = model.config.hidden_size
    total_start = time.time()
    total_rows = 0

    for ds in meta:
        for split in SPLITS:
            start = time.time()
            cls, mean = encode_split(model, ds, split)

            expected = meta[ds]["sizes"][split]
            if cls.shape[0] != expected:
                raise RuntimeError(
                    f"{ds}/{split}: encoded {cls.shape[0]} rows but metadata says "
                    f"{expected}. Re-run scripts.tokenize_smiles first."
                )

            np.savez_compressed(
                cache_path(ds, split), cls=cls, mean=mean, model=np.array(MODEL_NAME)
            )
            total_rows += cls.shape[0]
            print(
                f"  {ds:<15} {split:<6} {cls.shape[0]:>5} molecules -> "
                f"({cls.shape[0]}, {dim})  [{time.time() - start:.1f}s]"
            )

    elapsed = time.time() - total_start
    print(
        f"\nCached {total_rows} molecule embeddings in {elapsed / 60:.1f} min "
        f"({total_rows / max(elapsed, 1e-9):.0f} molecules/s)"
    )
    print("Re-run this after scripts.tokenize_smiles changes.")


if __name__ == "__main__":
    main()

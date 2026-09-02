"""
src/train/train_transformer.py

ChemBERTa baseline: a frozen pretrained SMILES encoder with a small trainable MLP head.

    python -m src.train.train_transformer

MISSING LABELS
--------------
Tox21 stores NaN for (molecule, assay) pairs that were never measured. Feeding NaN into
`nn.BCEWithLogitsLoss` produces a NaN loss, and a single NaN gradient step turns every
weight in the head into NaN -- the model is destroyed silently and predicts NaN from then
on. The loss is therefore computed per element, multiplied by a mask of "this label
exists", and averaged over the measured entries only, mirroring train_gnn.py.

WHY THE ENCODER RUNS IN EVAL MODE
---------------------------------
The encoder is frozen, but `model.train()` sets the whole module tree to training mode --
including the encoder's dropout. So the "frozen" encoder was emitting a *different* vector
for the same molecule on every epoch. That is not a deliberate augmentation, it is an
accident of how PyTorch propagates train mode, and it makes a frozen feature extractor
non-deterministic for no benefit.

The encoder is now pinned to eval mode. Two consequences:

  * A molecule has one fixed embedding, so the embeddings cached by
    scripts/cache_embeddings.py are exactly what a live forward pass would produce.
  * Training reduces to fitting a two-layer MLP on precomputed vectors, which takes
    seconds instead of the ~75 minutes a full re-encode costs. That is what makes
    five-seed evaluation affordable at all: the naive version would be six hours of
    recomputing identical vectors.

Dropout in the *head* is unaffected and still active during training.

NOTE ON BASELINE FIDELITY
-------------------------
Otherwise this stays a minimal correction of the inherited baseline: no shuffling, no
early stopping, no class weighting, 5 epochs. Those are modelling changes and belong to
Phase 1, and leaving them alone is what keeps the before/after comparison honest.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.metrics import is_classification, cls_metrics, reg_metrics
from src.utils.seed import set_seed

DATA_DIR = "data"
MODELS_DIR = "models"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"

for d in (MODELS_DIR, PRED_DIR, MET_DIR):
    os.makedirs(d, exist_ok=True)

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
POOLING = "cls"  # matches the inherited baseline; "mean" is available in the cache


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def load_labels(ds, split):
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_tok.pt"), weights_only=False)
    y = obj["y"].float()
    return y.unsqueeze(1) if y.ndim == 1 else y


def load_embeddings(ds, split, pooling=POOLING):
    """
    Frozen-encoder embeddings, from cache when available.

    The cache is not an approximation: with the encoder pinned to eval mode the cached
    vector is bit-for-bit what a live forward pass returns. If it is missing we encode on
    the spot rather than failing, so the script still works standalone.
    """
    path = os.path.join(DATA_DIR, f"{ds}_{split}_chemberta.npz")
    if os.path.exists(path):
        return torch.from_numpy(np.load(path)[pooling].astype(np.float32))

    print(f"  [cache miss] encoding {ds}/{split} live -- run scripts.cache_embeddings to avoid this")
    from transformers import AutoModel

    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_tok.pt"), weights_only=False)
    base = AutoModel.from_pretrained(MODEL_NAME).eval()
    outs = []
    with torch.no_grad():
        for i in range(0, obj["input_ids"].size(0), 64):
            h = base(input_ids=obj["input_ids"][i:i + 64],
                     attention_mask=obj["attention_mask"][i:i + 64]).last_hidden_state
            outs.append(h[:, 0, :] if pooling == "cls" else h.mean(dim=1))
    return torch.cat(outs)


# --------------------------------------------------------------------------------------
# Masked losses
# --------------------------------------------------------------------------------------
def masked_bce_with_logits(logits, y):
    """Binary cross-entropy ignoring entries whose label is NaN (never measured)."""
    mask = ~torch.isnan(y)
    loss = F.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(y, nan=0.0), reduction="none"
    )
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def masked_mse(preds, y):
    """Mean squared error ignoring entries whose target is NaN."""
    mask = ~torch.isnan(y)
    loss = F.mse_loss(preds, torch.nan_to_num(y, nan=0.0), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------------------------
# Head
# --------------------------------------------------------------------------------------
class TrfHead(nn.Module):
    """Same architecture as the inherited head, fed embeddings rather than token ids."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, d_in), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x):
        return self.head(x)


def run_ds(ds, epochs=5, batch_size=16, lr=1e-3):
    seed = set_seed()
    cls = is_classification(ds)

    X = {s: load_embeddings(ds, s) for s in ("train", "valid", "test")}
    Y = {s: load_labels(ds, s) for s in ("train", "valid", "test")}
    n_tasks = Y["train"].shape[1]

    print(f"\n=== {ds} (TRF) [seed={seed}] ===")
    print(f"  {len(X['train'])} molecules, {n_tasks} task(s), "
          f"{int(torch.isnan(Y['train']).sum())} unmeasured labels excluded from the loss")

    model = TrfHead(X["train"].shape[1], n_tasks)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for i in range(0, len(X["train"]), batch_size):
            logits = model(X["train"][i:i + batch_size])
            target = Y["train"][i:i + batch_size]
            loss = masked_bce_with_logits(logits, target) if cls else masked_mse(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {ep}/{epochs} loss={total:.4f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{ds}_trf.pt"))

    @torch.no_grad()
    def predict(split):
        model.eval()
        out = model(X[split]).numpy()
        return 1.0 / (1.0 + np.exp(-out)) if cls else out

    preds = {s: predict(s) for s in ("valid", "test")}
    for s, p in preds.items():
        np.save(os.path.join(PRED_DIR, f"{ds}_trf_{s}.npy"), p)

    score = cls_metrics if cls else (lambda a, b: reg_metrics(a, b, ds))
    for s in ("valid", "test"):
        m = score(Y[s].numpy(), preds[s])
        pd.DataFrame([m]).to_csv(os.path.join(MET_DIR, f"{ds}_trf_{s}.csv"), index=False)
        print(f"  {s.upper():<6}: {m}")


def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta:
        run_ds(ds)


if __name__ == "__main__":
    main()

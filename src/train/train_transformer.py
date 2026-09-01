"""
src/train/train_transformer.py

ChemBERTa baseline: a frozen pretrained SMILES language model with a small trainable
MLP head on the CLS token.

MISSING LABELS
--------------
Tox21 stores NaN for (molecule, assay) pairs that were never measured. Feeding NaN into
`nn.BCEWithLogitsLoss` produces a NaN loss, and a single NaN gradient step turns every
weight in the head into NaN -- the model is destroyed silently and simply predicts NaN
from then on.

So the loss is computed per element, multiplied by a mask of "this label exists", and
averaged over the measured entries only. This mirrors what train_gnn.py already does.

NOTE ON BASELINE FIDELITY
-------------------------
This file is deliberately a *minimal* correction of the original baseline: the encoder
stays frozen, batches are not shuffled, there is no early stopping and no class weighting.
Those are real weaknesses, but fixing them is a modelling change, not a bug fix, and it
belongs in Phase 1 -- keeping them here is what makes the "before vs after" comparison
honest.
"""

import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA_DIR = "data"
MODELS_DIR = "models"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"

for d in (MODELS_DIR, PRED_DIR, MET_DIR):
    os.makedirs(d, exist_ok=True)

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"


def load_tok(ds, split):
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_tok.pt"), weights_only=False)
    y = obj["y"].float()
    if y.ndim == 1:
        y = y.unsqueeze(1)
    return obj["input_ids"], obj["attention_mask"], y, obj["tasks"]


# --------------------------------------------------------------------------------------
# Masked losses
# --------------------------------------------------------------------------------------
def masked_bce_with_logits(logits, y):
    """Binary cross-entropy that ignores entries where the label is NaN (never measured)."""
    mask = ~torch.isnan(y)
    # Replace NaN with a dummy value so the elementwise op is well defined; the mask
    # removes its contribution immediately afterwards.
    y_filled = torch.nan_to_num(y, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, y_filled, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)


def masked_mse(preds, y):
    """Mean squared error that ignores entries where the target is NaN."""
    mask = ~torch.isnan(y)
    y_filled = torch.nan_to_num(y, nan=0.0)
    loss = F.mse_loss(preds, y_filled, reduction="none")
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
class TrfHead(nn.Module):
    def __init__(self, base, out_dim):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # frozen encoder (Phase 1 will add LoRA fine-tuning)
        hid = base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid, out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = out[:, 0, :]  # CLS token
        return self.head(pooled)


def batch_loader(ids, masks, ys=None, bsz=16):
    """Yield (ids, masks, ys) for training, or (ids, masks) for inference."""
    n = ids.size(0)
    for i in range(0, n, bsz):
        if ys is None:
            yield ids[i:i + bsz], masks[i:i + bsz]
        else:
            yield ids[i:i + bsz], masks[i:i + bsz], ys[i:i + bsz]


# --------------------------------------------------------------------------------------
# Train / predict
# --------------------------------------------------------------------------------------
def run_ds(ds, epochs=5, batch_size=16, lr=1e-3, device="cpu"):
    ids_tr, att_tr, ytr, tasks = load_tok(ds, "train")
    ids_va, att_va, yva, _ = load_tok(ds, "valid")
    ids_te, att_te, yte, _ = load_tok(ds, "test")

    n_tasks = ytr.shape[1]
    cls = is_classification(ds)

    n_missing = int(torch.isnan(ytr).sum())
    print(f"\n=== {ds} (TRF) ===")
    print(
        f"  {len(ids_tr)} molecules, {n_tasks} task(s), "
        f"{n_missing} unmeasured labels excluded from the loss"
    )

    base = AutoModel.from_pretrained(MODEL_NAME)
    model = TrfHead(base, out_dim=n_tasks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids_b, att_b, y_b in batch_loader(ids_tr, att_tr, ytr, batch_size):
            ids_b, att_b, y_b = ids_b.to(device), att_b.to(device), y_b.to(device)
            logits = model(ids_b, att_b)
            loss = masked_bce_with_logits(logits, y_b) if cls else masked_mse(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {ep}/{epochs} loss={total:.4f}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{ds}_trf.pt"))

    @torch.no_grad()
    def predict(ids, att):
        model.eval()
        outs = []
        for x, m in batch_loader(ids, att, None, batch_size):
            outs.append(model(x.to(device), m.to(device)).cpu().numpy())
        out = np.vstack(outs)
        return 1.0 / (1.0 + np.exp(-out)) if cls else out

    p_va = predict(ids_va, att_va)
    p_te = predict(ids_te, att_te)

    np.save(os.path.join(PRED_DIR, f"{ds}_trf_valid.npy"), p_va)
    np.save(os.path.join(PRED_DIR, f"{ds}_trf_test.npy"), p_te)

    yva_np, yte_np = yva.numpy(), yte.numpy()
    met_va = cls_metrics(yva_np, p_va) if cls else reg_metrics(yva_np, p_va)
    met_te = cls_metrics(yte_np, p_te) if cls else reg_metrics(yte_np, p_te)

    pd.DataFrame([met_va]).to_csv(os.path.join(MET_DIR, f"{ds}_trf_valid.csv"), index=False)
    pd.DataFrame([met_te]).to_csv(os.path.join(MET_DIR, f"{ds}_trf_test.csv"), index=False)

    print(f"  VALID: {met_va}")
    print(f"  TEST:  {met_te}")


def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta.keys():
        run_ds(ds)


if __name__ == "__main__":
    main()

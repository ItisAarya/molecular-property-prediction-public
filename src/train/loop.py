"""
src/train/loop.py

The one training loop every model in this project is fitted with.

    from src.train.loop import fit_and_score
    row = fit_and_score(model, loaders, y, ds=ds, tag=tag, cls=cls, unpack=unpack, ...)

WHY IT IS SHARED
----------------
Every single-view encoder and every fusion variant becomes a row in the same results
table. If those rows came from different training recipes -- different early stopping,
different class weighting, a different optimiser -- the comparison would measure the
recipes rather than the architectures, and the whole Phase 1/Phase 2 ladder would be
uninterpretable. So the loop lives in one place and the only thing that varies between
rows is the model.

WHAT IS HELD FIXED
------------------
Masked losses (unmeasured Tox21 labels are NaN and must never contribute), per-task
`pos_weight` over measured labels only, early stopping on validation -- legitimate because
Phase 0 moved every *fitted* stage off the validation split -- and restoring the
best-scoring weights rather than the last ones.

THE ORDERING HAZARD
-------------------
Predictions are saved as arrays that downstream stages read positionally against the
labels. Any loader that visits molecules out of order (length bucketing does) would
silently misalign them, producing plausible and wrong metrics. So batches carry their row
indices, results are scattered back into place, and a row that was never scored raises.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from src.eval.metrics import cls_metrics, reg_metrics
from src.models.heads import masked_bce, masked_mse, pos_weight_from_labels

PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
MODELS_DIR = "models"


def to_device(obj, device):
    """Move a batch to the device, whichever representation it is."""
    if device.type == "cpu":
        return obj
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(to_device(v, device) for v in obj)
    return obj.to(device) if hasattr(obj, "to") else obj


def checkpoint_state(model):
    """
    What is worth writing to disk: parameters training changed, plus fitted buffers.

    A frozen pretrained encoder is byte-identical in every checkpoint and reproducible
    from its name, so saving it once per fit would be gigabytes of duplication across the
    full protocol.
    """
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers()}
    return {k: v for k, v in model.state_dict().items() if k in keep}


@torch.no_grad()
def predict(model, loader, cls, device, n_rows, unpack):
    """Predictions for one split, always in dataset order."""
    model.eval()
    out, filled, cursor = None, None, 0

    for batch in loader:
        inputs, _, idx = unpack(batch)
        p = model(to_device(inputs, device)).detach().cpu().numpy()
        if out is None:
            out = np.empty((n_rows, p.shape[1]), dtype=np.float64)
            filled = np.zeros(n_rows, dtype=bool)
        if idx is None:
            rows = np.arange(cursor, cursor + p.shape[0])
            cursor += p.shape[0]
        else:
            rows = np.asarray(idx)
        out[rows] = p
        filled[rows] = True

    if not filled.all():
        raise RuntimeError(
            f"predict covered {int(filled.sum())} of {n_rows} rows -- some molecules were "
            "never scored, so predictions would not line up with labels."
        )
    return 1.0 / (1.0 + np.exp(-out)) if cls else out


def fit_and_score(model, loaders, y, ds, tag, cls, unpack, device,
                  epochs=100, patience=15, lr=1e-3, weight_decay=1e-4,
                  train_sampler=None, seed=None, extra=None, verbose=True,
                  class_weighting=True):
    """
    Train one model, restore its best epoch, and write predictions and metrics.

    Returns a summary row. `unpack(batch) -> (inputs, labels, row_indices)` is what makes
    this work for a PyG graph batch, a tuple of tensors and a dict of views alike.
    """
    for d in (PRED_DIR, MET_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)

    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    # `class_weighting=False` trains with unweighted BCE. It exists for one control only:
    # the paper's conformal section shows that turning it off moves a model from covering
    # the minority class to abandoning it (`desc_nopw`). Every reported model keeps it on.
    pos_w = (pos_weight_from_labels(y["train"]).to(device)
             if cls and class_weighting else None)

    n_par = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    if verbose:
        seed_txt = f", seed={seed}" if seed is not None else ""
        print(f"\n=== {ds} ({tag}) [{device.type}{seed_txt}] ===")
        print(f"  {int(y['train'].shape[0])} train molecules, {int(y['train'].shape[1])} "
              f"task(s), {n_par:,} params ({n_trainable:,} trainable)")

    best_score, best_state, bad, best_ep = float("-inf"), None, 0, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        total = 0.0
        for batch in loaders["train"]:
            inputs, yb, _ = unpack(batch)
            logits = model(to_device(inputs, device))
            yb = yb.to(device)
            loss = masked_bce(logits, yb, pos_w) if cls else masked_mse(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        p_val = predict(model, loaders["valid"], cls, device,
                        int(y["valid"].shape[0]), unpack)
        m = (cls_metrics(y["valid"].numpy(), p_val) if cls
             else reg_metrics(y["valid"].numpy(), p_val, ds))
        # One score, higher-is-better, so early stopping is identical for both task types.
        score = m["auc"] if cls else -m["rmse"]

        if score > best_score:
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if verbose and (ep % 10 == 0 or ep == 1):
            key = "val_auc" if cls else "val_rmse"
            print(f"  epoch {ep:>3}  loss={total:8.3f}  {key}={m['auc'] if cls else m['rmse']:.4f}")

        if bad >= patience:
            if verbose:
                print(f"  early stop at epoch {ep} (best was {best_ep})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(checkpoint_state(model), os.path.join(MODELS_DIR, f"{ds}_{tag}.pt"))

    results = {}
    for split in ("valid", "test"):
        p = predict(model, loaders[split], cls, device, int(y[split].shape[0]), unpack)
        np.save(os.path.join(PRED_DIR, f"{ds}_{tag}_{split}.npy"), p)
        m = (cls_metrics(y[split].numpy(), p) if cls
             else reg_metrics(y[split].numpy(), p, ds))
        results[split] = m
        # Provenance, written alongside the numbers rather than inferred later.
        #
        # Session 20 had to work out from wall-clock timings which archived results came
        # from the local CPU and which from a Colab T4, because nothing recorded it. That
        # mattered: the same code, seed and splits on a different device produce a
        # different model -- dropout masks are drawn on-device, so CPU and CUDA pull from
        # different generators -- and 18% of single-split numbers moved by more than the
        # minimum detectable effect. The five-split mean absorbs it, but a reader cannot
        # check that without knowing which rows came from which machine.
        row = dict(m)
        row["device"] = device.type if hasattr(device, "type") else str(device)
        row["seed"] = seed
        pd.DataFrame([row]).to_csv(
            os.path.join(MET_DIR, f"{ds}_{tag}_{split}.csv"), index=False)

    key = "auc" if cls else "rmse"
    if verbose:
        print(f"  valid {key}={results['valid'][key]:.4f}   "
              f"test {key}={results['test'][key]:.4f}   "
              f"({(time.time() - t0) / 60:.1f} min, best epoch {best_ep})")

    row = {"dataset": ds, "tag": tag, f"valid_{key}": results["valid"][key],
           f"test_{key}": results["test"][key], "params": n_par,
           "trainable": n_trainable, "best_epoch": best_ep}
    if extra:
        row.update(extra)
    return row

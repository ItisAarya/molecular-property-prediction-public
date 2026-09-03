"""
src/train/train_view.py

Train a single-view encoder standalone and write predictions in the pipeline's format.

    python -m src.train.train_view --encoder gine
    python -m src.train.train_view --encoder gin --tag gin_ref   # the inherited reference

Writes results/preds/<ds>_<tag>_{valid,test}.npy and results/metrics/<ds>_<tag>_*.csv, so a
new view drops straight into the existing stacking, calibration and reporting stages
without any of them needing to know it exists.

WHY A SHARED TRAINER
--------------------
Every single-view baseline is a row the fusion model will be compared against in Phase 2.
If those rows came from different training recipes -- different early stopping, different
class weighting -- the comparison would measure the recipes rather than the
representations. So all views train through this one loop with one objective, and the only
thing that varies is the encoder.

WHAT IS HELD FIXED FROM PHASE 0
-------------------------------
Masked losses (Tox21's unmeasured labels are NaN and must never contribute), per-task
`pos_weight` computed over measured labels only, seeding via `src.utils.seed`, and early
stopping on validation -- which is legitimate because Phase 0 moved every *fitted* stage
off the validation split, leaving it free for exactly this.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.eval.metrics import cls_metrics, is_classification, reg_metrics
from src.models.encoders.graph import build_graph_encoder
from src.models.heads import (
    SingleViewModel, masked_bce, masked_mse, pos_weight_from_labels,
)
from src.utils.seed import set_seed

DATA_DIR = "data"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
MODELS_DIR = "models"

for d in (PRED_DIR, MET_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


def load_graphs(ds, split):
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_graphs.pt"), weights_only=False)
    return obj["graphs"]


def labels_of(graphs):
    return torch.stack([g.y for g in graphs])


@torch.no_grad()
def predict(model, loader, cls):
    model.eval()
    out = torch.cat([model(b) for b in loader]).numpy()
    return 1.0 / (1.0 + np.exp(-out)) if cls else out


def run_ds(ds, encoder_name, tag, epochs, patience, batch_size, lr, weight_decay, enc_kwargs):
    seed = set_seed()
    cls = is_classification(ds)

    graphs = {s: load_graphs(ds, s) for s in ("train", "valid", "test")}
    y = {s: labels_of(graphs[s]) for s in graphs}
    n_tasks = y["train"].shape[1]
    in_dim = graphs["train"][0].x.size(1)

    loaders = {
        "train": DataLoader(graphs["train"], batch_size=batch_size, shuffle=True),
        "valid": DataLoader(graphs["valid"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(graphs["test"], batch_size=batch_size, shuffle=False),
    }

    encoder = build_graph_encoder(encoder_name, in_dim=in_dim, **enc_kwargs)
    model = SingleViewModel(encoder, n_tasks=n_tasks)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = pos_weight_from_labels(y["train"]) if cls else None

    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n=== {ds} ({tag}) [seed={seed}] ===")
    print(f"  {len(graphs['train'])} train molecules, {n_tasks} task(s), "
          f"{n_par:,} params, embedding dim {encoder.out_dim}")

    best_score, best_state, bad, best_ep = float("-inf"), None, 0, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loaders["train"]:
            logits = model(batch)
            yb = torch.stack([g.y for g in batch.to_data_list()])
            loss = masked_bce(logits, yb, pos_w) if cls else masked_mse(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        p_val = predict(model, loaders["valid"], cls)
        m = cls_metrics(y["valid"].numpy(), p_val) if cls else reg_metrics(y["valid"].numpy(), p_val, ds)
        # One score, higher-is-better, so early stopping is identical for both task types.
        score = m["auc"] if cls else -m["rmse"]

        if score > best_score:
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1:
            key = "val_auc" if cls else "val_rmse"
            val = m["auc"] if cls else m["rmse"]
            print(f"  epoch {ep:>3}  loss={total:8.3f}  {key}={val:.4f}")

        if bad >= patience:
            print(f"  early stop at epoch {ep} (best was {best_ep})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{ds}_{tag}.pt"))

    results = {}
    for split in ("valid", "test"):
        p = predict(model, loaders[split], cls)
        np.save(os.path.join(PRED_DIR, f"{ds}_{tag}_{split}.npy"), p)
        m = cls_metrics(y[split].numpy(), p) if cls else reg_metrics(y[split].numpy(), p, ds)
        pd.DataFrame([m]).to_csv(os.path.join(MET_DIR, f"{ds}_{tag}_{split}.csv"), index=False)
        results[split] = m

    key = "auc" if cls else "rmse"
    print(f"  valid {key}={results['valid'][key]:.4f}   test {key}={results['test'][key]:.4f}"
          f"   ({(time.time() - t0) / 60:.1f} min, best epoch {best_ep})")
    return {"dataset": ds, "tag": tag, f"valid_{key}": results["valid"][key],
            f"test_{key}": results["test"][key], "params": n_par, "best_epoch": best_ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="gine", choices=["gine", "gin"])
    ap.add_argument("--tag", default=None, help="output name; defaults to --encoder")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--jk", action="store_true")
    ap.add_argument("--readout", default="mean+sum", choices=["mean", "sum", "mean+sum"])
    args = ap.parse_args()

    tag = args.tag or args.encoder
    enc_kwargs = dict(hidden=args.hidden)
    if args.encoder == "gine":
        enc_kwargs.update(n_layers=args.layers, residual=not args.no_residual,
                          jk=args.jk, readout=args.readout)

    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    datasets = args.datasets or list(meta.keys())

    rows = [run_ds(ds, args.encoder, tag, args.epochs, args.patience, args.batch_size,
                   args.lr, args.weight_decay, enc_kwargs) for ds in datasets]
    pd.DataFrame(rows).to_csv(os.path.join(MET_DIR, f"{tag}_summary.csv"), index=False)
    print(f"\nWrote results/metrics/{tag}_summary.csv")


if __name__ == "__main__":
    main()

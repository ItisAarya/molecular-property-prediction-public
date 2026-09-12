"""
src/train/train_quantile.py

A conditional-quantile regressor, so that Conformalized Quantile Regression (CQR) can be
measured rather than described.

    python -m src.train.train_quantile --variants deepchem seed0 seed1 seed2 seed3 seed4

WHY THIS EXISTS AS A SEPARATE TRAINER
-------------------------------------
`02_ENHANCEMENT_PLAN.md` §8 names CQR as a Phase 3 deliverable. CQR is not a scoring rule
that can be applied to an existing model's output: it needs a model that predicts an
*interval* -- a low and a high conditional quantile -- and every model in this project
predicts a single conditional mean. Nothing already archived can supply one.

It is deliberately not routed through `src/train/loop.py`. That loop is shared by every
committed result in the repository and is built around a scalar prediction per task with an
early-stopping criterion of AUC or RMSE; adding a quantile branch would touch the code path
behind Phases 1 and 2. The objective, the early-stopping criterion and the saved output all
differ here, so a ~120-line separate trainer is cheaper and safer than a flag.

WHAT IT TRAINS
--------------
The descriptor view, because Phase 1 measured it as the strongest single view on all three
regression datasets (ESOL, Lipophilicity, FreeSolv) and it costs seconds per split. The
encoder, the projection width and the head are exactly the ones `SingleViewModel` gives
every other view -- only the output width (2 quantiles per task instead of 1 mean) and the
loss differ, so the interval is not being produced by a quietly better model.

THE OBJECTIVE
-------------
Pinball (quantile) loss. For target quantile tau, an over-prediction is charged (1-tau) per
unit and an under-prediction tau per unit; the minimiser is the tau-th conditional quantile.
Fitting tau = alpha/2 and tau = 1-alpha/2 gives a nominal (1-alpha) interval directly --
before any conformal correction, which is the point: CQR then repairs whatever miscalibration
that interval has, and the size of the repair is itself a measurement.

Early stopping is on validation pinball loss, not RMSE. A model tuned for the mean is not
tuned for the tails, and stopping on the wrong criterion would hand CQR a worse interval
than the method deserves.

WHAT IT WRITES
--------------
`results/preds/<ds>_<tag>_<split>.npy`, shape (n_molecules, 2) -- the low and high quantile
in the same z-scored units every other saved prediction uses, so `src/eval/conformal.py`
converts them back to chemical units through the same `label_scale` path and no special
case is needed on the reading side.
"""

import argparse
import os

import numpy as np
import torch

from src.data.materialize import active_variant, materialize
from src.eval.metrics import is_classification
from src.models.encoders.descriptor import DescriptorEncoder
from src.models.heads import EMBED_DIM, SingleViewModel
from src.utils.seed import set_seed

DATA_DIR = "data"
PRED_DIR = os.path.join("results", "preds")
RUNS_DIR = os.path.join("results", "runs")
SPLITS = ("train", "valid", "test")


def pinball(pred, y, taus):
    """
    Mean pinball loss over measured targets.

    `pred` is (B, n_tasks * n_taus) laid out task-major, `y` is (B, n_tasks), `taus` is a
    1-D tensor of target quantiles. NaN targets contribute nothing, matching every other
    loss in this project.
    """
    b, n_tasks = y.shape
    pred = pred.view(b, n_tasks, len(taus))
    err = y.unsqueeze(-1) - pred
    loss = torch.maximum(taus * err, (taus - 1.0) * err)
    mask = (~torch.isnan(y)).unsqueeze(-1).expand_as(loss)
    return (torch.nan_to_num(loss, nan=0.0) * mask).sum() / mask.sum().clamp(min=1)


def load_split(ds, split):
    ecfp = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"), allow_pickle=True)
    desc = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_desc.npz"), allow_pickle=True)
    return (torch.tensor(ecfp["X"], dtype=torch.float32),
            torch.tensor(desc["X"], dtype=torch.float32),
            torch.tensor(ecfp["y"], dtype=torch.float32))


def fit_one(ds, alpha, epochs, patience, batch_size, lr, weight_decay, verbose=True):
    """Train the two-quantile model on the currently materialised split."""
    seed = set_seed()
    parts = {s: load_split(ds, s) for s in SPLITS}
    n_tasks = parts["train"][2].shape[1]
    taus = torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float32)

    encoder = DescriptorEncoder.fit(parts["train"][0], parts["train"][1],
                                    hidden=EMBED_DIM, dropout=0.2)
    model = SingleViewModel(encoder, n_tasks=n_tasks * len(taus), embed_dim=EMBED_DIM,
                            head_hidden=EMBED_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    xe, xd, ytr = parts["train"]
    n = len(ytr)
    if verbose:
        print(f"\n=== {ds} (quantile tau={alpha / 2:.3f}/{1 - alpha / 2:.3f}) "
              f"[seed={seed}] ===")
        print(f"  {n} train molecules, {n_tasks} task(s), "
              f"{sum(p.numel() for p in model.parameters()):,} params")

    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            rows = perm[i:i + batch_size]
            # A batch of one puts the descriptor encoder's BatchNorm in the same failure
            # mode the bucketed sampler already guards against (FreeSolv is 513 train rows
            # against a batch of 128, so the last batch is one row).
            if len(rows) == 1:
                continue
            opt.zero_grad()
            loss = pinball(model((xe[rows], xd[rows])), ytr[rows], taus)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            ve, vd, yv = parts["valid"]
            val = float(pinball(model((ve, vd)), yv, taus))
        if val < best - 1e-6:
            best, bad, best_ep = val, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"  epoch {ep:>3}  val pinball={val:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    os.makedirs(PRED_DIR, exist_ok=True)
    with torch.no_grad():
        for s in ("valid", "test"):
            xe_s, xd_s, _ = parts[s]
            q = model((xe_s, xd_s)).view(len(xd_s), n_tasks, len(taus))[:, 0, :].numpy()
            # A quantile crossing (low above high) is a known pathology of fitting the two
            # independently. Sorting is the standard repair and cannot hurt coverage: it
            # only ever widens toward the correct ordering.
            np.save(os.path.join(PRED_DIR, f"{ds}_qdesc_{s}.npy"), np.sort(q, axis=1))
    if verbose:
        print(f"  best val pinball={best:.4f} at epoch {best_ep}")
    return best


def archive(variant, datasets):
    dest = os.path.join(RUNS_DIR, variant, "preds")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for ds in datasets:
        for s in ("valid", "test"):
            src = os.path.join(PRED_DIR, f"{ds}_qdesc_{s}.npy")
            if os.path.exists(src):
                np.save(os.path.join(dest, f"{ds}_qdesc_{s}.npy"), np.load(src))
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Conditional-quantile regressor for CQR.")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="default: every regression dataset on disk")
    ap.add_argument("--variants", nargs="+",
                    default=["deepchem"] + [f"seed{i}" for i in range(5)])
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="miss rate; fits the alpha/2 and 1-alpha/2 quantiles")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    import json
    pool_index = json.load(open(os.path.join("data", "pool", "pool_index.json")))
    datasets = args.datasets or [d for d in pool_index if not is_classification(d)]
    restore = active_variant()

    for v in args.variants:
        print(f"\n{'=' * 70}\n{v}\n{'=' * 70}")
        materialize(v, verbose=False, artifacts=["ecfp", "desc"])
        for ds in datasets:
            fit_one(ds, args.alpha, args.epochs, args.patience, args.batch_size,
                    args.lr, args.weight_decay)
        print(f"  archived {archive(v, datasets)} prediction file(s) -> "
              f"{os.path.join(RUNS_DIR, v, 'preds')}")

    if restore and restore != args.variants[-1]:
        materialize(restore, verbose=False)
        print(f"\nrestored active split: {restore}")


if __name__ == "__main__":
    main()

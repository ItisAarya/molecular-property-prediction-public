"""
src/train/make_oof.py

Generate out-of-fold (OOF) base-model predictions over the training set.

    python -m src.train.make_oof                 # all datasets, all models
    python -m src.train.make_oof --models rf trf # skip the slow one while iterating

Writes results/preds/<ds>_<model>_oof.npy -- one prediction per training molecule, each
produced by a model that never saw that molecule during fitting.

WHY THIS IS NECESSARY
---------------------
A stacking meta-learner is trained on base-model predictions. If those predictions are
in-sample -- the base model was fitted on the very molecule it is predicting -- they are
far too good, and the meta-learner learns to trust them accordingly. It then meets
genuinely out-of-sample predictions at test time and its learned weights are wrong.

The inherited pipeline sidestepped this by fitting the meta-learner on the *validation*
split, where base predictions genuinely are out-of-sample. That is defensible in
isolation, but the same split was then also used to choose ensemble weights, fit
calibrators, tune thresholds and report validation performance. Fitting and evaluating on
the same rows is what produced BBBP validation AUC 0.99 against test AUC 0.75.

K-fold OOF fixes it properly. For each fold, base learners are fitted on the other K-1
folds and predict the held-out fold. Stitching the folds back together yields an honest
prediction for every training molecule. The meta-level then gets the whole training set to
learn from instead of a few hundred validation rows -- for ClinTox's CT_TOX task that is
roughly 95 positive examples instead of 2.

Early stopping inside a fold uses the real `valid` split. That is legitimate: `valid` is
never an OOF target, so no information flows from a held-out fold into the model that
predicts it.

COST
----
The transformer's encoder is frozen, so its embeddings come from the cache written by
scripts/cache_embeddings.py and each fold only trains a small MLP head -- seconds rather
than the ~75 minutes a full re-encode would take. The GNN genuinely retrains per fold and
dominates the runtime.
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from src.data.splits import kfold_indices, load_y
from src.eval.metrics import is_classification
from src.train.train_gnn import (
    GINNet,
    compute_pos_weight,
    load_graphs,
    masked_bce_with_logits,
    masked_mse,
    predict as gnn_predict,
)
from src.train.train_transformer import masked_bce_with_logits as trf_masked_bce
from src.train.train_transformer import masked_mse as trf_masked_mse
from src.utils.seed import set_seed

DATA_DIR = "data"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
N_FOLDS = 5

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)


def oof_path(ds, model):
    return os.path.join(PRED_DIR, f"{ds}_{model}_oof.npy")


def load_ecfp(ds, split):
    d = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"))
    return d["X"].astype(np.float32)


def load_embeddings(ds, split, pooling="cls"):
    path = os.path.join(DATA_DIR, f"{ds}_{split}_chemberta.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run: python -m scripts.cache_embeddings"
        )
    return np.load(path)[pooling].astype(np.float32)


# --------------------------------------------------------------------------------------
# Random Forest
# --------------------------------------------------------------------------------------
def oof_rf(ds, X, y, folds, cls):
    """One forest per (fold, task). Rows without a label for a task are excluded."""
    n, n_tasks = y.shape
    oof = np.full((n, n_tasks), np.nan, dtype=np.float32)

    for k, hold in enumerate(folds):
        fit_idx = np.setdiff1d(np.arange(n), hold)

        if cls:
            for t in range(n_tasks):
                col = y[fit_idx, t]
                labelled = ~np.isnan(col)
                classes = np.unique(col[labelled])
                if labelled.sum() == 0 or len(classes) < 2:
                    # Nothing to learn: emit the base rate so the column keeps its shape.
                    oof[hold, t] = float(classes[0]) if len(classes) == 1 else 0.5
                    continue
                rf = RandomForestClassifier(
                    n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
                )
                rf.fit(X[fit_idx][labelled], col[labelled])
                oof[hold, t] = rf.predict_proba(X[hold])[:, 1]
        else:
            col = y[fit_idx, 0]
            labelled = ~np.isnan(col)
            scaler = StandardScaler().fit(X[fit_idx][labelled])
            rf = RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1)
            rf.fit(scaler.transform(X[fit_idx][labelled]), col[labelled])
            oof[hold, 0] = rf.predict(scaler.transform(X[hold]))

        print(f"    rf fold {k + 1}/{len(folds)} done ({len(hold)} held out)")

    return oof


# --------------------------------------------------------------------------------------
# Transformer head on cached embeddings
# --------------------------------------------------------------------------------------
class CachedHead(nn.Module):
    """The same head shape train_transformer.py uses, but fed precomputed embeddings."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_in, d_in), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_in, d_out),
        )

    def forward(self, x):
        return self.head(x)


def oof_trf(ds, emb, y, folds, cls, epochs=5, bsz=16, lr=1e-3):
    n, n_tasks = y.shape
    oof = np.full((n, n_tasks), np.nan, dtype=np.float32)
    y_t = torch.from_numpy(y)
    emb_t = torch.from_numpy(emb)

    for k, hold in enumerate(folds):
        set_seed()  # identical initialisation per fold, so folds differ only by data
        fit_idx = np.setdiff1d(np.arange(n), hold)

        model = CachedHead(emb.shape[1], n_tasks)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

        for _ in range(epochs):
            model.train()
            for i in range(0, len(fit_idx), bsz):
                batch = fit_idx[i:i + bsz]
                logits = model(emb_t[batch])
                target = y_t[batch]
                loss = trf_masked_bce(logits, target) if cls else trf_masked_mse(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            out = model(emb_t[hold]).numpy()
        oof[hold] = 1.0 / (1.0 + np.exp(-out)) if cls else out
        print(f"    trf fold {k + 1}/{len(folds)} done ({len(hold)} held out)")

    return oof


# --------------------------------------------------------------------------------------
# GNN
# --------------------------------------------------------------------------------------
def oof_gnn(ds, graphs, y, folds, cls, epochs=100, patience=15, bsz=128, lr=1e-3, wd=1e-4):
    """
    Retrain the GIN per fold. Early stopping uses the real `valid` split, which is never
    an OOF target, so no held-out fold information reaches the model predicting it.
    """
    n, n_tasks = y.shape
    oof = np.full((n, n_tasks), np.nan, dtype=np.float32)

    val_graphs, _ = load_graphs(ds, "valid")
    val_loader = DataLoader(val_graphs, batch_size=bsz, shuffle=False)
    y_val = torch.stack([g.y for g in val_graphs]).numpy()

    from src.eval.metrics import cls_metrics, reg_metrics

    for k, hold in enumerate(folds):
        set_seed()
        t0 = time.time()
        fit_idx = np.setdiff1d(np.arange(n), hold)

        fit_graphs = [graphs[i] for i in fit_idx]
        hold_graphs = [graphs[i] for i in hold]
        fit_loader = DataLoader(fit_graphs, batch_size=bsz, shuffle=True)
        hold_loader = DataLoader(hold_graphs, batch_size=bsz, shuffle=False)

        model = GINNet(in_dim=graphs[0].x.size(1), hidden=256, out_dim=n_tasks, dropout=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        pos_w = compute_pos_weight(fit_graphs, n_tasks) if cls else None

        best_score, best_state, bad = float("-inf"), None, 0
        for _ in range(epochs):
            model.train()
            for batch in fit_loader:
                logits = model(batch)
                yb = torch.stack([g.y for g in batch.to_data_list()])
                loss = masked_bce_with_logits(logits, yb, pos_w) if cls else masked_mse(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            p_val = gnn_predict(model, val_loader, "cpu", cls)
            score = cls_metrics(y_val, p_val)["auc"] if cls else -reg_metrics(y_val, p_val, ds)["rmse"]
            if score > best_score:
                best_score, bad = score, 0
                best_state = {kk: v.cpu().clone() for kk, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        oof[hold] = gnn_predict(model, hold_loader, "cpu", cls)
        print(
            f"    gnn fold {k + 1}/{len(folds)} done ({len(hold)} held out, "
            f"{time.time() - t0:.0f}s)"
        )

    return oof


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------
def run_dataset(ds, models):
    y = load_y(ds, "train")
    cls = is_classification(ds)
    folds = kfold_indices(ds, y, n_folds=N_FOLDS)

    print(f"\n=== {ds} ({'classification' if cls else 'regression'}, "
          f"{y.shape[0]} train molecules, {y.shape[1]} task(s), {N_FOLDS} folds) ===")

    if "rf" in models:
        oof = oof_rf(ds, load_ecfp(ds, "train"), y, folds, cls)
        np.save(oof_path(ds, "rf"), oof)

    if "trf" in models:
        oof = oof_trf(ds, load_embeddings(ds, "train"), y, folds, cls)
        np.save(oof_path(ds, "trf"), oof)

    if "gnn" in models:
        graphs, _ = load_graphs(ds, "train")
        if len(graphs) != len(y):
            raise RuntimeError(
                f"{ds}: {len(graphs)} graphs vs {len(y)} labels -- re-run scripts.make_graphs"
            )
        oof = oof_gnn(ds, graphs, y, folds, cls)
        np.save(oof_path(ds, "gnn"), oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "trf", "gnn"])
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    datasets = args.datasets or list(meta.keys())

    start = time.time()
    for ds in datasets:
        run_dataset(ds, args.models)
    print(f"\nOOF generation finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()

# src/deploy/freeze_winners.py
import os, json, math, joblib, numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

# ---- config ----
ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA = ROOT / "data"
RESULTS = ROOT / "results"
WINNERS_JSON = RESULTS / "metrics" / "winners.json"
ART = ROOT / "models"  # where we will save frozen artifacts
HF_MODEL = "seyonec/ChemBERTa-zinc-base-v1"  # same family you used

CLASS_DATASETS = {"tox21", "bbbp", "clintox"}
REG_DATASETS = {"esol", "lipophilicity"}

# ---- utils ----
def load_npz(ds: str, split: str, kind="ecfp"):
    p = DATA / f"{ds}_{split}_{kind}.npz"
    d = np.load(p, allow_pickle=True)
    X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
    smiles = d["smiles"].tolist()
    return X, y, smiles

def is_classification(ds: str) -> bool:
    return ds in CLASS_DATASETS

def out_dim_from_y(y: np.ndarray) -> int:
    return int(y.shape[1]) if y.ndim == 2 else 1

# ---- RF freezer (scikit-learn) ----
def train_save_rf(ds: str):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    Xa, ya, _ = load_npz(ds, "train", "ecfp")
    Xv, yv, _ = load_npz(ds, "valid", "ecfp")
    X, y = np.vstack([Xa, Xv]), np.vstack([ya, yv])

    if is_classification(ds):
        model = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=-1, random_state=17, class_weight=None)
    else:
        model = RandomForestRegressor(n_estimators=800, max_depth=None, n_jobs=-1, random_state=17)

    model.fit(X, y.ravel() if y.shape[1]==1 else y)
    out_dir = ART / ds / "rf"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    meta = {"type":"rf","input":"ecfp","task":"classification" if is_classification(ds) else "regression",
            "ecfp_dim": int(X.shape[1])}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[RF] saved → {out_dir}")

# ---- Hybrid (ECFP + ChemBERTa embedding) ----
class HybridNet(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, task: str):
        super().__init__()
        self.task = task
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_hidden, d_hidden//2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_hidden//2, d_out)
        )

    def forward(self, x):
        return self.mlp(x)

class SmilesEmbedder:
    def __init__(self, model_name=HF_MODEL, device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.trf = AutoModel.from_pretrained(model_name)
        self.trf.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trf.to(self.device)

    @torch.no_grad()
    def encode(self, smiles: List[str], max_len=128, batch=32) -> np.ndarray:
        outs = []
        for i in range(0, len(smiles), batch):
            batch_sm = smiles[i:i+batch]
            enc = self.tok(batch_sm, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k:v.to(self.device) for k,v in enc.items()}
            out = self.trf(**enc)
            # prefer pooler if available, else CLS token
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:,0,:]
            outs.append(emb.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

class HybridDataset(Dataset):
    def __init__(self, X_ecfp: np.ndarray, X_trf: np.ndarray, y: np.ndarray, task: str):
        self.x = np.hstack([X_ecfp, X_trf]).astype(np.float32)
        self.y = y.astype(np.float32)
        self.task = task

    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i):
        xi = self.x[i]
        yi = self.y[i]
        return xi, yi

def bce_with_mask(logits, targets):
    # targets can be -1 where labels are missing (Tox21)
    mask = (targets != -1).float()
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.where(mask>0, targets, torch.zeros_like(targets)), reduction='none')
    loss = (loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
    return loss

def train_save_hybrid(ds: str):
    # load ECFP npz + SMILES for transformer embeddings
    Xa, ya, sa = load_npz(ds, "train", "ecfp")
    Xv, yv, sv = load_npz(ds, "valid", "ecfp")
    X, y, smi = np.vstack([Xa, Xv]), np.vstack([ya, yv]), (sa+sv)

    # build transformer embeddings
    emb = SmilesEmbedder()
    Xtrf = emb.encode(smi, max_len=128, batch=32)
    d_in = X.shape[1] + Xtrf.shape[1]
    d_out = out_dim_from_y(y)
    task = "classification" if is_classification(ds) else "regression"

    ds_train = HybridDataset(X, Xtrf, y, task)
    loader = DataLoader(ds_train, batch_size=64, shuffle=True, drop_last=False)

    model = HybridNet(d_in=d_in, d_hidden=512, d_out=d_out, task=task)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    epochs = 15 if task=="classification" else 30
    for ep in range(1, epochs+1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            if task=="classification":
                if yb.ndim==1 or yb.shape[1]==1:
                    yb = yb.view(-1,1)
                loss = bce_with_mask(logits, yb)
            else:
                loss = nn.functional.mse_loss(logits.view_as(yb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()*xb.size(0)
        if ep % 5 == 0 or ep==1:
            print(f"[{ds} HYBRID] epoch {ep}/{epochs} loss={tot/len(ds_train):.4f}")

    # save
    out_dir = ART / ds / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "d_in": d_in, "d_out": d_out, "task": task,
                "ecfp_dim": int(X.shape[1]), "trf_dim": int(Xtrf.shape[1]),
                "hf_model": HF_MODEL}, out_dir / "model.pt")
    meta = {"type":"hybrid","input":"ecfp+trf","task":task}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[HYBRID] saved → {out_dir}")

# ---- TRF (pure ChemBERTa) only for ClinTox (winner ensemble uses trf) ----
class TrfHead(nn.Module):
    def __init__(self, d_in, d_out, task):
        super().__init__()
        self.task = task
        self.head = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, d_out))
    def forward(self, x): return self.head(x)

def train_save_trf(ds: str):
    assert ds=="clintox", "TRF freezer only implemented for clintox here."
    Xa, ya, sa = load_npz(ds, "train", "ecfp")  # just for SMILES list
    Xv, yv, sv = load_npz(ds, "valid", "ecfp")
    y = np.vstack([ya, yv])
    smi = sa + sv
    emb = SmilesEmbedder()
    Xtrf = emb.encode(smi, max_len=128, batch=32)
    d_in = Xtrf.shape[1]; d_out = out_dim_from_y(y)
    task = "classification"

    class SimpleDS(Dataset):
        def __len__(self): return len(smi)
        def __getitem__(self, i): return Xtrf[i].astype(np.float32), y[i].astype(np.float32)

    ds_train = SimpleDS()
    loader = DataLoader(ds_train, batch_size=64, shuffle=True)
    model = TrfHead(d_in, d_out, task)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for ep in range(1, 15+1):
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if yb.ndim==1: yb = yb.view(-1,1)
            opt.zero_grad()
            logits = model(xb)
            loss = bce_with_mask(logits, yb)
            loss.backward(); opt.step()
            tot += loss.item()*xb.size(0)
        if ep % 5 == 0 or ep==1:
            print(f"[{ds} TRF] epoch {ep}/15 loss={tot/len(ds_train):.4f}")

    out_dir = ART / ds / "trf"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "d_in": d_in, "d_out": d_out, "task": task,
                "hf_model": HF_MODEL}, out_dir / "model.pt")
    meta = {"type":"trf","input":"trf","task":task}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[TRF] saved → {out_dir}")

# ---- main: read winners.json and freeze each winner ----
def main():
    winners = json.loads(WINNERS_JSON.read_text())
    ART.mkdir(parents=True, exist_ok=True)

    for ds, cfg in winners.items():
        m = cfg["model"]
        print(f"== Freezing {ds} → {m} ==")
        if m == "rf":
            train_save_rf(ds)
        elif m == "hybrid":
            train_save_hybrid(ds)
        elif m == "trf":
            # not used by winners, but kept for completeness
            train_save_trf(ds)
        elif m == "ens":
            # For ensembles, freeze the needed components + save an ensemble.json
            ens = cfg["ensemble"]
            need = set(ens["models"])
            if "rf" in need: train_save_rf(ds)
            if "hybrid" in need: train_save_hybrid(ds)
            if "trf" in need: 
                if ds != "clintox":
                    print(f"  [warn] TRF freezing skipped for {ds}; no TRF head implemented.")
                else:
                    train_save_trf(ds)

            out_dir = ART / ds / "ensemble"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save config with weights/thresholds
            ens_cfg = {
                "models": ens["models"],
                "weights": ens["weights"],
                "thresholds": cfg.get("thresholds", {}).get("thresholds", None),
                "metric": cfg.get("thresholds", {}).get("metric", None)
            }
            (out_dir / "ensemble.json").write_text(json.dumps(ens_cfg, indent=2))
            print(f"[ENS] wrote config → {out_dir / 'ensemble.json'}")
        else:
            print(f"  [skip] Unknown winner type {m}")

        # Always store thresholds next to model
        thr = cfg.get("thresholds", None)
        if thr:
            # choose the subdir that holds the core model
            target = ART / ds / (m if m != "ens" else "ensemble")
            (target / "thresholds.json").write_text(json.dumps(thr, indent=2))

    print("\nAll winners frozen into ./models/<dataset>/<model>/")

if __name__ == "__main__":
    main()

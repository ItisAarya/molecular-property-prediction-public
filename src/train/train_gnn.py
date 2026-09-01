# src/train/train_gnn.py
import os, json, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm

from src.eval.metrics import is_classification, cls_metrics, reg_metrics
from src.utils.seed import set_seed

DATA_DIR = "data"
MODELS_DIR = "models"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)

# ---------------- Model ----------------
class GINNet(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=1, dropout=0.3):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINConv(self.mlp1)
        self.bn1   = BatchNorm(hidden)
        self.conv2 = GINConv(self.mlp2)
        self.bn2   = BatchNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.lin   = nn.Linear(hidden, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index); x = self.bn1(x); x = F.relu(x)
        x = self.conv2(x, edge_index); x = self.bn2(x); x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.lin(x)

# ------------- Utilities --------------
def load_graphs(ds, split):
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_graphs.pt"))
    return obj["graphs"], obj["tasks"]

def compute_pos_weight(graphs, T):
    Y = torch.stack([g.y for g in graphs])   # [N, T]
    w = []
    for t in range(T):
        yt = Y[:, t]
        mask = ~torch.isnan(yt)
        pos = (yt[mask] == 1).sum().item()
        neg = (yt[mask] == 0).sum().item()
        w.append((neg / max(pos, 1)) if pos > 0 else 1.0)
    return torch.tensor(w, dtype=torch.float32)

def masked_bce_with_logits(logits, y, pos_weight=None):
    mask = ~torch.isnan(y)
    y_f = torch.nan_to_num(y, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, y_f, pos_weight=pos_weight, reduction='none')
    loss = loss * mask
    denom = mask.sum().clamp(min=1)
    return loss.sum() / denom

def masked_mse(logits, y):
    mask = ~torch.isnan(y)
    y_f = torch.nan_to_num(y, nan=0.0)
    loss = F.mse_loss(logits, y_f, reduction='none')
    loss = loss * mask
    denom = mask.sum().clamp(min=1)
    return loss.sum() / denom

@torch.no_grad()
def predict(model, loader, device, cls: bool):
    model.eval(); outs=[]
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch).cpu().numpy()
        outs.append(logits)
    P = np.vstack(outs)
    if cls: P = 1/(1+np.exp(-P))
    return P

def get_y_numpy(graphs):
    return torch.stack([g.y for g in graphs]).numpy()

# ------------- Training loop --------------
def run_one(ds, epochs=100, batch_size=128, lr=1e-3, weight_decay=1e-4, patience=15, device="cpu"):
    # Seed per dataset so a single-dataset run reproduces the all-dataset run.
    seed = set_seed()
    gtr, tasks = load_graphs(ds, "train")
    gva, _     = load_graphs(ds, "valid")
    gte, _     = load_graphs(ds, "test")

    in_dim = gtr[0].x.size(1)
    T = gtr[0].y.numel()
    cls = is_classification(ds)

    Ltr = DataLoader(gtr, batch_size=batch_size, shuffle=True)
    Lva = DataLoader(gva, batch_size=batch_size, shuffle=False)
    Lte = DataLoader(gte, batch_size=batch_size, shuffle=False)

    model = GINNet(in_dim=in_dim, hidden=256, out_dim=T, dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_w = compute_pos_weight(gtr, T).to(device) if cls else None

    # FIX: unified early-stopping score (maximize)
    best_score = float("-inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs+1):
        model.train(); ep_loss = 0.0
        for batch in Ltr:
            batch = batch.to(device)
            logits = model(batch)
            yb = torch.stack([g.y for g in batch.to_data_list()]).to(device)
            loss = masked_bce_with_logits(logits, yb, pos_w) if cls else masked_mse(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        # validate
        Pva = predict(model, Lva, device, cls)
        yva = get_y_numpy(gva)
        if cls:
            m = cls_metrics(yva, Pva)        # maximize AUC
            score = m["auc"]
        else:
            m = reg_metrics(yva, Pva)        # maximize -RMSE
            score = -m["rmse"]

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0:
            if cls:
                print(f"{ds} Epoch {ep}/{epochs} loss={ep_loss:.4f} val=AUC={m['auc']:.3f}")
            else:
                print(f"{ds} Epoch {ep}/{epochs} loss={ep_loss:.4f} valRMSE={-score:.3f}")

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # predict & save
    Pva = predict(model, Lva, device, cls)
    Pte = predict(model, Lte, device, cls)
    np.save(os.path.join(PRED_DIR, f"{ds}_gnn_valid.npy"), Pva)
    np.save(os.path.join(PRED_DIR, f"{ds}_gnn_test.npy"),  Pte)

    yva = get_y_numpy(gva)
    yte = get_y_numpy(gte)

    m_va = cls_metrics(yva, Pva) if cls else reg_metrics(yva, Pva)
    m_te = cls_metrics(yte, Pte) if cls else reg_metrics(yte, Pte)

    pd.DataFrame([m_va]).to_csv(os.path.join(MET_DIR, f"{ds}_gnn_valid.csv"), index=False)
    pd.DataFrame([m_te]).to_csv(os.path.join(MET_DIR, f"{ds}_gnn_test.csv"),  index=False)
    print(f"{ds} GNN (GIN) done [seed={seed}]. Valid:", m_va, "Test:", m_te)

def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta.keys():
        run_one(ds)

if __name__ == "__main__":
    main()

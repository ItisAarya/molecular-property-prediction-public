import os, json, numpy as np, torch, torch.nn as nn, pandas as pd
from transformers import AutoModel
from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA_DIR="data"; MODELS_DIR="models"; PRED_DIR="results/preds"; MET_DIR="results/metrics"
os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(PRED_DIR, exist_ok=True); os.makedirs(MET_DIR, exist_ok=True)

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

def load_tok(ds, split):
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_tok.pt"))
    return obj["input_ids"], obj["attention_mask"], obj["y"].float(), obj["tasks"]

class TrfHead(nn.Module):
    def __init__(self, base, out_dim):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # freeze encoder
        hid = base.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hid, out_dim)
        )
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = out[:,0,:]  # CLS token
        return self.head(pooled)

def batch_loader(ids, masks, ys=None, bsz=16):
    """Yields (ids, masks, ys) during training; (ids, masks) for inference."""
    N = ids.size(0)
    for i in range(0, N, bsz):
        if ys is None:
            yield ids[i:i+bsz], masks[i:i+bsz]
        else:
            yield ids[i:i+bsz], masks[i:i+bsz], ys[i:i+bsz]

def run_ds(ds, epochs=5, batch_size=16, lr=1e-3, device="cpu"):
    ids_tr, m_tr, ytr, tasks = load_tok(ds, "train")
    ids_va, m_va, yva, _     = load_tok(ds, "valid")
    ids_te, m_te, yte, _     = load_tok(ds, "test")

    T = ytr.shape[1] if ytr.ndim==2 else 1

    base = AutoModel.from_pretrained(MODEL_NAME)
    model = TrfHead(base, out_dim=T).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if is_classification(ds) else nn.MSELoss()

    # ---- TRAIN ----
    for ep in range(1, epochs+1):
        model.train(); total = 0.0
        for ids_b, ma_b, ys_b in batch_loader(ids_tr, m_tr, ytr, batch_size):
            ids_b, ma_b, ys_b = ids_b.to(device), ma_b.to(device), ys_b.to(device)
            logits = model(ids_b, ma_b)
            # handle 1D target
            if ys_b.ndim == 1: ys_b = ys_b.unsqueeze(1)
            loss = loss_fn(logits, ys_b)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"{ds} TRF epoch {ep}/{epochs} loss={total:.4f}")

    # save head weights
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{ds}_trf.pt"))

    # ---- PREDICT ----
    def predict(ids, ma):
        model.eval(); outs=[]
        with torch.no_grad():
            for x, m in batch_loader(ids, ma, None, batch_size):
                x, m = x.to(device), m.to(device)
                logit = model(x, m).cpu().numpy()
                outs.append(logit)
        outs = np.vstack(outs)
        if is_classification(ds):
            return 1.0/(1.0 + np.exp(-outs))  # sigmoid
        return outs

    p_va = predict(ids_va, m_va)
    p_te = predict(ids_te, m_te)

    # save predictions for stacking
    np.save(os.path.join(PRED_DIR, f"{ds}_trf_valid.npy"), p_va)
    np.save(os.path.join(PRED_DIR, f"{ds}_trf_test.npy"),  p_te)

    # metrics
    if is_classification(ds):
        m_va = cls_metrics(yva, p_va)
        m_te = cls_metrics(yte, p_te)
    else:
        m_va = reg_metrics(yva, p_va)
        m_te = reg_metrics(yte, p_te)

    pd.DataFrame([m_va]).to_csv(os.path.join(MET_DIR, f"{ds}_trf_valid.csv"), index=False)
    pd.DataFrame([m_te]).to_csv(os.path.join(MET_DIR, f"{ds}_trf_test.csv"),  index=False)
    print(ds, "TRF done. Valid:", m_va, "Test:", m_te)

def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta.keys():
        run_ds(ds)

if __name__ == "__main__":
    main()

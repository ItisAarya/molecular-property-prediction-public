import os, json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA_DIR="data"; PRED_DIR="results/preds"; MET_DIR="results/metrics"
os.makedirs(MET_DIR, exist_ok=True)

def load_y(ds, split):
    d = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"))
    return d["y"].astype(np.float32)

def load_preds(ds, model, split):
    return np.load(os.path.join(PRED_DIR, f"{ds}_{model}_{split}.npy"))

def run_ds(ds):
    # collect base preds
    P_va, P_te, names = [], [], []
    for m in ["rf","gnn","trf"]:
        pva = load_preds(ds, m, "valid")
        pte = load_preds(ds, m, "test")
        # ensure 2D
        if pva.ndim==1: pva=pva.reshape(-1,1)
        if pte.ndim==1: pte=pte.reshape(-1,1)
        P_va.append(pva); P_te.append(pte); names.append(m)

    Xva = np.hstack(P_va)
    Xte = np.hstack(P_te)

    yva = load_y(ds, "valid")
    yte = load_y(ds, "test")

    if is_classification(ds):
        # fit per-task logistic regression
        T = yva.shape[1] if yva.ndim==2 else 1
        probs_va, probs_te = [], []
        for t in range(T):
            yt = yva[:,t] if T>1 else yva.reshape(-1)
            # Some tasks can be NaN-heavy; filter valid rows
            mask = ~np.isnan(yt)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xva[mask], yt[mask])
            probs_va.append(clf.predict_proba(Xva)[:,1])
            probs_te.append(clf.predict_proba(Xte)[:,1])
        Pva = np.vstack(probs_va).T
        Pte = np.vstack(probs_te).T
        m_va = cls_metrics(yva, Pva)
        m_te = cls_metrics(yte, Pte)
        np.save(os.path.join(PRED_DIR, f"{ds}_hybrid_valid.npy"), Pva)
        np.save(os.path.join(PRED_DIR, f"{ds}_hybrid_test.npy"), Pte)
    else:
        # regression ridge stack
        yva1 = yva.reshape(-1)
        yte1 = yte.reshape(-1)
        reg = Ridge(alpha=1.0).fit(Xva, yva1)
        Pva = reg.predict(Xva).reshape(-1,1)
        Pte = reg.predict(Xte).reshape(-1,1)
        m_va = reg_metrics(yva1, Pva, ds)
        m_te = reg_metrics(yte1, Pte, ds)
        np.save(os.path.join(PRED_DIR, f"{ds}_hybrid_valid.npy"), Pva)
        np.save(os.path.join(PRED_DIR, f"{ds}_hybrid_test.npy"), Pte)

    pd.DataFrame([m_va]).to_csv(os.path.join(MET_DIR, f"{ds}_hybrid_valid.csv"), index=False)
    pd.DataFrame([m_te]).to_csv(os.path.join(MET_DIR, f"{ds}_hybrid_test.csv"), index=False)
    print(ds, "HYBRID done. Valid:", m_va, "Test:", m_te)

def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta.keys():
        run_ds(ds)

if __name__ == "__main__":
    main()

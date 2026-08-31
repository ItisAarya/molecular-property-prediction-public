import os, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from src.eval.metrics import is_classification, cls_metrics, reg_metrics

DATA_DIR = "data"
MODELS_DIR = "models"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MET_DIR, exist_ok=True)

def load_npz(ds, split):
    d = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"))
    return d["X"].astype(np.float32), d["y"].astype(np.float32), d["smiles"]

def train_one_dataset(ds):
    Xtr, ytr, _ = load_npz(ds, "train")
    Xva, yva, _ = load_npz(ds, "valid")
    Xte, yte, _ = load_npz(ds, "test")

    if is_classification(ds):
        # multi-task: train per task and average metrics
        n_tasks = ytr.shape[1] if ytr.ndim==2 else 1
        probs_va, probs_te = [], []
        for t in range(n_tasks):
            ytr_t = ytr[:,t] if n_tasks>1 else ytr.reshape(-1)
            yva_t = yva[:,t] if n_tasks>1 else yva.reshape(-1)
            yte_t = yte[:,t] if n_tasks>1 else yte.reshape(-1)
            rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
            rf.fit(Xtr, ytr_t)
            joblib.dump(rf, os.path.join(MODELS_DIR, f"{ds}_rf_task{t}.pkl"))
            probs_va.append(rf.predict_proba(Xva)[:,1])
            probs_te.append(rf.predict_proba(Xte)[:,1])
        probs_va = np.vstack(probs_va).T  # [Nval, T]
        probs_te = np.vstack(probs_te).T
        # metrics
        m_va = cls_metrics(yva, probs_va)
        m_te = cls_metrics(yte, probs_te)
        # save preds
        np.save(os.path.join(PRED_DIR, f"{ds}_rf_valid.npy"), probs_va)
        np.save(os.path.join(PRED_DIR, f"{ds}_rf_test.npy"), probs_te)
    else:
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
        joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ds}_rf_scaler.pkl"))
        rf = RandomForestRegressor(n_estimators=800, random_state=42, n_jobs=-1)
        rf.fit(Xtr_s, ytr.reshape(-1))
        joblib.dump(rf, os.path.join(MODELS_DIR, f"{ds}_rf.pkl"))
        pred_va = rf.predict(Xva_s).reshape(-1,1)
        pred_te = rf.predict(Xte_s).reshape(-1,1)
        m_va = reg_metrics(yva, pred_va)
        m_te = reg_metrics(yte, pred_te)
        np.save(os.path.join(PRED_DIR, f"{ds}_rf_valid.npy"), pred_va)
        np.save(os.path.join(PRED_DIR, f"{ds}_rf_test.npy"), pred_te)

    # save metrics
    pd.DataFrame([m_va]).to_csv(os.path.join(MET_DIR, f"{ds}_rf_valid.csv"), index=False)
    pd.DataFrame([m_te]).to_csv(os.path.join(MET_DIR, f"{ds}_rf_test.csv"), index=False)
    print(ds, "RF done. Valid:", m_va, "Test:", m_te)

def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta.keys():
        train_one_dataset(ds)

if __name__ == "__main__":
    main()

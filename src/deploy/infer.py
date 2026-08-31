# src/deploy/infer.py
import json, joblib, numpy as np, torch, torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any, Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from transformers import AutoTokenizer, AutoModel

# ---- paths / defaults --------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "models"            # where freeze_winners saved your artifacts
HF_DEFAULT = "seyonec/ChemBERTa-zinc-base-v1"


# ---- utils -------------------------------------------------------------------
def ecfp_from_smiles(smiles: List[str], n_bits=2048, radius=2) -> np.ndarray:
    arr = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        onbits = list(fp.GetOnBits())
        arr[i, onbits] = 1.0
    return arr


def _load_state_dict_compat(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """
    Make checkpoints saved with prefixes ('head.', 'module.') compatible with the
    model definitions used here (Sequential with numeric keys or HybridNet with 'net.*').
    """
    model_keys = list(model.state_dict().keys())
    expects_sequential_top = model_keys and model_keys[0].split(".")[0].isdigit()
    expects_net_prefix = any(k.startswith("net.") for k in model_keys)

    fixed = {}
    for k, v in state_dict.items():
        k2 = k

        # Strip DataParallel prefix if present
        if k2.startswith("module."):
            k2 = k2[len("module."):]

        # Map 'head.*' (training-time naming) to the structure we expect at inference
        if k2.startswith("head."):
            if expects_net_prefix:
                # e.g., 'head.0.weight' -> 'net.0.weight'
                k2 = "net." + k2[len("head."):]
            elif expects_sequential_top:
                # e.g., 'head.0.weight' -> '0.weight'
                k2 = k2[len("head."):]

        fixed[k2] = v

    # Load non-strict to be forgiving if there are harmless extras
    model.load_state_dict(fixed, strict=False)


# ---- simple modules ----------------------------------------------------------
class HybridNet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, task):
        super().__init__()
        self.task = task
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, d_out),
        )

    def forward(self, x):
        return self.net(x)


class TrfEmbedder:
    def __init__(self, model_name=HF_DEFAULT, device=None):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.trf = AutoModel.from_pretrained(model_name)
        self.trf.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trf.to(self.device)

    @torch.no_grad()
    def encode(self, smiles, max_len=128, batch=32):
        outs = []
        for i in range(0, len(smiles), batch):
            batch_sm = smiles[i:i + batch]
            enc = self.tok(
                batch_sm,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.trf(**enc)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out.last_hidden_state[:, 0, :]
            outs.append(emb.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)


# ---- predictor base ----------------------------------------------------------
class Predictor:
    def __init__(self, dataset: str, thresholds: Optional[List[float]] = None):
        self.dataset = dataset
        self.thresholds = thresholds

    def postprocess_cls(self, probs: np.ndarray) -> Dict[str, Any]:
        thr = (
            np.array(self.thresholds)
            if self.thresholds is not None
            else np.full(probs.shape[1], 0.5)
        )
        labels = (probs >= thr).astype(int)
        return {
            "probs": probs.tolist(),
            "labels": labels.tolist(),
            "thresholds": thr.tolist(),
        }


# ---- RF ----------------------------------------------------------------------
class RFPredictor(Predictor):
    def __init__(self, dataset: str):
        super().__init__(dataset)
        base = ART / dataset / "rf"
        self.cfg = json.loads((base / "meta.json").read_text())
        self.model = joblib.load(base / "model.joblib")
        thr_path = base / "thresholds.json"
        if thr_path.exists():
            self.thresholds = json.loads(thr_path.read_text())["thresholds"]

    def predict(self, smiles: List[str]) -> Dict[str, Any]:
        X = ecfp_from_smiles(smiles, n_bits=int(self.cfg["ecfp_dim"]))
        if self.cfg["task"] == "classification":
            probs = self.model.predict_proba(X)
            # scikit-learn returns a list for multi-output classification
            if isinstance(probs, list):
                probs = np.stack([p[:, 1] for p in probs], axis=1)
            else:
                probs = probs[:, 1:2]  # single-task
            return self.postprocess_cls(probs)
        else:
            y = self.model.predict(X).reshape(-1, 1)
            return {"y": y.tolist()}


# ---- Hybrid (ECFP + Transformer embedding) ----------------------------------
class HybridPredictor(Predictor):
    def __init__(self, dataset: str):
        base = ART / dataset / "hybrid"
        ckpt = torch.load(base / "model.pt", map_location="cpu")
        super().__init__(dataset)
        self.task = ckpt["task"]
        self.ecfp_dim = ckpt["ecfp_dim"]
        self.trf_dim = ckpt["trf_dim"]
        self.embedder = TrfEmbedder(ckpt.get("hf_model", HF_DEFAULT))

        self.model = HybridNet(
            d_in=ckpt["d_in"], d_hidden=512, d_out=ckpt["d_out"], task=self.task
        )
        _load_state_dict_compat(self.model, ckpt["state_dict"])
        self.model.eval()

        thr_path = base / "thresholds.json"
        if thr_path.exists():
            self.thresholds = json.loads(thr_path.read_text())["thresholds"]

    @torch.no_grad()
    def predict(self, smiles: List[str]) -> Dict[str, Any]:
        X_ecfp = ecfp_from_smiles(smiles, n_bits=self.ecfp_dim)
        X_trf = self.embedder.encode(smiles)
        x = torch.from_numpy(np.hstack([X_ecfp, X_trf]).astype(np.float32))
        logits = self.model(x)
        if self.task == "classification":
            probs = torch.sigmoid(logits).cpu().numpy()
            if probs.ndim == 1:
                probs = probs[:, None]
            return self.postprocess_cls(probs)
        else:
            y = logits.cpu().numpy()
            return {"y": y.tolist()}


# ---- Transformer head only ---------------------------------------------------
class TrfPredictor(Predictor):
    def __init__(self, dataset: str):
        base = ART / dataset / "trf"
        ckpt = torch.load(base / "model.pt", map_location="cpu")
        super().__init__(dataset)
        self.task = ckpt["task"]
        self.embedder = TrfEmbedder(ckpt.get("hf_model", HF_DEFAULT))
        self.model = nn.Sequential(
            nn.Linear(ckpt["d_in"], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, ckpt["d_out"]),
        )
        _load_state_dict_compat(self.model, ckpt["state_dict"])
        self.model.eval()

        thr_path = base / "thresholds.json"
        if thr_path.exists():
            self.thresholds = json.loads(thr_path.read_text())["thresholds"]

    @torch.no_grad()
    def predict(self, smiles: List[str]) -> Dict[str, Any]:
        X = self.embedder.encode(smiles)
        logits = self.model(torch.from_numpy(X))
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 1:
            probs = probs[:, None]
        return self.postprocess_cls(probs)


# ---- Ensemble ---------------------------------------------------------------
class EnsemblePredictor(Predictor):
    def __init__(self, dataset: str):
        base = ART / dataset / "ensemble"
        cfg = json.loads((base / "ensemble.json").read_text())
        super().__init__(dataset, thresholds=cfg.get("thresholds", None))
        self.members = []
        for m in cfg["models"]:
            if m == "rf":
                self.members.append(RFPredictor(dataset))
            elif m == "hybrid":
                self.members.append(HybridPredictor(dataset))
            elif m == "trf":
                self.members.append(TrfPredictor(dataset))
            else:
                raise ValueError(f"Unknown ensemble member: {m}")
        self.weights = np.array(cfg["weights"], dtype=np.float32)
        self.weights = self.weights / self.weights.sum()

    def predict(self, smiles: List[str]) -> Dict[str, Any]:
        outs = [pred.predict(smiles) for pred in self.members]

        # Classification: average probabilities
        if "probs" in outs[0]:
            P = np.zeros_like(np.array(outs[0]["probs"]), dtype=np.float32)
            for w, o in zip(self.weights, outs):
                P += w * np.array(o["probs"])
            return self.postprocess_cls(P)

        # Regression: average predictions
        Y = np.zeros_like(np.array(outs[0]["y"]), dtype=np.float32)
        for w, o in zip(self.weights, outs):
            Y += w * np.array(o["y"])
        return {"y": Y.tolist()}


# ---- entry point -------------------------------------------------------------
def load_winner(dataset: str):
    """
    Load the frozen 'winner' for a dataset, trying ensemble -> hybrid -> rf -> trf.
    """
    base = ART / dataset
    if (base / "ensemble").exists():
        return EnsemblePredictor(dataset)
    if (base / "hybrid").exists():
        return HybridPredictor(dataset)
    if (base / "rf").exists():
        return RFPredictor(dataset)
    if (base / "trf").exists():
        return TrfPredictor(dataset)
    raise FileNotFoundError(
        f"No frozen model for {dataset}. Run: python -m src.deploy.freeze_winners"
    )

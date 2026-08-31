# app/streamlit_app.py
from __future__ import annotations

import os
import sys
import json
import inspect
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import pandas as pd
import streamlit as st

# ---------- make `src` importable when running via `streamlit run app/...`
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- project imports
from src.deploy.infer import load_winner
from src.deploy.verbalize import add_narratives, TASKS

# ---------- constants & paths
RESULTS_DIR = ROOT / "results" / "metrics"
WINNERS_JSON = RESULTS_DIR / "winners.json"
THRESHOLDS_JSON = RESULTS_DIR / "thresholds.json"

CLASS_DATASETS = ("tox21", "bbbp", "clintox")
REGR_DATASETS = ("esol", "lipophilicity")

# ---------- utilities
def _first(d: dict, keys: List[str]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None

def load_thresholds(dataset: str) -> Optional[List[float]]:
    try:
        with open(THRESHOLDS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get(dataset)
        if entry is None:
            return None
        if isinstance(entry, dict) and "thresholds" in entry:
            return entry["thresholds"]
        if isinstance(entry, list):
            return entry
    except Exception:
        pass
    return None

def parse_smiles_block(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

def get_predictor(dataset: str):
    """
    Call load_winner(...) in a signature-agnostic way:
    1) kwarg 'winners_path'
    2) positional path
    3) no path (loader uses internal default)
    """
    try:
        sig = inspect.signature(load_winner)
        if "winners_path" in sig.parameters:
            return load_winner(dataset, winners_path=str(WINNERS_JSON))
    except Exception:
        pass
    try:
        return load_winner(dataset, str(WINNERS_JSON))
    except TypeError:
        return load_winner(dataset)

def _to_2d(a: Any) -> np.ndarray:
    """Convert anything list-like to a 2D numpy array without copying if already 2D."""
    arr = np.asarray(a)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def normalize_output_to_df(dataset: str, smiles_in: List[str], out: Any) -> pd.DataFrame:
    """
    Accepts whatever the predictor returned and produces a clean wide DataFrame:
    - 'smiles' column
    - For classifiers:
        single-task: prob, label
        multi-task: <task>_prob, <task>_label for each task
    - For regressors:
        single-task: pred
        multi-task: <task>_pred
    This function *slices to the common minimum length* so pandas never sees
    different-length columns.
    """
    N = len(smiles_in)
    tasks = TASKS.get(dataset, [])

    # Already a DataFrame
    if isinstance(out, pd.DataFrame):
        df = out.reset_index(drop=True)
        # ensure smiles exists
        if "smiles" not in df.columns:
            df.insert(0, "smiles", smiles_in[: len(df)])
        return df

    # Predictor returns a dict (most common)
    if isinstance(out, dict):
        # try common keys
        probs = _first(out, ["prob", "probs", "y_prob", "y_proba", "p"])
        labels = _first(out, ["label", "labels", "y_pred", "pred_labels"])
        values = _first(out, ["value", "values", "pred", "y"])  # regression
        smiles_out = out.get("smiles", smiles_in)

        # Determine target length (min across available arrays and smiles)
        lengths = [len(smiles_out)]
        for x in (probs, labels, values):
            if x is not None:
                x_arr = np.asarray(x)
                lengths.append(x_arr.shape[0])
        tgt = min([N] + [L for L in lengths if isinstance(L, (int, np.integer))])

        df = pd.DataFrame({"smiles": list(smiles_out)[:tgt]})

        # Classification path (probs present)
        if probs is not None:
            P = _to_2d(probs)[:tgt]
            T = P.shape[1]
            if not tasks or len(tasks) != T:
                tasks = [f"task{i+1}" for i in range(T)]
            for j, t in enumerate(tasks):
                df[f"{t}_prob"] = P[:, j]
            # labels
            if labels is not None:
                L = _to_2d(labels)[:tgt]
                for j, t in enumerate(tasks):
                    df[f"{t}_label"] = L[:, j].astype(int)
            else:
                thr = load_thresholds(dataset) or [0.5] * len(tasks)
                if len(thr) != len(tasks):
                    thr = [0.5] * len(tasks)
                for j, t in enumerate(tasks):
                    df[f"{t}_label"] = (df[f"{t}_prob"].values >= float(thr[j])).astype(int)
            return df

        # Regression path (values present)
        if values is not None:
            V = _to_2d(values)[:tgt]
            T = V.shape[1]
            if T == 1:
                df["pred"] = V[:, 0]
            else:
                if not tasks or len(tasks) != T:
                    tasks = [f"task{i+1}" for i in range(T)]
                for j, t in enumerate(tasks):
                    df[f"{t}_pred"] = V[:, j]
            return df

        # Fallback: collect any keys whose value is 1-D of length >= tgt
        for k, v in out.items():
            arr = np.asarray(v)
            if arr.ndim == 1 and len(arr) >= tgt:
                df[k] = arr[:tgt]
        return df

    # List / ndarray output
    arr = np.asarray(out)
    tgt = min(N, arr.shape[0])
    df = pd.DataFrame({"smiles": smiles_in[:tgt]})
    if dataset in REGR_DATASETS:
        if arr.ndim == 1:
            df["pred"] = arr[:tgt]
        else:
            T = arr.shape[1]
            names = tasks if tasks and len(tasks) == T else [f"task{i+1}" for i in range(T)]
            for j, t in enumerate(names):
                df[f"{t}_pred"] = arr[:tgt, j]
    else:
        # classification single task assumption
        if arr.ndim > 1:
            arr = arr[:, 0]
        df["prob"] = arr[:tgt]
        thr = load_thresholds(dataset) or [0.5]
        df["label"] = (df["prob"].values >= float(thr[0])).astype(int)
    return df

def run_model(dataset: str, smiles: List[str]) -> pd.DataFrame:
    predictor = get_predictor(dataset)
    # Call different predictor shapes
    if callable(predictor):
        out = predictor(smiles)
    elif hasattr(predictor, "predict"):
        out = predictor.predict(smiles)
    else:
        for attr in ("run", "infer"):
            if hasattr(predictor, attr):
                out = getattr(predictor, attr)(smiles)
                break
        else:
            raise RuntimeError("Loaded predictor is not callable and has no .predict/.run/.infer API.")
    return normalize_output_to_df(dataset, smiles, out)

# ---------- UI
st.set_page_config(page_title="Molecular Property Prediction", layout="wide")
st.title("🧪 Molecular Property Prediction — Hybrid/Ensemble")
st.caption(
    "Single-platform data (MoleculeNet via DeepChem). Models are frozen from "
    "`results/metrics/winners.json`. This app adds plain-English explanations for each endpoint."
)

with st.sidebar:
    st.header("Dataset / Task")
    dataset = st.selectbox(
        "Choose a dataset",
        options=list(CLASS_DATASETS + REGR_DATASETS),
        index=0,
        help=(
            "Tox21 = 12 toxicity assays • BBBP = BBB permeability • ClinTox = clinical toxicity/FDA-approved • "
            "ESOL = solubility (logS) • Lipophilicity = logD7.4"
        ),
    )
    tasks = TASKS.get(dataset, [])
    if tasks:
        st.write("**Endpoints**")
        st.write(", ".join(tasks))

st.subheader("SMILES input")
smiles_text = st.text_area(
    "Enter one SMILES per line",
    height=180,
    placeholder="CC(=O)Oc1ccccc1C(=O)O\nCCO",
    label_visibility="collapsed",
)

if st.button("Predict", type="primary"):
    smiles_list = parse_smiles_block(smiles_text)
    if not smiles_list:
        st.warning("Please paste at least one SMILES string.")
        st.stop()

    try:
        with st.spinner("Running inference..."):
            df_raw = run_model(dataset, smiles_list)

        thresholds = load_thresholds(dataset)
        df_words = add_narratives(dataset, df_raw, thresholds=thresholds)

        st.subheader("Predictions")
        st.dataframe(df_words, use_container_width=True)

        with st.expander("Explain each row in words"):
            for _, r in df_words.iterrows():
                name = r.get("name", "compound")
                st.markdown(f"**{name}**  `{r['smiles']}`\n\n{r['explanation']}\n")

        st.success("Done.")
    except Exception as e:
        st.error(f"Could not run prediction: {e}")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Notes: For classifiers, probabilities are calibrated and thresholded when available. "
    "For regressors (ESOL/Lipophilicity), values are in log units and also mapped to qualitative bands."
)

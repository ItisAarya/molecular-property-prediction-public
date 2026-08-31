# src/deploy/verbalize.py
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# ----- Human-friendly names & short descriptions -----
TASKS: Dict[str, List[str]] = {
    "tox21": [
        "NR-AR","NR-AR-LBD","NR-Aromatase","NR-ER","NR-ER-LBD","NR-AhR",
        "SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53","SR-PR"
    ],
    "bbbp": ["BBB-permeable"],
    "clintox": ["CT_TOX", "FDA_APPROVED"],
    "esol": ["logS"],
    "lipophilicity": ["logD7.4"],
}

DESC: Dict[str, Dict[str, str]] = {
    "tox21": {
        "NR-AR":"Androgen receptor activity.",
        "NR-AR-LBD":"AR ligand-binding domain activity.",
        "NR-Aromatase":"Aromatase inhibition.",
        "NR-ER":"Estrogen receptor activity.",
        "NR-ER-LBD":"ER ligand-binding domain activity.",
        "NR-AhR":"Aryl hydrocarbon receptor activity.",
        "SR-ARE":"Oxidative stress response (ARE).",
        "SR-ATAD5":"Genotoxicity (ATAD5).",
        "SR-HSE":"Heat shock response (HSE).",
        "SR-MMP":"Mitochondrial membrane potential.",
        "SR-p53":"DNA damage response (p53).",
        "SR-PR":"Progesterone receptor activity.",
    },
    "bbbp": {
        "BBB-permeable":"Probability the compound crosses the blood–brain barrier."
    },
    "clintox": {
        "CT_TOX":"Clinical-trial toxicity flag (1 = toxic).",
        "FDA_APPROVED":"Belongs to the set of FDA-approved drugs (1 = yes).",
    },
    "esol": {
        "logS":"Aqueous solubility in log10(mol/L)."
    },
    "lipophilicity": {
        "logD7.4":"Experimental logD at pH 7.4 (lipophilicity)."
    },
}

# ----- Utility: map calibrated probability to words -----
def prob_to_words(p: float, thr: float = 0.5) -> str:
    """
    Turn a probability and a decision threshold into a short phrase.
    Works for calibrated probabilities. Uses margin around the threshold.
    """
    margin = p - thr
    # confidence proxy: farther from 0.5 -> higher confidence
    conf = 1.0 - 2.0 * min(p, 1 - p)  # 0..1
    conf_word = "high" if conf >= 0.6 else ("moderate" if conf >= 0.3 else "low")

    if p >= max(thr, 0.8):
        lik = "very likely"
    elif p >= max(thr, 0.65):
        lik = "likely"
    elif abs(margin) <= 0.05:
        lik = "borderline"
    elif p >= 0.4:
        lik = "somewhat unlikely"
    else:
        lik = "very unlikely"

    return f"{lik} (p={p:.2f}, thr={thr:.2f}, confidence {conf_word})"

# ----- Utility: map regression values to qualitative bands -----
def solubility_band(logS: float) -> str:
    # Source: common medicinal chemistry ranges
    if logS > 0:
        level = "very soluble"
    elif logS > -2:
        level = "soluble"
    elif logS > -4:
        level = "moderately soluble"
    elif logS > -5:
        level = "poorly soluble"
    else:
        level = "practically insoluble"
    return f"{level} (logS={logS:.2f} log10 mol/L)"

def lipophilicity_band(logD: float) -> str:
    # Simple, pragmatic bands for logD7.4
    if logD > 3.5:
        level = "very high lipophilicity"
    elif logD > 3.0:
        level = "high lipophilicity"
    elif logD > 1.0:
        level = "moderate lipophilicity"
    else:
        level = "low lipophilicity (more hydrophilic)"
    return f"{level} (logD7.4={logD:.2f})"

# ----- Public API: attach narratives to a prediction dataframe -----
def add_narratives(
    dataset: str,
    df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    name_lookup: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Adds columns:
      - 'property' (human task/dataset name)
      - 'explanation' (plain-English sentence)
      - 'name' (optional molecule common name if provided)
    The function expects the same df structure produced by the app:
      * Classification: columns like task1_prob, task1_label, ...
      * Regression: a 'prob' or value column; label may not exist.
    """
    out = df.copy()

    if dataset in ("esol", "lipophilicity"):
        # Regression path: one value per row
        prop = TASKS[dataset][0]
        out["property"] = prop
        if dataset == "esol":
            out["explanation"] = out["prob"].apply(solubility_band)
        else:
            out["explanation"] = out["prob"].apply(lipophilicity_band)
    else:
        # Classification path: one or multiple tasks -> find *_prob columns in order
        task_names = TASKS[dataset]
        prob_cols = [c for c in out.columns if c.endswith("_prob")]
        prob_cols.sort()  # task1_prob, task2_prob, ...
        if thresholds is None:
            thresholds = [0.5] * len(prob_cols)
        # Build a multi-line explanation per row
        def _row_to_text(row) -> str:
            lines = []
            for i, col in enumerate(prob_cols):
                p = float(row[col])
                thr = float(thresholds[i]) if i < len(thresholds) else 0.5
                name = task_names[i] if i < len(task_names) else f"task{i+1}"
                desc = DESC.get(dataset, {}).get(name, "")
                phrase = prob_to_words(p, thr)
                lines.append(f"**{name}**: {phrase}. {desc}")
            return "\n".join(f"- {ln}" for ln in lines)

        out["property"] = ", ".join(task_names[: min(3, len(task_names))]) + ("…" if len(task_names) > 3 else "")
        out["explanation"] = out.apply(_row_to_text, axis=1)

    # Optional: attach common names if a lookup dict is given {smiles: name}
    if name_lookup and "smiles" in out.columns:
        out["name"] = out["smiles"].map(name_lookup).fillna("—")

    return out

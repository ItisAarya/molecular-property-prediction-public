"""
app/predict_app.py

Type a molecule in, get every property this project can predict, with its uncertainty.

    python -m streamlit run app/predict_app.py

WHAT THIS SERVES
----------------
The multi-view fusion model (`proposed`: cross-attention -> bilinear -> gate over a graph
view, a ChemBERTa view and a descriptor view), one model per dataset, all eight trained on
the canonical DeepChem scaffold split under the single fixed hyper-parameter setting in
`configs/shared.yaml`.

This is a different app from `app/streamlit_app.py`, which serves the inherited Phase 0
pipeline on five datasets. Both are kept: the old one is the baseline this project was
built on, and overwriting it would erase the thing the new numbers are measured against.

WHAT IT DELIBERATELY SHOWS
--------------------------
Every prediction is displayed next to two things the model cannot choose for itself: the
conformal prediction set or interval, calibrated on a validation split the model never
trained on, and the model's own measured test performance on that dataset. A demo that
shows a probability alone invites the viewer to believe it. The numbers here are honest,
which means several of them are unimpressive, and the app says so rather than hiding it.
"""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # every path in src/deploy is relative to the project root

from src.deploy.lookup import find  # noqa: E402
from src.deploy.predict import DESCRIPTIONS, Predictor, datasets  # noqa: E402
from src.eval.metrics import is_classification  # noqa: E402

st.set_page_config(page_title="Molecular property prediction", page_icon="*", layout="wide")

# Molecules a viewer will recognise, so a demo does not depend on typing SMILES correctly.
#
# The first group is the point of the demo. Each was screened against all eight datasets and
# is novel at three levels: the molecule is in none of the pools, its Bemis-Murcko scaffold is
# in none of the training splits, and its nearest-neighbour Tanimoto to the training set is
# low (shown in brackets). The 0.26-0.35 cases sit in the 0.0-0.3 band where section 6.6
# measures active coverage collapsing to 52.5% against a nominal 90% -- so these are the
# hardest cases this project measures, which is the honest thing to demonstrate on.
#
# The second group is famous and therefore mostly *in* the training data. They are kept
# because the contrast is the lesson: the app labels them a memory test, not a prediction.
EXAMPLES = {
    "NEW · Apixaban, blood thinner [0.26]": "COc1ccc(-n2nc(C(N)=O)c3c2CCCC3)cc1-n1nc(-c2ccc(N3CCCCC3=O)cc2)cc1",
    "NEW · Imidacloprid, insecticide [0.28]": "Clc1ccc(CN2CCN/C2=N\\[N+](=O)[O-])nc1",
    "NEW · Osimertinib, lung cancer [0.35]": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
    "NEW · Remdesivir, COVID [0.36]": "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",
    "NEW · Nirmatrelvir, Paxlovid [0.44]": "CC1(C)[C@@H]2[C@H]1[C@H](C(=O)N[C@@H](CC1CCNC1=O)C#N)N(C(=O)[C@@H](NC(=O)C(F)(F)F)C(C)(C)C)C2",
    "NEW · Glyphosate, herbicide [0.55]": "OC(=O)CNCP(=O)(O)O",
    "Aspirin (painkiller)": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine (stimulant)": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "Paracetamol / acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen (anti-inflammatory)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    "Penicillin G (antibiotic)": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "Benzene (a known carcinogen)": "c1ccccc1",
    "Ethanol (drinking alcohol)": "CCO",
}

FRIENDLY = {
    "tox21": "Toxicity screen",
    "bbbp": "Brain penetration",
    "clintox": "Clinical trial outcome",
    "bace": "Alzheimer's target (BACE-1)",
    "sider": "Side effects",
    "esol": "Water solubility",
    "lipophilicity": "Fat vs water preference",
    "freesolv": "Hydration energy",
}


@st.cache_resource(show_spinner=False)
def get_predictor(ds):
    """One loaded model per dataset, kept across reruns. Streamlit re-executes this whole
    file on every widget change, so without the cache each keystroke would reload eight
    models and a transformer."""
    return Predictor(ds)


@st.cache_data(show_spinner=False)
def provenance(smiles, ds):
    """Whether this molecule was in the data, and its measured answer if so. Cached
    because building the canonical-SMILES index costs a few seconds per dataset."""
    return find(smiles, ds)


def honesty_banner(prov):
    """
    Say plainly what kind of result the viewer is about to look at.

    This is the most important thing on the page. Almost every molecule a person thinks to
    type is in the training set of most of these datasets, and a prediction about a
    molecule the model learned from is not a prediction.
    """
    if not prov["found"]:
        st.info("**A real prediction.** This molecule is not in this dataset, so the model "
                "has never seen it. There is no answer key to check it against.")
        return
    if prov["split"] == "train":
        st.warning("**Not a prediction — the model learned from this molecule.** It is in "
                   "this dataset's training set, so a good answer here shows memory, not "
                   "ability. Try a molecule marked *held out* to see real performance.")
    else:
        st.success(f"**A real prediction, and the answer is known.** This molecule was held "
                   f"out in the *{prov['split']}* set — the model never learned from it — "
                   f"and the measured answer is shown below for comparison.")

    if prov.get("conflict"):
        st.error(f"**This dataset contains this molecule {prov['n_copies']} times, and the "
                 f"copies disagree with each other.** The measured answer shown below is "
                 f"the first of them. No model can be right about both — this is a limit "
                 f"of the benchmark, not of the model.")
    elif prov.get("n_copies", 1) > 1:
        st.caption(f"This molecule appears {prov['n_copies']} times in this dataset. The "
                   f"copies agree.")


def truth_text(pred, prov, task_index):
    """The measured answer for one task, formatted for display, or None."""
    if not prov["found"]:
        return None
    y = prov["y"]
    if task_index >= len(y) or not np.isfinite(y[task_index]):
        return None
    if pred.cls:
        return "active" if y[task_index] >= 0.5 else "inactive"
    return f"{y[task_index]:.2f}"


@st.cache_data(show_spinner=False)
def draw(smiles):
    """A picture of the molecule. Returns None rather than raising on an odd structure."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        return Draw.MolToImage(mol, size=(320, 240)) if mol else None
    except Exception:
        return None


def available():
    """Datasets whose deployment checkpoint actually exists, so a partly-trained tree
    still runs instead of erroring on the first missing file."""
    from src.deploy.predict import DEFAULT_TAG
    return [d for d in datasets()
            if os.path.exists(os.path.join("models", f"{d}_{DEFAULT_TAG}.pt"))]


def performance_caption(pred):
    """The model's measured test score, so a prediction is never shown bare."""
    m = pred.metrics
    if not m:
        return "Measured accuracy for this dataset is not archived."
    if pred.cls:
        return (f"This model scores **AUC {m['auc']:.3f}** on the held-out test set "
                f"(0.5 = guessing, 1.0 = perfect).")
    return (f"This model is off by **{m['rmse']:.2f}** on average on the held-out test set "
            f"(root-mean-square error, same units as the prediction).")


def show_regression(pred, task, truth):
    value, unit = task["value"], task["unit"]
    delta = None if truth is None else f"{value - float(truth):+.2f} vs measured"
    st.metric(FRIENDLY.get(pred.ds, pred.ds), f"{value:.2f}", delta=delta,
              delta_color="off", help=DESCRIPTIONS[pred.ds])
    st.caption(f"Units: {unit}")
    if truth is not None:
        st.write(f"**Measured answer: {truth} {unit}**")
    if "low" in task:
        inside = truth is not None and task["low"] <= float(truth) <= task["high"]
        st.write(f"**90% confidence range:** {task['low']:.2f} to {task['high']:.2f} {unit}")
        if truth is not None:
            st.write("The measured value **is** inside that range." if inside
                     else "The measured value is **outside** that range.")
        st.caption("Calibrated so that, across many molecules, the true value falls inside "
                   "this range about 90% of the time. Any single molecule can fall outside "
                   "it — that is what 90% means.")
    else:
        st.caption("No calibrated range available for this model.")


def show_classification(pred, tasks, prov):
    """Probabilities plus conformal sets. Summarised first, because SIDER has 27 tasks and
    a wall of 27 rows tells a viewer nothing."""
    flagged = [t for t in tasks if t["probability"] >= 0.5]
    st.metric(FRIENDLY.get(pred.ds, pred.ds),
              f"{len(flagged)} of {len(tasks)} flagged",
              help=DESCRIPTIONS[pred.ds])

    undecided = [t for t in tasks if len(t.get("set", [])) == 2]
    if undecided:
        st.caption(f"The model cannot decide on {len(undecided)} of {len(tasks)} "
                   f"(its 90% answer includes both possibilities).")

    rows, correct, scored = [], 0, 0
    for i, t in enumerate(tasks):
        s = t.get("set")
        if s is None:
            verdict = "-"
        elif len(s) == 2:
            verdict = "could be either"
        elif len(s) == 1:
            verdict = s[0]
        else:
            verdict = "neither (low confidence)"
        row = {"Target": t["task"], "Probability active": round(t["probability"], 3),
               "Best guess": t["label"], "90% answer": verdict}
        truth = truth_text(pred, prov, i)
        if truth is not None:
            row["Measured"] = truth
            scored += 1
            correct += int(truth == t["label"])
        rows.append(row)

    if scored and prov.get("conflict"):
        # Scoring against one of two contradictory measurements would dress a coin-flip up
        # as an accuracy. The table still shows the first copy's label, captioned above.
        st.caption("Not scored: this dataset disagrees with itself about this molecule.")
    elif scored:
        st.caption(f"Where the answer is known, the best guess is right on "
                   f"**{correct} of {scored}**.")

    rows.sort(key=lambda r: -r["Probability active"])
    with st.expander(f"All {len(rows)} results", expanded=len(rows) <= 3):
        st.dataframe(rows, use_container_width=True, hide_index=True)


st.title("Molecular property prediction")
st.write("Type a molecule as a **SMILES** string — a text way of writing a chemical "
         "structure — and the trained models predict eight kinds of property, each with "
         "an honest measure of how much to trust it.")

ready = available()
if not ready:
    st.error("No deployment checkpoints found in `models/`. Train them first:\n\n"
             "```\npython -m src.train.train_fusion --mode proposed "
             "--tag deploy_proposed --seq cached --device cpu\n```")
    st.stop()

with st.sidebar:
    st.header("Molecule")
    choice = st.selectbox("Pick an example", ["(type my own)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "CC(=O)Oc1ccccc1C(=O)O")
    smiles = st.text_input("SMILES", value=default)

    st.header("Properties")
    picked = st.multiselect("Which to predict", ready, default=ready,
                            format_func=lambda d: FRIENDLY.get(d, d))
    go = st.button("Predict", type="primary", use_container_width=True)

    st.divider()
    st.caption("Model: multi-view fusion (graph + ChemBERTa + descriptors), trained on the "
               "DeepChem scaffold split. One fixed hyper-parameter setting for every "
               "model — nothing here is tuned per dataset.")

if not smiles.strip():
    st.info("Enter a SMILES string in the sidebar to begin.")
    st.stop()

col_img, col_txt = st.columns([1, 2])
with col_img:
    img = draw(smiles)
    if img is not None:
        st.image(img, caption=smiles)
    else:
        st.error("RDKit could not read that as a molecule. Check the SMILES string.")
with col_txt:
    st.subheader("What the model is being asked")
    st.write("Each property below was learned from a different public dataset of measured "
             "molecules, and each has its own separately trained model.")
    st.write("Whether this particular molecule was part of a dataset's training data "
             "differs from dataset to dataset, so **each result below says which case it "
             "is**. A prediction about a molecule the model learned from is a memory test, "
             "not a prediction, and the app labels it as such.")

if not go:
    st.info("Press **Predict** in the sidebar.")
    st.stop()
if img is None:
    st.stop()

st.divider()
for ds in picked:
    with st.spinner(f"Loading the {FRIENDLY.get(ds, ds)} model…"):
        pred = get_predictor(ds)
    rows = pred.predict([smiles])
    row = rows[0]

    st.subheader(FRIENDLY.get(ds, ds))
    st.caption(DESCRIPTIONS[ds])
    if not row["ok"]:
        st.error(row["error"])
        continue

    prov = provenance(smiles, ds)
    honesty_banner(prov)

    left, right = st.columns([1, 2])
    with left:
        if is_classification(ds):
            show_classification(pred, row["tasks"], prov)
        else:
            show_regression(pred, row["tasks"][0], truth_text(pred, prov, 0))
    with right:
        st.write(performance_caption(pred))
        if is_classification(ds):
            st.caption("A '90% answer' of *could be either* is a real result, not a "
                       "failure: it means the evidence does not separate the two "
                       "possibilities at that confidence level. On datasets where actives "
                       "are rare, read section 6.2 of the paper before trusting these — "
                       "the guarantee holds on average while the rare class is covered "
                       "far less often than the headline number suggests.")
    st.divider()

st.caption("Predictions are research output, not advice. Nothing here is validated for "
           "any clinical, safety or regulatory use.")

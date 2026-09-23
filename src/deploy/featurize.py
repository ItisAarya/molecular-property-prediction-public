"""
src/deploy/featurize.py

Turn a raw SMILES string into exactly the three views the trained models were fed.

    from src.deploy.featurize import featurize
    views, ok = featurize(["CC(=O)Oc1ccccc1C(=O)O"])   # aspirin

WHY THIS FILE IS DELICATE
------------------------
Training read its features from `data/pool/`, built once by the prep pipeline. Inference
has to rebuild the same numbers from a string the user just typed, and nothing checks that
the two agree -- a model handed subtly different features does not crash, it quietly
returns a confident wrong answer. That is the worst failure mode a demo can have.

So every featuriser here is the *same code path* the pool was built from, not a
reimplementation:

  ECFP    1024-bit Morgan, radius 2. Verified bit-for-bit against `data/pool/*_ecfp.npz`
          by `scripts/check_deploy.py`, which is the only reason this is written out
          rather than imported -- the pool was built through DeepChem's featuriser, and
          pinning DeepChem into a Streamlit process costs a heavy import for a function
          that is four lines of RDKit.
  desc    `scripts.make_descriptors.descriptor_functions` / `compute_one`, imported
          directly. 217 RDKit 2-D descriptors, NaN where undefined.
  graph   `scripts.make_graphs.mol_to_graph`, imported directly.
  seq     Frozen ChemBERTa, `concatenate([cls, mean])` -> 1536-d, mirroring
          `scripts/cache_embeddings.py`. Pass `dataset=` so ClinTox and BBBP inputs are
          canonicalised exactly as their training strings were (src/data/smiles.py). The masked mean matters: without the mask, a
          short molecule's embedding is dragged toward the padding vector.

`scripts/check_deploy.py` re-derives all four for real pool molecules and asserts they
match what training actually used. Run it after touching anything here.

THE MODEL IS NEVER LOADED TWICE
-------------------------------
ChemBERTa is ~350 MB and takes seconds to load. `_encoder()` caches it at module level,
so a Streamlit rerun -- which re-executes the whole script on every widget change --
pays for it once per process.
"""

import numpy as np
import torch

from scripts.make_descriptors import compute_one, descriptor_functions
from scripts.make_graphs import mol_to_graph

# Must match scripts/tokenize_smiles.py and scripts/cache_embeddings.py exactly.
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MAX_LEN = 128
N_ECFP = 1024
ECFP_RADIUS = 2

_DESC_FUNCS = None
_TOK = None
_TRF = None


def _funcs():
    global _DESC_FUNCS
    if _DESC_FUNCS is None:
        _DESC_FUNCS = descriptor_functions()
    return _DESC_FUNCS


def _encoder():
    """The frozen ChemBERTa, loaded once per process."""
    global _TOK, _TRF
    if _TRF is None:
        from transformers import AutoModel, AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL_NAME)
        _TRF = AutoModel.from_pretrained(MODEL_NAME)
        _TRF.eval()
    return _TOK, _TRF


def parse(smiles):
    """
    RDKit molecule or None.

    Sanitisation is left to RDKit's default, which is what the prep pipeline used. A
    string that fails here is not a molecule this project can say anything about, and the
    caller is expected to tell the user so rather than substitute a zero vector.
    """
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles)


def ecfp(mols):
    """1024-bit Morgan fingerprints, radius 2 -- bit-identical to the pooled features."""
    from rdkit.Chem import AllChem
    out = np.zeros((len(mols), N_ECFP), dtype=np.float32)
    for i, m in enumerate(mols):
        if m is None:
            continue
        bits = AllChem.GetMorganFingerprintAsBitVect(m, ECFP_RADIUS, nBits=N_ECFP)
        out[i, list(bits.GetOnBits())] = 1.0
    return out


def descriptors(mols):
    """217 RDKit 2-D descriptors. NaN where undefined -- the encoder imputes, not us."""
    funcs = _funcs()
    out = np.full((len(mols), len(funcs)), np.nan, dtype=np.float64)
    for i, m in enumerate(mols):
        if m is not None:
            out[i] = compute_one(m, funcs)
    return out


@torch.no_grad()
def chemberta(smiles, dataset=None):
    """
    Frozen ChemBERTa embeddings: [CLS] and masked mean, concatenated to 1536-d.

    Padding length cannot affect the result -- [CLS] is position 0, and the mean is taken
    over real tokens only -- so encoding one molecule gives the same vector as encoding it
    inside the full split, which is what makes this comparable to the cached features.
    """
    tok, trf = _encoder()
    # A model trained on canonical strings must be shown canonical strings: see
    # src/data/smiles.py for which datasets and why.
    from src.data.smiles import sequence_input
    smiles = sequence_input(smiles, dataset) if dataset else list(smiles)
    enc = tok(list(smiles), padding=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    hidden = trf(input_ids=enc["input_ids"],
                 attention_mask=enc["attention_mask"]).last_hidden_state

    cls = hidden[:, 0, :]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.cat([cls, mean], dim=1).float()


def featurize(smiles, views=("graph", "seq", "desc"), dataset=None):
    """
    Build the view dict the fusion models expect, for a list of SMILES strings.

    Returns `(views_dict, ok)` where `ok` is a boolean array marking which inputs were
    parsable. Unparsable rows are dropped from the views entirely rather than zero-filled,
    because a zero vector is a perfectly valid input that means "a molecule with no
    features" -- the model will score it, and the score will be meaningless.

    A molecule RDKit parses but `mol_to_graph` rejects (no atoms, or over the atom cap)
    is also dropped, so the graph view can never be short a row relative to the others.
    """
    mols = [parse(s) for s in smiles]
    graphs = [mol_to_graph(s) if m is not None else None for s, m in zip(smiles, mols)]

    ok = np.array([m is not None and g is not None for m, g in zip(mols, graphs)])
    if not ok.any():
        return {}, ok

    keep = [i for i, good in enumerate(ok) if good]
    kept_smiles = [smiles[i] for i in keep]
    kept_mols = [mols[i] for i in keep]

    out = {}
    if "graph" in views:
        from torch_geometric.data import Batch
        out["graph"] = Batch.from_data_list([graphs[i] for i in keep])
    if "desc" in views:
        out["desc"] = (torch.from_numpy(ecfp(kept_mols)),
                       torch.from_numpy(descriptors(kept_mols).astype(np.float32)))
    if "seq" in views:
        out["seq"] = chemberta(kept_smiles, dataset)
    return out, ok

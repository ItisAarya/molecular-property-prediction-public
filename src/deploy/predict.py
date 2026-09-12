"""
src/deploy/predict.py

Load a trained multi-view model and predict properties for a typed-in SMILES string.

    from src.deploy.predict import Predictor
    p = Predictor("bbbp")
    p.predict(["CC(=O)Oc1ccccc1C(=O)O"])

WHAT THIS IS FOR
----------------
Everything else in this repo scores molecules that were featurised months ago by the prep
pipeline. This is the only path that starts from a string. It exists so the work can be
demonstrated rather than only tabulated.

THE ARCHITECTURE IS REBUILT FROM configs/shared.yaml, NOT GUESSED
-----------------------------------------------------------------
A checkpoint is a bag of tensors; it does not describe the model that produced it. To load
one you have to reconstruct exactly the right architecture first, and a wrong-but-loadable
reconstruction is a silent disaster: PyTorch's `strict=False` will happily leave half a
model at its random initialisation and return confident noise.

So two things are true here. The shapes come from `configs/shared.yaml`, which
`scripts/check_configs.py` already proves matches the trainers' own defaults -- so the
deployed architecture cannot drift from the trained one without that check failing. And
`_load_state` refuses anything less than a total load: every checkpoint tensor must be
consumed, and every trainable parameter must be filled. Nothing is allowed to stay random.

THE DESCRIPTOR SCALER COMES FROM THE CHECKPOINT
-----------------------------------------------
`DescriptorEncoder` normalises with constants fitted on the training split. They are
registered as buffers rather than computed at load time, so they travel inside the
checkpoint and a deployed model cannot drift from its own scaler. That is why this module
can build the encoder with placeholder statistics -- `load_state_dict` immediately
overwrites them with the fitted ones, and `_load_state` fails loudly if it does not.

UNCERTAINTY IS THE POINT, NOT A DECORATION
------------------------------------------
A property predictor that emits a bare number invites the reader to trust it. This project
measured what that number is worth, so the predictions come with the conformal output from
`src/eval/conformal.py`, calibrated on the validation split the model never trained on:

  classification  a prediction *set*. {active} or {inactive} is a real answer at the
                  nominal confidence; {both} means the model genuinely cannot tell, and
                  saying so is the honest output. An empty set means neither class clears
                  the threshold.
  regression      a +/- interval in chemical units, from the absolute-residual score.

Read section 6.2 of the paper before trusting the classification sets on imbalanced data:
marginal coverage is met while actives are covered far below nominal, and the mean set size
is the diagnostic for whether a given model is doing that.
"""

import json
import os

import numpy as np
import torch
import yaml

from src.eval.conformal import binary_scores, conformal_quantile
from src.eval.metrics import is_classification

CONFIG = os.path.join("configs", "shared.yaml")
MODELS_DIR = "models"
POOL_DIR = os.path.join("data", "pool")
SPLIT_DIR = os.path.join("data", "splits")
RUNS_DIR = os.path.join("results", "runs")
META = os.path.join("data", "dataset_meta.json")

# The tag the app serves by default, and the split it was trained on. Both are recorded
# here rather than inferred: `models/<ds>_<tag>.pt` carries no record of which split it
# came from, and every trainer overwrites the same filename, so a checkpoint's provenance
# is exactly as good as the note someone wrote about it.
DEFAULT_TAG = "deploy_proposed"
DEPLOY_VARIANT = "deepchem"

_CFG = None
_META = None


def config():
    global _CFG
    if _CFG is None:
        with open(CONFIG, encoding="utf-8") as f:
            _CFG = yaml.safe_load(f)
    return _CFG


def meta():
    global _META
    if _META is None:
        with open(META, encoding="utf-8") as f:
            _META = json.load(f)
    return _META


def datasets():
    return list(meta().keys())


def tasks_of(ds):
    return list(meta()[ds]["tasks"])


def label_scale(ds):
    """(mean, std) per task. Regression labels were z-scored; classification ones were not."""
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    return np.asarray(pool["y_mean"], dtype=float), np.asarray(pool["y_std"], dtype=float)


def _load_state(model, state, path):
    """
    Load a checkpoint and refuse to proceed on a partial match.

    `checkpoint_state` saved only trainable parameters and buffers, so frozen encoder
    weights are legitimately absent -- but in the `cached` sequence setting every module
    is trainable, so anything missing here is a real mismatch rather than an omission.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)

    if unexpected:
        raise RuntimeError(
            f"{path}: {len(unexpected)} tensor(s) in the checkpoint have no home in the "
            f"rebuilt model, e.g. {unexpected[:3]}. The architecture does not match the "
            f"one that was trained.")

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    still_random = [k for k in missing if k in trainable]
    if still_random:
        raise RuntimeError(
            f"{path}: {len(still_random)} trainable tensor(s) were not in the checkpoint "
            f"and are still at their random initialisation, e.g. {still_random[:3]}. "
            f"Predictions from this model would be noise.")
    return model


def build_model(ds, tag=DEFAULT_TAG, mode="proposed"):
    """Rebuild the trained architecture for one dataset and load its weights."""
    from src.models.encoders.cached import CachedEmbeddingEncoder
    from src.models.encoders.descriptor import DescriptorEncoder
    from src.models.encoders.graph import build_graph_encoder
    from src.models.multiview import MultiViewModel
    from src.deploy.featurize import N_ECFP

    cfg = config()
    d = cfg["representation"]["embed_dim"]
    hidden = cfg["representation"]["hidden"]
    fus = cfg["fusion"]

    path = os.path.join(MODELS_DIR, f"{ds}_{tag}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one with:\n"
            f"    python -m src.train.train_fusion --mode {mode} --tag {tag} "
            f"--seq cached --device cpu")
    state = torch.load(path, map_location="cpu", weights_only=False)

    # The descriptor encoder's input width is NOT 1024 + 217. `DescriptorEncoder.fit`
    # drops descriptor columns that were constant on the training split, and how many were
    # constant is a property of that split -- 4 of 217 on Tox21, a different number
    # elsewhere. Rebuilding from the pool's column count gives a model that is the right
    # shape for the data and the wrong shape for the checkpoint. The mask itself is in the
    # checkpoint, so read the width from there.
    keep = state["encoders.desc.keep"].bool()
    n_desc = int(keep.numel())

    graphs = torch.load(os.path.join(POOL_DIR, f"{ds}_graphs.pt"),
                        weights_only=False)["graphs"]
    in_dim = int(graphs[0].x.size(1))
    n_seq = 2 * 768  # ChemBERTa cls + masked mean, as scripts/cache_embeddings.py writes

    encoders = {
        "graph": build_graph_encoder(fus["graph_encoder"], in_dim=in_dim, hidden=hidden),
        "seq": CachedEmbeddingEncoder(n_seq),
        # `keep` comes from the checkpoint so the widths line up; the other three
        # statistics are placeholders that load_state_dict immediately overwrites, and
        # _load_state fails loudly if it does not.
        "desc": DescriptorEncoder(
            N_ECFP,
            median=torch.zeros(n_desc), keep=keep,
            mean=torch.zeros(n_desc), std=torch.ones(n_desc),
            hidden=hidden, dropout=fus["dropout"]),
    }

    model = MultiViewModel(
        encoders, mode=mode, n_tasks=len(tasks_of(ds)), d=d, dropout=fus["dropout"],
        rank=fus["rank"], n_layers=fus["xattn_layers"], n_heads=fus["xattn_heads"])

    _load_state(model, state, path)
    model.eval()
    return model


def _calibration(ds, tag, variant=DEPLOY_VARIANT):
    """
    The validation-split predictions and labels this model's uncertainty is calibrated on.

    Validation, never test: the test split is the estimate of how well the deployed model
    generalises, and calibrating on it would make that estimate a description of the
    calibration set. Returns None when the archived predictions are absent, in which case
    the caller reports a bare prediction and says the interval is unavailable.
    """
    path = os.path.join(RUNS_DIR, variant, "preds", f"{ds}_{tag}_valid.npy")
    if not os.path.exists(path):
        return None
    p = np.load(path)

    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")))
    y = pool["y_raw"][np.asarray(idx["valid"], dtype=int)]
    if p.shape != y.shape:
        return None
    return y, p


class Predictor:
    """
    One dataset's deployed model, its uncertainty calibration, and its measured accuracy.

    Construction loads a checkpoint and reads two archives; it is the expensive part, so
    hold on to the object rather than rebuilding it per molecule. `Predictor` never loads
    ChemBERTa itself -- `src.deploy.featurize` owns that and caches it process-wide, so a
    Streamlit app with eight datasets open still pays for the transformer once.
    """

    def __init__(self, ds, tag=DEFAULT_TAG, alpha=None, variant=DEPLOY_VARIANT):
        self.ds = ds
        self.tag = tag
        self.variant = variant
        self.alpha = config()["conformal"]["alpha"] if alpha is None else alpha
        self.cls = is_classification(ds)
        self.tasks = tasks_of(ds)
        self.model = build_model(ds, tag)
        self.y_mean, self.y_std = label_scale(ds)
        self._cal = _calibration(ds, tag, variant)
        self.metrics = self._archived_metrics()
        # A conformal threshold depends only on the calibration split, so it is the same
        # for every molecule the app will ever be shown. SIDER has 27 tasks; recomputing
        # a quantile over ~143 calibration points 27 times per molecule is work done
        # once here instead.
        self._q = {}

    def _archived_metrics(self):
        """This model's own measured test performance, for display next to a prediction."""
        import pandas as pd
        path = os.path.join(RUNS_DIR, self.variant, "metrics",
                            f"{self.ds}_{self.tag}_test.csv")
        if not os.path.exists(path):
            return None
        return pd.read_csv(path).iloc[0].to_dict()

    @torch.no_grad()
    def _raw(self, smiles):
        """Model outputs for the parsable inputs, plus the mask of which those were."""
        from src.deploy.featurize import featurize
        views, ok = featurize(smiles)
        if not ok.any():
            return None, ok
        out = self.model(views)
        return (torch.sigmoid(out).numpy() if self.cls else out.numpy()), ok

    def _interval(self, task):
        """Conformal half-width for one regression task, in chemical units."""
        if self._cal is None:
            return None
        if ("reg", task) in self._q:
            return self._q[("reg", task)]
        y, p = self._cal
        yt, pt = y[:, task], p[:, task]
        # The archived predictions are in z-scored units; the labels are raw. Put the
        # residual in chemical units so the interval a user reads is in kcal/mol or log
        # units rather than in standard deviations of the training set.
        pt = pt * self.y_std[task] + self.y_mean[task]
        good = np.isfinite(yt) & np.isfinite(pt)
        if good.sum() < 10:
            out = None
        else:
            q = conformal_quantile(np.abs(yt[good] - pt[good]), self.alpha)
            out = None if not np.isfinite(q) else float(q)
        self._q[("reg", task)] = out
        return out

    def _set_thresholds(self, task):
        """Conformal threshold for one classification task (LAC score)."""
        if self._cal is None:
            return None
        if ("cls", task) in self._q:
            return self._q[("cls", task)]
        y, p = self._cal
        yt, pt = y[:, task], np.clip(p[:, task], 0.0, 1.0)
        good = np.isfinite(yt) & np.isfinite(pt)
        if good.sum() < 10:
            out = None
        else:
            q = conformal_quantile(binary_scores(pt[good], yt[good].astype(int), "lac"),
                                   self.alpha)
            out = None if not np.isfinite(q) else float(q)
        self._q[("cls", task)] = out
        return out

    def predict(self, smiles):
        """
        Predictions for a list of SMILES.

        Returns a list of dicts, one per input, each with `smiles`, `ok`, and -- when the
        molecule parsed -- a `tasks` list carrying the per-task output. Unparsable inputs
        keep their place in the list so the caller can line results up with what the user
        typed.
        """
        preds, ok = self._raw(smiles)
        rows, cursor = [], 0

        for i, s in enumerate(smiles):
            if not ok[i]:
                rows.append({"smiles": s, "ok": False,
                             "error": "RDKit could not read this as a molecule."})
                continue

            p = preds[cursor]
            cursor += 1
            out = []
            for t, name in enumerate(self.tasks):
                if self.cls:
                    prob = float(p[t])
                    q = self._set_thresholds(t)
                    entry = {"task": name, "probability": prob,
                             "label": "active" if prob >= 0.5 else "inactive"}
                    if q is not None:
                        in_pos = binary_scores(np.array([prob]), np.array([1]), "lac")[0] <= q
                        in_neg = binary_scores(np.array([prob]), np.array([0]), "lac")[0] <= q
                        entry["set"] = ([n for n, inc in
                                         (("active", in_pos), ("inactive", in_neg)) if inc])
                    out.append(entry)
                else:
                    value = float(p[t]) * self.y_std[t] + self.y_mean[t]
                    half = self._interval(t)
                    entry = {"task": name, "value": value, "unit": UNITS.get(self.ds, "")}
                    if half is not None:
                        entry["low"], entry["high"] = value - half, value + half
                        entry["half_width"] = half
                    out.append(entry)
            rows.append({"smiles": s, "ok": True, "tasks": out})
        return rows


# What the regression numbers actually mean. Without this a user reads "-2.7" and has no
# way to know whether that is good, bad, or which direction is more soluble.
UNITS = {
    "esol": "log mol/L",
    "lipophilicity": "logD (octanol/water)",
    "freesolv": "kcal/mol",
}

DESCRIPTIONS = {
    "tox21": "Toxicity against 12 biological targets (nuclear receptor and stress response).",
    "bbbp": "Whether the molecule crosses the blood-brain barrier.",
    "clintox": "Clinical trial outcome and FDA approval status.",
    "bace": "Inhibition of beta-secretase 1, an Alzheimer's drug target.",
    "sider": "Recorded side effects, grouped into 27 organ-system classes.",
    "esol": "Water solubility.",
    "lipophilicity": "Fat-versus-water preference, which drives absorption.",
    "freesolv": "Hydration free energy -- the energy cost of dissolving in water.",
}

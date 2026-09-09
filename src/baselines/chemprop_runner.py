"""
src/baselines/chemprop_runner.py

Chemprop (D-MPNN, Yang et al. 2019) as an external baseline, driven through its CLI.

    pip install chemprop
    python -m src.baselines.chemprop_runner --variants deepchem seed0 seed1 seed2 seed3 seed4

WHY A SUBPROCESS AND NOT AN ENCODER
-----------------------------------
AttentiveFP is wrapped as an encoder in `src/models/encoders/graph.py` so it trains through
this project's loop and head, which keeps the architecture the only thing that differs.
Chemprop cannot be treated the same way: it owns its own featurisation (its atom and bond
feature sets are not the 34/7 dimensions in `scripts/make_graphs.py`), its own scaling, its
own ensembling and its own early stopping. Re-implementing it would produce something that
is not Chemprop, and the point of an external baseline is to be the published method.

So it runs as itself, on the same molecules and the same splits, and the comparison is
stated for what it is: two complete methods, each under its own recipe, on identical data.
That asymmetry is deliberate and belongs in the paper's text.

WHAT IS HELD IDENTICAL
----------------------
The split. Chemprop is given `--separate_val_path` and `--separate_test_path` built from
this project's split indices, so it never sees its own scaffold splitter and never touches
test. Its validation set is the same molecules every other model early-stops on.

Metrics are recomputed here from its saved predictions with `src/eval/metrics.py`, not read
from Chemprop's own output, so AUC and RMSE mean exactly what they mean for every other row
in the results table -- NaN-masked, per-task, and in chemical units. Chemprop reports
regression error on its internally scaled target; taking that number at face value would
repeat the Phase 0 bug where regression metrics were in z-scored units.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from src.data.materialize import load_split
from src.eval.metrics import cls_metrics, is_classification, reg_metrics

POOL_DIR = os.path.join("data", "pool")
MET_DIR = os.path.join("results", "metrics")
PRED_DIR = os.path.join("results", "preds")
RUNS_DIR = os.path.join("results", "runs")
SPLITS = ("train", "valid", "test")


def have_chemprop():
    return shutil.which("chemprop_train") is not None or _importable("chemprop")


def _importable(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def write_csv(path, smiles, y, task_names):
    """Chemprop's input format: a smiles column plus one column per task, blank for NaN."""
    df = pd.DataFrame(y, columns=task_names)
    df.insert(0, "smiles", smiles)
    df.to_csv(path, index=False)


def run_one(ds, variant, tag, epochs, workdir):
    """Train Chemprop on one dataset and one split; return the metric row."""
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    smiles, y = pool["smiles"], pool["y"]
    cls = is_classification(ds)
    idx = load_split(ds, variant)
    tasks = [f"t{i}" for i in range(y.shape[1])]

    paths = {}
    for s in SPLITS:
        paths[s] = os.path.join(workdir, f"{ds}_{s}.csv")
        write_csv(paths[s], smiles[idx[s]], y[idx[s]], tasks)

    save_dir = os.path.join(workdir, f"{ds}_ckpt")
    cmd = [
        sys.executable, "-m", "chemprop.train",
        "--data_path", paths["train"],
        "--separate_val_path", paths["valid"],
        "--separate_test_path", paths["test"],
        "--dataset_type", "classification" if cls else "regression",
        "--save_dir", save_dir,
        "--epochs", str(epochs),
        "--quiet",
    ]
    subprocess.run(cmd, check=True)

    rows = {}
    for s in ("valid", "test"):
        out = os.path.join(workdir, f"{ds}_{s}_pred.csv")
        subprocess.run([sys.executable, "-m", "chemprop.predict",
                        "--test_path", paths[s], "--checkpoint_dir", save_dir,
                        "--preds_path", out], check=True)
        p = pd.read_csv(out)[tasks].values.astype(float)
        np.save(os.path.join(PRED_DIR, f"{ds}_{tag}_{s}.npy"), p)
        # Recomputed here, deliberately -- see the module docstring. `y` is the
        # z-scored pool target, exactly what every other model is trained and scored on,
        # and reg_metrics converts both sides back to chemical units.
        m = (cls_metrics(y[idx[s]], p) if cls else reg_metrics(y[idx[s]], p, ds))
        pd.DataFrame([m]).to_csv(
            os.path.join(MET_DIR, f"{ds}_{tag}_{s}.csv"), index=False)
        rows[s] = m
    return rows


def main():
    ap = argparse.ArgumentParser(description="Chemprop D-MPNN external baseline.")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+",
                    default=["deepchem"] + [f"seed{i}" for i in range(5)])
    ap.add_argument("--tag", default="chemprop")
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    if not have_chemprop():
        raise SystemExit(
            "chemprop is not installed in this environment.\n"
            "  pip install chemprop\n"
            "It is a heavy dependency (its own torch/lightning pins), so it is deliberately "
            "not in requirements.txt -- install it only when running this baseline, and "
            "prefer a Colab session or a separate venv so it cannot move the pinned "
            "versions every other result in this repository was produced under."
        )

    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    datasets = args.datasets or list(pool_index)
    os.makedirs(MET_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    for v in args.variants:
        print(f"\n{'=' * 70}\n{v}\n{'=' * 70}")
        with tempfile.TemporaryDirectory() as work:
            for ds in datasets:
                print(f"  {ds} ...", flush=True)
                r = run_one(ds, v, args.tag, args.epochs, work)
                key = "auc" if is_classification(ds) else "rmse"
                print(f"    valid {r['valid'][key]:.4f}   test {r['test'][key]:.4f}")
        dest = os.path.join(RUNS_DIR, v, "metrics")
        os.makedirs(dest, exist_ok=True)
        for ds in datasets:
            for s in ("valid", "test"):
                f = f"{ds}_{args.tag}_{s}.csv"
                src = os.path.join(MET_DIR, f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dest, f))


if __name__ == "__main__":
    main()

"""
src/baselines/chemprop_runner.py

Chemprop (D-MPNN) as an external baseline, driven through its command-line interface.

    pip install chemprop            # 2.x; see the version note below
    python -m src.baselines.chemprop_runner --variants deepchem seed0 seed1 seed2 seed3 seed4

WHY A SUBPROCESS AND NOT AN ENCODER
-----------------------------------
AttentiveFP is wrapped as an encoder in `src/models/encoders/graph.py` so it trains through
this project's loop and head, which keeps the architecture the only thing that differs.
Chemprop cannot be treated the same way: it owns its featurisation (its atom and bond feature
sets are not the 34/7 dimensions in `scripts/make_graphs.py`), its scaling, its ensembling
and its early stopping. Re-implementing it would produce something that is not Chemprop, and
the point of an external baseline is to be the published method.

So it runs as itself, on the same molecules and the same splits, and the comparison is stated
for what it is: two complete methods, each under its own recipe, on identical data. That
asymmetry is deliberate and belongs in the paper's text.

WHICH CHEMPROP, AND WHY IT MATTERS
----------------------------------
This targets **Chemprop 2.x**, which is what `pip install chemprop` gives. That is not a
detail: v1 and v2 have incompatible command lines, and an earlier version of this file was
written against v1's (`python -m chemprop.train --data_path ... --separate_val_path ...`).
None of those flags exist in v2. Concretely, v2:

* uses a `chemprop` console script with `train` / `predict` subcommands, not `-m` modules;
* uses hyphenated flags (`--data-path`, not `--data_path`);
* has **no `--separate-val-path` / `--separate-test-path`**. Splits are passed as a *column*
  in one CSV via `--splits-column`, holding `train` / `val` / `test` per row.

That last change is why this file writes a single CSV per dataset carrying a `split` column
built from this project's split indices, rather than three files. Chemprop therefore never
sees its own scaffold splitter and never touches test during training.

WHAT IS HELD IDENTICAL, AND WHAT IS NOT
---------------------------------------
Identical: the molecules, the split assignment, and the fact that test is scored once.

Three things were read out of Chemprop's own source before trusting them, because each one
would have silently invalidated the comparison:

* `--splits-column` genuinely overrides the splitter. It groups the CSV by that column and
  uses those exact row indices; the `split: RANDOM` that appears in its startup log is an
  unused default. Our scaffold assignment is what it trains on.
* `best.pt` is the **best** checkpoint by validation loss, not the last epoch's, so its
  reported result is selected the same way ours is.
* Early stopping is active with `patience = epochs` when `--patience` is unset, so it runs the
  full budget but still restores the best epoch.

Not identical, unavoidably: featurisation, optimiser schedule, learning-rate warmup and
internal target scaling are Chemprop's own. This is the honest form of an external-baseline
comparison and the paper says so.

Metrics are recomputed here from the saved predictions with `src/eval/metrics.py` rather than
read from Chemprop's output, so AUC and RMSE mean exactly what they mean for every other row
in the results table -- NaN-masked, per task, and in chemical units. Chemprop reports
regression error on its internally scaled target; taking that at face value would repeat the
Phase 0 bug where regression metrics were reported in z-scored units.
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

from src.eval.metrics import cls_metrics, is_classification, reg_metrics

SPLIT_DIR = os.path.join("data", "splits")

POOL_DIR = os.path.join("data", "pool")
MET_DIR = os.path.join("results", "metrics")
PRED_DIR = os.path.join("results", "preds")
RUNS_DIR = os.path.join("results", "runs")
SPLITS = ("train", "valid", "test")
# Chemprop's vocabulary for the split column; ours says "valid", theirs says "val".
SPLIT_NAME = {"train": "train", "valid": "val", "test": "test"}


def load_split(ds, variant):
    """
    This variant's train/valid/test index arrays.

    Deliberately duplicated from `src.data.materialize` rather than imported. That module
    imports torch at module level, and this is the one script in the repository that runs
    in an environment where torch has been **rewritten by the baseline being measured** --
    Chemprop pins its own torch and lightning. Importing a torch-dependent module here would
    make this runner fail for a reason that has nothing to do with Chemprop or with the data.

    The two implementations are asserted equal by `tests` in the module docstring's smoke
    path; the function is four lines and reads the same JSON.
    """
    path = os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} -- run: python -m scripts.make_splits")
    d = json.load(open(path))
    return {s: np.asarray(d[s], dtype=int) for s in SPLITS}


def chemprop_exe():
    """The `chemprop` console script, or None. v2 has no importable CLI module."""
    return shutil.which("chemprop")


def chemprop_version():
    try:
        import chemprop

        return getattr(chemprop, "__version__", "unknown")
    except ImportError:
        return None


def write_dataset_csv(path, smiles, y, split_of_row, tasks):
    """
    One CSV holding every molecule, its targets, and which split it belongs to.

    Missing labels are written as empty cells, which is how Chemprop marks an unmeasured
    (molecule, task) pair. Writing them as zeros is the Tox21 bug this project exists to
    have caught: 24% of that benchmark's canonical test labels are unmeasured.
    """
    df = pd.DataFrame(y, columns=tasks)
    df.insert(0, "smiles", smiles)
    df["split"] = split_of_row
    df.to_csv(path, index=False, na_rep="")


def read_preds(path, smiles_col, tasks):
    """
    Chemprop's prediction CSV, as an (n, len(tasks)) array in task order.

    Chemprop writes the input file back out with the prediction columns appended, naming them
    from the model's stored output columns -- which are the target names we passed in. So the
    reliable route is to look those names up by name, which also keeps multi-task order
    correct: Tox21 has twelve columns and reading them in the wrong order would silently
    scramble twelve tasks against their labels.

    Falls back to position only if the names are absent, and asserts the count either way.
    """
    df = pd.read_csv(path)
    if all(t in df.columns for t in tasks):
        return df[list(tasks)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    cols = [c for c in df.columns if c != smiles_col]
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().any(axis=0)]
    if num.shape[1] < len(tasks):
        raise RuntimeError(
            f"{path}: expected {len(tasks)} prediction column(s), found {num.shape[1]} "
            f"among {list(df.columns)}")
    return num.iloc[:, :len(tasks)].to_numpy(dtype=float)


def run_one(ds, variant, tag, epochs, workdir, accelerator, seed=42):
    """Train Chemprop on one dataset and one split; return the metric rows."""
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    smiles, y = pool["smiles"], pool["y"]
    cls = is_classification(ds)
    idx = load_split(ds, variant)
    tasks = [f"t{i}" for i in range(y.shape[1])]

    split_of_row = np.empty(len(smiles), dtype=object)
    for s in SPLITS:
        split_of_row[idx[s]] = SPLIT_NAME[s]
    if (split_of_row == None).any():  # noqa: E711 -- object array, `is None` won't vectorise
        raise RuntimeError(f"{ds}/{variant}: split indices do not cover every molecule")

    data_csv = os.path.join(workdir, f"{ds}_all.csv")
    write_dataset_csv(data_csv, smiles, y, split_of_row, tasks)

    out_dir = os.path.join(workdir, f"{ds}_out")
    train_cmd = [
        chemprop_exe(), "train",
        "--data-path", data_csv,
        "--task-type", "classification" if cls else "regression",
        "--output-dir", out_dir,
        "--smiles-columns", "smiles",
        "--target-columns", *tasks,
        "--splits-column", "split",
        "--epochs", str(epochs),
        # Chemprop refuses to start unless epochs > warmup_epochs, and its warmup default is
        # 2. That is invisible at the real setting (50 epochs) and fatal at the smoke-test
        # setting, which is exactly where it bit: a 2-epoch check died with "The number of
        # epochs should be higher than the number of epochs during warmup". Capping warmup at
        # one below the epoch count keeps the published default of 2 for any real run and only
        # shrinks it for short checks.
        "--warmup-epochs", str(min(2, max(1, epochs - 1))),
        "--pytorch-seed", str(seed),
        "--data-seed", str(seed),
        "--num-workers", "0",
        "--accelerator", accelerator,
    ]
    subprocess.run(train_cmd, check=True)

    rows = {}
    for s in ("valid", "test"):
        part = os.path.join(workdir, f"{ds}_{s}.csv")
        pd.DataFrame({"smiles": smiles[idx[s]]}).to_csv(part, index=False)
        preds_csv = os.path.join(workdir, f"{ds}_{s}_pred.csv")
        subprocess.run([
            chemprop_exe(), "predict",
            "--test-path", part,
            "--model-path", out_dir,      # a directory: v2 finds the .pt files itself
            "--preds-path", preds_csv,
            "--smiles-columns", "smiles",
            "--num-workers", "0",
            "--accelerator", accelerator,
        ], check=True)

        p = read_preds(preds_csv, "smiles", tasks)
        if p.shape[0] != len(idx[s]):
            raise RuntimeError(
                f"{ds}/{variant}/{s}: got {p.shape[0]} predictions for {len(idx[s])} "
                "molecules -- rows would not line up with labels")
        os.makedirs(PRED_DIR, exist_ok=True)
        np.save(os.path.join(PRED_DIR, f"{ds}_{tag}_{s}.npy"), p)

        # Recomputed here, deliberately. `y` is the z-scored pool target that every other
        # model is trained and scored on, and reg_metrics converts both sides back to
        # chemical units.
        m = cls_metrics(y[idx[s]], p) if cls else reg_metrics(y[idx[s]], p, ds)
        row = dict(m)
        row["device"] = accelerator
        row["seed"] = seed
        os.makedirs(MET_DIR, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            os.path.join(MET_DIR, f"{ds}_{tag}_{s}.csv"), index=False)
        rows[s] = m
    return rows


def archive(variant, datasets, tag):
    n = 0
    for kind, ext, srcd in (("metrics", "csv", MET_DIR), ("preds", "npy", PRED_DIR)):
        dest = os.path.join(RUNS_DIR, variant, kind)
        os.makedirs(dest, exist_ok=True)
        for ds in datasets:
            for s in ("valid", "test"):
                f = f"{ds}_{tag}_{s}.{ext}"
                src = os.path.join(srcd, f)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dest, f))
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Chemprop D-MPNN external baseline (v2 CLI).")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+",
                    default=["deepchem"] + [f"seed{i}" for i in range(5)])
    ap.add_argument("--tag", default="chemprop")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--accelerator", default="auto",
                    choices=["auto", "gpu", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if chemprop_exe() is None:
        raise SystemExit(
            "The `chemprop` command is not on PATH.\n"
            "  pip install chemprop\n\n"
            "Install it in a SEPARATE environment. Chemprop pins its own torch and "
            "lightning versions and will move the ones every other result in this "
            "repository was produced under -- which, per this project's own findings, is "
            "enough on its own to change results. A Colab runtime or a throwaway venv is "
            "the right place; see notebooks/phase4_chemprop_colab.ipynb."
        )

    print(f"chemprop {chemprop_version()} at {chemprop_exe()}")
    pool_index = json.load(open(os.path.join(POOL_DIR, "pool_index.json")))
    datasets = args.datasets or list(pool_index)

    for v in args.variants:
        print(f"\n{'=' * 70}\n{v}\n{'=' * 70}", flush=True)
        with tempfile.TemporaryDirectory() as work:
            for ds in datasets:
                print(f"  {ds} ...", flush=True)
                r = run_one(ds, v, args.tag, args.epochs, work, args.accelerator,
                            args.seed)
                key = "auc" if is_classification(ds) else "rmse"
                print(f"    valid {r['valid'][key]:.4f}   test {r['test'][key]:.4f}",
                      flush=True)
        print(f"  archived {archive(v, datasets, args.tag)} file(s)")


if __name__ == "__main__":
    main()

"""
scripts/train_deploy.py

Train the models the Streamlit app serves, and archive them with their provenance.

    python -m scripts.train_deploy                 # train what is missing, then archive
    python -m scripts.train_deploy --archive-only  # just archive what is already trained

WHY A SEPARATE TAG RATHER THAN REUSING THE LADDER'S CHECKPOINTS
---------------------------------------------------------------
`models/<ds>_<tag>.pt` carries no record of which split produced it, and every trainer
writes the same filename. The ladder was trained across six splits in sequence, so the
checkpoints left on disk afterwards are from whichever split happened to run last -- which
is not written down anywhere and is not the canonical split. Serving those would mean the
app could not answer "what was this model trained on?", and the measured accuracy shown
next to each prediction would belong to a different fit than the one doing the predicting.

So the deployed models get their own tag, `deploy_proposed`, trained on the canonical
DeepChem split and archived under `results/runs/deepchem/` like any other result. That
makes three things line up that otherwise would not: the checkpoint, the validation
predictions its conformal intervals are calibrated on, and the test metrics the app
displays.

The architecture and every hyper-parameter are the trainer's own defaults, which
`scripts/check_configs.py` proves match `configs/shared.yaml`. This script chooses nothing.
"""

import argparse
import os
import shutil
import subprocess
import sys

from src.data.materialize import dataset_names

MET_DIR = os.path.join("results", "metrics")
PRED_DIR = os.path.join("results", "preds")
RUNS_DIR = os.path.join("results", "runs")
MODELS_DIR = "models"
VARIANT = "deepchem"
TAG = "deploy_proposed"
MODE = "proposed"


def missing(datasets):
    return [d for d in datasets
            if not os.path.exists(os.path.join(MODELS_DIR, f"{d}_{TAG}.pt"))]


def archive(datasets):
    """Copy this run's metrics and predictions into the canonical split's archive."""
    n = 0
    for sub, src_dir, ext in (("metrics", MET_DIR, "csv"), ("preds", PRED_DIR, "npy")):
        dest = os.path.join(RUNS_DIR, VARIANT, sub)
        os.makedirs(dest, exist_ok=True)
        for ds in datasets:
            for split in ("valid", "test"):
                for name in (f"{ds}_{TAG}_{split}.{ext}", f"{ds}_{TAG}_{split}_gate.{ext}"):
                    src = os.path.join(src_dir, name)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(dest, name))
                        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--archive-only", action="store_true")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    datasets = args.datasets or dataset_names()

    if not args.archive_only:
        from src.data.materialize import active_variant, materialize
        if active_variant() != VARIANT:
            print(f"materialising the {VARIANT} split...")
            materialize(VARIANT, verbose=False)

        todo = missing(datasets)
        if todo:
            print(f"training {TAG} for: {' '.join(todo)}")
            cmd = [sys.executable, "-u", "-m", "src.train.train_fusion",
                   "--mode", MODE, "--tag", TAG, "--seq", "cached",
                   "--device", args.device, "--datasets", *todo]
            if subprocess.run(cmd).returncode != 0:
                raise SystemExit("training failed; nothing archived.")
        else:
            print(f"all {len(datasets)} checkpoints already exist; skipping training.")

    n = archive(datasets)
    print(f"archived {n} file(s) into {RUNS_DIR}/{VARIANT}/")

    # Remove the working copies, the same way `run_view_multiseed` does and for the same
    # reason: a file in results/metrics/ carries no record of which split produced it, and
    # leaving one there next to a differently-split data/ directory has already caused one
    # wrong comparison in this project. The per-variant archive is authoritative.
    removed = 0
    for ds in datasets:
        for split in ("valid", "test"):
            for path in (os.path.join(MET_DIR, f"{ds}_{TAG}_{split}.csv"),
                         os.path.join(PRED_DIR, f"{ds}_{TAG}_{split}.npy"),
                         os.path.join(PRED_DIR, f"{ds}_{TAG}_{split}_gate.npy")):
                if os.path.exists(path):
                    os.remove(path)
                    removed += 1
    summary = os.path.join(MET_DIR, f"{TAG}_summary.csv")
    if os.path.exists(summary):
        os.remove(summary)
        removed += 1
    print(f"removed {removed} working-copy file(s); {RUNS_DIR}/{VARIANT}/ is authoritative")

    done = [d for d in datasets if d not in missing(datasets)]
    print(f"\n{len(done)}/{len(datasets)} datasets ready to serve.")
    print("Verify the live featuriser still reproduces the training features:")
    print("    python -m scripts.check_deploy")
    print("Then:")
    print("    streamlit run app/predict_app.py")


if __name__ == "__main__":
    main()

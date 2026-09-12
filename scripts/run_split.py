"""
scripts/run_split.py

Run the whole leak-free pipeline against one or more split variants and archive the
results per variant.

    python -m scripts.run_split --variants seed0 seed1 seed2 seed3 seed4
    python -m scripts.run_split --variants deepchem --skip-oof   # quick re-check

For each variant it materialises the data (src/data/materialize.py), runs every pipeline
stage in order, then copies results/metrics and results/preds into
results/runs/<variant>/. Afterwards `data/` is restored to the `deepchem` variant so the
repository is left in a known state.

STAGE ORDER MATTERS
-------------------
    train_ml, train_transformer, train_gnn   base models -> valid/test predictions
    make_oof                                 out-of-fold base predictions over train
    train_hybrid                             meta-learner, fitted on the OOF predictions
    calibration                              calibrators, fitted on cross-fitted OOF
    train_ensemble                           blend weights (prefers calibrated inputs)
    thresholds                               decision thresholds, fitted out-of-fold
    report_with_thresholds, select_winners   final table; winner chosen on validation

Each stage consumes the previous one's output. `make_oof` in particular has to run before
`train_hybrid`, because the meta-learner is fitted on out-of-fold base predictions rather
than on the validation split -- that is the whole point of Phase 0 Part 2.

RUNTIME
-------
Roughly 50 minutes per variant on this machine, dominated by the GNN: it trains once for
the final model plus five more times inside `make_oof`. The transformer is now seconds
because its frozen encoder reads cached embeddings, and the Random Forest is a few minutes.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from src.data.materialize import active_variant, materialize

RESULTS = "results"
RUNS_DIR = os.path.join(RESULTS, "runs")

STAGES = [
    ("base: random forest", "src.train.train_ml"),
    ("base: transformer", "src.train.train_transformer"),
    ("base: gnn", "src.train.train_gnn"),
    ("out-of-fold predictions", "src.train.make_oof"),
    ("meta-learner", "src.train.train_hybrid"),
    ("calibration", "src.eval.calibration"),
    ("ensemble", "src.train.train_ensemble"),
    ("thresholds", "src.eval.thresholds"),
    ("final report", "src.eval.report_with_thresholds"),
    ("select winners", "src.eval.select_winners"),
]

NOISE = (
    "DEPRECATION", "Skipped loading", "No normalization", "WARNING", "ConvergenceWarning",
    "STOP: TOTAL", "Increase the number", "Some weights", "You should probably",
    "not removing", "Failed to featurize", "did not match", "CanonicalRankAtoms",
)


def run_stage(module, log):
    """Run one pipeline stage as a subprocess; return True on success."""
    proc = subprocess.run(
        [sys.executable, "-u", "-m", module],
        capture_output=True, text=True, errors="replace",
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip() and not any(n in line for n in NOISE):
            log.write(line + "\n")
    log.flush()
    return proc.returncode == 0


def archive(variant):
    """Copy this variant's metrics and predictions into results/runs/<variant>/."""
    dest = os.path.join(RUNS_DIR, variant)
    os.makedirs(dest, exist_ok=True)
    for sub in ("metrics", "preds"):
        src = os.path.join(RESULTS, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, sub), dirs_exist_ok=True)
    return dest


def clear_predictions():
    """
    Remove the previous variant's predictions before a new run.

    Without this, a stage that fails or skips a dataset would leave the previous split's
    .npy in place and the next stage would silently consume it -- predictions for one
    partition scored against another partition's labels. Better to fail loudly on a
    missing file.
    """
    d = os.path.join(RESULTS, "preds")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".npy"):
                os.remove(os.path.join(d, f))


def run_variant(variant, stages, log_dir):
    print(f"\n{'=' * 70}\n{variant}\n{'=' * 70}")
    print("  materialising data...")
    materialize(variant, verbose=True)
    assert active_variant() == variant, "materialise did not record the active variant"
    clear_predictions()

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{variant}.log")
    started = time.time()

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"variant: {variant}\n\n")
        for label, module in stages:
            t0 = time.time()
            log.write(f"\n{'-' * 60}\n{label}  ({module})\n{'-' * 60}\n")
            ok = run_stage(module, log)
            mins = (time.time() - t0) / 60
            status = "ok" if ok else "FAILED"
            print(f"    {label:<28} {status:<7} {mins:5.1f} min")
            if not ok:
                print(f"    see {log_path}")
                return False, log_path

    dest = archive(variant)
    print(f"  archived -> {dest}   ({(time.time() - started) / 60:.1f} min total)")
    return True, log_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["seed0", "seed1", "seed2", "seed3", "seed4"])
    ap.add_argument("--skip-oof", action="store_true",
                    help="skip make_oof and the stages that depend on it (base models only)")
    ap.add_argument("--restore", default="deepchem",
                    help="variant to leave in data/ afterwards; 'none' to leave as-is")
    args = ap.parse_args()

    stages = STAGES
    if args.skip_oof:
        stages = [s for s in STAGES if s[1] in
                  ("src.train.train_ml", "src.train.train_transformer", "src.train.train_gnn")]

    log_dir = os.path.join(RESULTS, "logs")
    started = time.time()
    done, failed = [], []

    for variant in args.variants:
        ok, _ = run_variant(variant, stages, log_dir)
        (done if ok else failed).append(variant)

    if args.restore != "none":
        print(f"\nRestoring '{args.restore}' into data/ ...")
        materialize(args.restore, verbose=False)

    print(f"\n{'=' * 70}")
    print(f"completed: {done}")
    if failed:
        print(f"FAILED:    {failed}")
    print(f"active split in data/: {active_variant()}")
    print(f"total: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

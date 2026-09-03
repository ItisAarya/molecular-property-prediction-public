"""
scripts/run_view_multiseed.py

Train single-view encoders across the seeded scaffold splits, so an architecture change can
be tested rather than asserted.

    python -m scripts.run_view_multiseed --tags gin_ref gine

Writes results/runs/<variant>/metrics/<ds>_<tag>_test.csv for every seeded split, which
src/eval/view_stats.py then aggregates into means, intervals and paired tests.

WHY THIS RUN EXISTS
-------------------
On the DeepChem split the edge-aware GINE encoder beat the inherited GIN by +0.028 AUC on
Tox21 and +0.017 on ClinTox, and lost on the other three. Phase 0 measured the 95% CI
half-width for these datasets at 0.015 to 0.024 AUC, so both "wins" sit right at the edge
of what a single split can resolve, and the three losses are well inside it.

That is not a result either way. The change also costs 5x the parameters and roughly 7x the
training time, so "it is probably a bit better" is not good enough to adopt it -- and not
good enough to reject it either. Five paired splits give an interval and a paired test,
which is the difference between an opinion and a finding.

The encoders are trained through the same loop (src/train/train_view.py) with the same
loss, class weighting and early stopping, so the comparison isolates the representation.
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
MET_DIR = os.path.join(RESULTS, "metrics")

ENCODER_OF = {"gin_ref": "gin", "gine": "gine"}

NOISE = ("DEPRECATION", "Skipped loading", "No normalization", "WARNING",
         "Some weights", "You should probably", "not removing")


def run_tag(tag, log):
    """Train one encoder across all datasets for the currently materialised split."""
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "src.train.train_view",
         "--encoder", ENCODER_OF[tag], "--tag", tag],
        capture_output=True, text=True, errors="replace",
    )
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip() and not any(n in line for n in NOISE):
            log.write(line + "\n")
    log.flush()
    return proc.returncode == 0


def archive(variant, tags):
    """
    Copy this variant's single-view metrics into its existing run directory.

    Files are named `<dataset>_<tag>_<split>.csv`, so matching on `_<tag>_` picks up
    exactly this run's outputs and leaves the pipeline's own metrics untouched.
    """
    dest = os.path.join(RUNS_DIR, variant, "metrics")
    os.makedirs(dest, exist_ok=True)
    wanted = tuple(f"_{t}_" for t in tags)

    n = 0
    for f in os.listdir(MET_DIR):
        if f.endswith(".csv") and any(w in f for w in wanted):
            shutil.copy2(os.path.join(MET_DIR, f), os.path.join(dest, f))
            n += 1
    return dest, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["gin_ref", "gine"])
    ap.add_argument("--variants", nargs="+",
                    default=["seed0", "seed1", "seed2", "seed3", "seed4"])
    ap.add_argument("--restore", default="deepchem")
    args = ap.parse_args()

    for t in args.tags:
        if t not in ENCODER_OF:
            raise SystemExit(f"unknown tag {t}; known: {list(ENCODER_OF)}")

    log_dir = os.path.join(RESULTS, "logs")
    os.makedirs(log_dir, exist_ok=True)
    started = time.time()
    failed = []

    for variant in args.variants:
        print(f"\n{'=' * 70}\n{variant}\n{'=' * 70}")
        materialize(variant, verbose=False)
        assert active_variant() == variant

        with open(os.path.join(log_dir, f"views_{variant}.log"), "w", encoding="utf-8") as log:
            for tag in args.tags:
                t0 = time.time()
                ok = run_tag(tag, log)
                print(f"    {tag:<10} {'ok' if ok else 'FAILED':<7} {(time.time() - t0) / 60:5.1f} min")
                if not ok:
                    failed.append(f"{variant}/{tag}")

        dest, n = archive(variant, args.tags)
        print(f"  archived {n} metric files -> {dest}")

    if args.restore != "none":
        materialize(args.restore, verbose=False)
        # The loop leaves the LAST variant's metrics sitting in results/metrics/ while
        # data/ now holds the restored variant -- two different splits, no marker saying
        # so. That already caused one wrong comparison. The archived per-variant copies
        # under results/runs/ are authoritative, so remove the ambiguous working copies.
        removed = 0
        for f in os.listdir(MET_DIR):
            if f.endswith(".csv") and any(f"_{t}_" in f for t in args.tags):
                os.remove(os.path.join(MET_DIR, f))
                removed += 1
        print(f"\nremoved {removed} stale working-copy metric file(s) from {MET_DIR}/")
        print(f"per-split results remain under {RUNS_DIR}/<variant>/metrics/")

    print(f"\n{'=' * 70}")
    if failed:
        print(f"FAILED: {failed}")
    print(f"active split in data/: {active_variant()}")
    print(f"total: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

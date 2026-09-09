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
import json
import os
import shutil
import subprocess
import sys
import time

from src.data.materialize import active_variant, materialize

RESULTS = "results"
RUNS_DIR = os.path.join(RESULTS, "runs")
MET_DIR = os.path.join(RESULTS, "metrics")
PRED_DIR = os.path.join(RESULTS, "preds")

ENCODER_OF = {
    "gin_ref": "gin",        # the inherited 2-layer GIN, as an ablation reference
    "gine": "gine",          # edge-aware graph encoder (Phase 1a)
    "desc": "desc",          # ECFP + RDKit 2-D descriptors (Phase 1b)
    "lora": "lora",          # ChemBERTa + LoRA (Phase 1c)
    "seq_frozen": "seq_frozen",   # frozen ChemBERTa, the matched control for lora
    # External baseline (Phase 4). Registered here rather than in a parallel script so it
    # is measured on the same splits, under the same protocol, as everything it is being
    # compared against.
    "attentivefp": "attentivefp",
}

# Fusion variants run through a different trainer but the same loop, splits and archive
# layout, so they are driven from here rather than from a parallel script.
FUSION_OF = {f"fuse_{m}": m
             for m in ("concat", "gated", "xattn", "bilinear", "proposed")}

# Leave-one-view-out ablations. The gate gives the graph view 0.4-1.1% of the weight on
# every dataset, which suggests it is redundant -- but a low weight only says the gate
# routed little through, not that nothing was lost. Retraining without each view is what
# actually answers that.
FUSION_VIEWS = {
    "fuse_gated_nograph": ("gated", ["seq", "desc"]),
    "fuse_gated_noseq": ("gated", ["graph", "desc"]),
    "fuse_gated_nodesc": ("gated", ["graph", "seq"]),
}
FUSION_OF.update({tag: mode for tag, (mode, _) in FUSION_VIEWS.items()})

# Bilinear-rank sweep (02_ENHANCEMENT_PLAN.md section 7, ablation 4). The rank sets how
# many parameters the second-order term gets: 2*d*r per view against d^2 for a full
# bilinear form, so r=16 is a 16x smaller interaction than r=128. The default everywhere
# else is 64. Registered rather than run -- see PROGRESS.md.
FUSION_RANK = {f"fuse_bilinear_r{r}": ("bilinear", r) for r in (16, 32, 128)}
FUSION_OF.update({tag: mode for tag, (mode, _) in FUSION_RANK.items()})

NOISE = ("DEPRECATION", "Skipped loading", "No normalization", "WARNING",
         "Some weights", "You should probably", "not removing")


def datasets_on_disk():
    with open(os.path.join("data", "dataset_meta.json")) as f:
        return list(json.load(f).keys())


def already_done(variant, tag, datasets):
    """
    True if this (variant, tag) has a complete set of archived metrics.

    Free-tier Colab sessions get interrupted, so a run has to be resumable. The archived
    per-variant metrics under results/runs/ are the authoritative record, so their
    presence -- for every dataset, not just some -- is what "done" means.
    """
    dest = os.path.join(RUNS_DIR, variant, "metrics")
    return all(
        os.path.exists(os.path.join(dest, f"{ds}_{tag}_{sp}.csv"))
        for ds in datasets for sp in ("valid", "test")
    )


def run_tag(tag, log, extra_args=(), seq="cached", out_tag=None):
    """
    Train one view or one fusion variant, across all datasets, for the live split.

    `out_tag` is what results are written under; it differs from `tag` when a suffix is in
    use. The architecture is still looked up by `tag`, so `fuse_proposed_e2e` trains the
    `proposed` mode and archives under a name that cannot collide with the cached run.
    """
    out_tag = out_tag or tag
    if tag in FUSION_OF:
        cmd = [sys.executable, "-u", "-m", "src.train.train_fusion",
               "--mode", FUSION_OF[tag], "--tag", out_tag, "--seq", seq]
        if tag in FUSION_VIEWS:
            cmd += ["--views", *FUSION_VIEWS[tag][1]]
        if tag in FUSION_RANK:
            cmd += ["--rank", str(FUSION_RANK[tag][1])]
        cmd += list(extra_args)
    else:
        cmd = [sys.executable, "-u", "-m", "src.train.train_view",
               "--encoder", ENCODER_OF[tag], "--tag", out_tag, *extra_args]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    output = proc.stdout + proc.stderr
    for line in output.splitlines():
        if line.strip() and not any(n in line for n in NOISE):
            log.write(line + chr(10))
    log.flush()

    if proc.returncode != 0:
        # A failure used to go only to the log file, so the console showed "FAILED" with
        # no reason -- and on a remote runtime the log is the awkward thing to reach.
        # Print the tail of what actually happened.
        print(f"    --- {tag} failed, last 20 lines ---")
        for line in output.strip().splitlines()[-20:]:
            print(f"    | {line}")
        print("    --- end ---")
    return proc.returncode == 0


def archive(variant, tags):
    """
    Copy this variant's single-view metrics into its existing run directory.

    Results are named exactly `<dataset>_<tag>_<split>.csv`, and that full form is what
    gets matched. Matching the looser `_<tag>_` substring instead also swept up the
    comparison tables written by `view_stats` -- `view_compare_gine_vs_gin_ref.csv`
    contains `_gine_` -- so every split archive gained a copy of a cross-split summary
    that does not belong to any one split.
    """
    datasets = datasets_on_disk()
    splits = ("valid", "test")

    dest = os.path.join(RUNS_DIR, variant, "metrics")
    os.makedirs(dest, exist_ok=True)
    wanted = {f"{ds}_{t}_{sp}.csv" for ds in datasets for t in tags for sp in splits}

    n = 0
    for f in os.listdir(MET_DIR):
        if f in wanted:
            shutil.copy2(os.path.join(MET_DIR, f), os.path.join(dest, f))
            n += 1

    # Predictions too, not just metrics.
    #
    # These were metrics-only, which meant per-split predictions for the view and fusion
    # models were never kept -- results/preds/ held only whichever variant happened to run
    # last. Conformal prediction and calibration both work from saved predictions, so
    # Phase 3 could not be applied to any model trained after Phase 0 without retraining
    # everything. The metrics are a summary; the predictions are the evidence.
    pred_dest = os.path.join(RUNS_DIR, variant, "preds")
    os.makedirs(pred_dest, exist_ok=True)
    wanted_p = {f"{ds}_{t}_{sp}.npy" for ds in datasets for t in tags for sp in splits}
    # Fusion gate attributions, where the variant has a gate.
    wanted_p |= {f"{ds}_{t}_{sp}_gate.npy"
                 for ds in datasets for t in tags for sp in splits}
    for f in os.listdir(PRED_DIR):
        if f in wanted_p:
            shutil.copy2(os.path.join(PRED_DIR, f), os.path.join(pred_dest, f))
            n += 1

    return dest, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["gin_ref", "gine"])
    ap.add_argument("--variants", nargs="+",
                    default=["seed0", "seed1", "seed2", "seed3", "seed4"])
    ap.add_argument("--restore", default="deepchem")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="cap training length (default: the trainer's own). Changing this "
                         "changes the protocol, so keep it equal across compared tags.")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--tag-suffix", default="", metavar="S",
                    help="append '_S' to the name results are archived under, while still "
                         "selecting the architecture by the base tag. Use it when the same "
                         "architecture is run in a second setting -- the end-to-end fusion "
                         "runs would otherwise overwrite the cached ones file for file.")
    ap.add_argument("--seq", default="cached", choices=["cached", "lora"],
                    help="fusion tags only: which sequence view to use. cached reuses the "
                         "frozen Phase 0 embeddings and runs on CPU; lora trains adapters "
                         "end to end and needs a GPU.")
    ap.add_argument("--keep-going", action="store_true",
                    help="carry on trying a tag that has already failed once. By default "
                         "a tag that fails is dropped from the remaining splits, because "
                         "the usual cause is an environment problem that will not fix "
                         "itself and repeating it just burns the session.")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="restrict to these datasets (default: all). Mainly for smoke "
                         "testing a new encoder before committing to the full run.")
    ap.add_argument("--artifacts", nargs="+", default=None,
                    help="which views to materialise (default: all). A sequence-only run "
                         "needs just `tok ecfp`, which is what a Colab bundle carries.")
    ap.add_argument("--redo", nargs="+", default=None, metavar="TAG",
                    help="discard archived results for these tags before running. "
                         "Needed when a tag's ARCHITECTURE changed: --resume only checks "
                         "that files exist, not which code wrote them, so it would "
                         "otherwise skip a tag that needs recomputing. Combining --redo "
                         "with --resume clears the stale tags once and still protects "
                         "everything else against a disconnect.")
    ap.add_argument("--resume", action="store_true",
                    help="skip any (variant, tag) whose archived metrics are already "
                         "complete -- for restarting after a Colab disconnect")
    args = ap.parse_args()

    known = list(ENCODER_OF) + list(FUSION_OF)
    for t in args.tags:
        if t not in known:
            raise SystemExit(f"unknown tag {t}; known: {known}")

    log_dir = os.path.join(RESULTS, "logs")
    # The trainer creates these when it runs, but archive() reads results/metrics even if
    # every tag failed before getting that far -- which turned one failure into two.
    for d in (log_dir, MET_DIR, PRED_DIR, RUNS_DIR):
        os.makedirs(d, exist_ok=True)
    started = time.time()
    failed = []
    all_datasets = args.datasets or datasets_on_disk()
    broken = set()

    # Clear stale archives first, so the resume check below sees these tags as unfinished.
    if args.redo:
        unknown = [t for t in args.redo if t not in ENCODER_OF and t not in FUSION_OF]
        if unknown:
            raise SystemExit(f"--redo names unknown tag(s): {unknown}")
        removed = 0
        for variant in args.variants:
            dest = os.path.join(RUNS_DIR, variant, "metrics")
            if not os.path.isdir(dest):
                continue
            for f in os.listdir(dest):
                if any(f"_{t}{suffix}_" in f for t in args.redo):
                    os.remove(os.path.join(dest, f))
                    removed += 1
        print(f"--redo {' '.join(args.redo)}: discarded {removed} archived metric file(s)")

    # Options handed straight to the trainer. Held identical across tags so the
    # comparison isolates the encoder rather than the training budget.
    suffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
    out_tag_of = {t: f"{t}{suffix}" for t in args.tags}

    passthrough = ["--device", args.device]
    if args.datasets:
        passthrough += ["--datasets", *args.datasets]
    for flag, value in (("--epochs", args.epochs), ("--patience", args.patience),
                        ("--batch-size", args.batch_size)):
        if value is not None:
            passthrough += [flag, str(value)]

    for variant in args.variants:
        print(f"\n{'=' * 70}\n{variant}\n{'=' * 70}")
        todo = args.tags
        if args.resume:
            done = [t for t in args.tags if already_done(variant, out_tag_of[t], all_datasets)]
            todo = [t for t in args.tags if t not in done]
            for t in done:
                print(f"    {t:<10} already complete, skipping")
            if not todo:
                continue

        if not args.keep_going:
            for t in list(todo):
                if t in broken:
                    print(f"    {t:<10} skipped: it failed on an earlier split "
                          f"(use --keep-going to retry anyway)")
            todo = [t for t in todo if t not in broken]
            if not todo:
                continue

        materialize(variant, verbose=False, artifacts=args.artifacts)
        assert active_variant() == variant

        mode = "a" if args.resume else "w"
        with open(os.path.join(log_dir, f"views_{variant}.log"), mode, encoding="utf-8") as log:
            for tag in todo:
                t0 = time.time()
                ok = run_tag(tag, log, passthrough, seq=args.seq, out_tag=out_tag_of[tag])
                print(f"    {tag:<10} {'ok' if ok else 'FAILED':<7} {(time.time() - t0) / 60:5.1f} min")
                if not ok:
                    failed.append(f"{variant}/{tag}")
                    broken.add(tag)

        dest, n = archive(variant, [out_tag_of[t] for t in args.tags])
        print(f"  archived {n} metric files -> {dest}")

    if args.restore != "none":
        materialize(args.restore, verbose=False, artifacts=args.artifacts)
        # The loop leaves the LAST variant's metrics sitting in results/metrics/ while
        # data/ now holds the restored variant -- two different splits, no marker saying
        # so. That already caused one wrong comparison. The archived per-variant copies
        # under results/runs/ are authoritative, so remove the ambiguous working copies.
        removed = 0
        for f in os.listdir(MET_DIR):
            if f.endswith(".csv") and any(f"_{out_tag_of[t]}_" in f for t in args.tags):
                os.remove(os.path.join(MET_DIR, f))
                removed += 1
        print(f"\nremoved {removed} stale working-copy metric file(s) from {MET_DIR}/")
        print(f"per-split results remain under {RUNS_DIR}/<variant>/metrics/")

    print(f"\n{'=' * 70}")
    if failed:
        print(f"FAILED: {failed}")
        if broken and not args.keep_going:
            print(f"dropped after first failure: {sorted(broken)}")
        print("Fix the cause, then re-run the same command: --resume keeps everything "
              "that already succeeded.")
    print(f"active split in data/: {active_variant()}")
    print(f"total: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()

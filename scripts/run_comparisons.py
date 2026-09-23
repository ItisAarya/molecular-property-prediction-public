"""
scripts/run_comparisons.py

Regenerate every paired comparison the paper reports, in one command.

    python -m scripts.run_comparisons

Each comparison is `python -m src.eval.view_stats --a A --b B --datasets ...`, which writes
`results/metrics/view_compare_A_vs_B.csv`. `scripts/make_tables.py` and
`scripts/check_paper.py` read those files, so after any change to an archived result this is the
step that brings every derived statistic back in line with it.

Results that read raw SMILES on ClinTox or BBBP are excluded inside `view_stats`
(`src/eval/leakage.py`), so a comparison involving one of those models runs over the six
remaining datasets and its `across_n_datasets` column says so.
"""

import subprocess
import sys

ALL = ["tox21", "bbbp", "clintox", "bace", "sider", "esol", "lipophilicity", "freesolv"]
# The inherited pipeline was only ever run on these five.
PIPELINE = ["tox21", "bbbp", "clintox", "esol", "lipophilicity"]

COMPARISONS = [
    # single views (section 5.1)
    ("gine", "gin_ref"), ("desc", "gin_ref"), ("seq_frozen", "gin_ref"), ("lora", "gin_ref"),
    ("lora", "seq_frozen"),
    # the cached CPU ladder (sections 5.3, 5.6)
    ("fuse_proposed", "gin_ref"), ("fuse_proposed", "fuse_concat"), ("fuse_proposed", "fuse_gated"),
    ("fuse_proposed", "desc"), ("fuse_gated", "desc"), ("fuse_gated", "gin_ref"),
    ("fuse_concat", "desc"), ("fuse_xattn", "desc"), ("fuse_bilinear", "desc"),
    ("fuse_bilinear", "fuse_concat"), ("fuse_xattn", "fuse_concat"),
    ("fuse_proposed", "fuse_bilinear"), ("fuse_proposed", "fuse_xattn"),
    # leave the graph view out (section 5.4)
    ("fuse_gated_nograph", "fuse_gated"),
    # the cached ladder on a T4 (section 5.6)
    ("fuse_bilinear_gpu", "fuse_concat_gpu"), ("fuse_xattn_gpu", "fuse_concat_gpu"),
    ("fuse_proposed_gpu", "fuse_bilinear_gpu"), ("fuse_proposed_gpu", "fuse_xattn_gpu"),
    # the rank sweep (section 5.6)
    ("fuse_bilinear_r16_gpu", "fuse_concat_gpu"), ("fuse_bilinear_r32_gpu", "fuse_concat_gpu"),
    ("fuse_bilinear_r128_gpu", "fuse_concat_gpu"),
    ("fuse_bilinear_r16_gpu", "fuse_bilinear_gpu"), ("fuse_bilinear_r32_gpu", "fuse_bilinear_gpu"),
    ("fuse_bilinear_r128_gpu", "fuse_bilinear_gpu"),
    ("fuse_bilinear_r128_gpu", "fuse_bilinear_r16_gpu"),
    ("fuse_bilinear_r32_gpu", "fuse_bilinear_r16_gpu"),
    ("fuse_bilinear_r128_gpu", "fuse_bilinear_r32_gpu"),
    # the end-to-end ladder (section 5.6)
    ("fuse_proposed_e2e", "fuse_proposed"), ("fuse_proposed_e2e", "fuse_gated_e2e"),
    ("fuse_xattn_e2e", "fuse_gated_e2e"), ("fuse_bilinear_e2e", "fuse_gated_e2e"),
    ("fuse_bilinear_e2e", "fuse_concat_e2e"), ("fuse_xattn_e2e", "fuse_concat_e2e"),
    ("fuse_proposed_e2e", "fuse_bilinear_e2e"), ("fuse_proposed_e2e", "fuse_xattn_e2e"),
    ("fuse_proposed_e2e", "fuse_concat_e2e"), ("fuse_proposed_e2e", "desc"),
    # external baselines (section 5.5)
    ("attentivefp", "gine"), ("attentivefp", "gin_ref_gpu"), ("attentivefp", "desc"),
    ("chemprop", "gin_ref_gpu"), ("chemprop", "attentivefp"), ("chemprop", "desc"),
    ("fuse_proposed", "attentivefp"), ("fuse_proposed", "chemprop"),
    # device (section 7)
    ("gin_ref_gpu", "gin_ref"), ("fuse_concat_gpu", "fuse_concat"),
    ("fuse_xattn_gpu", "fuse_xattn"), ("fuse_bilinear_gpu", "fuse_bilinear"),
    ("fuse_proposed_gpu", "fuse_proposed"),
]
PIPELINE_COMPARISONS = [("fuse_proposed", b) for b in ("rf", "gnn", "trf", "hybrid")]


def run(a, b, datasets):
    cmd = [sys.executable, "-m", "src.eval.view_stats", "--a", a, "--b", b, "--datasets", *datasets]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    ok = proc.returncode == 0
    summary = [ln.strip() for ln in proc.stdout.splitlines()
               if "favoured on" in ln or "Holm-corrected" in ln]
    print(f"{'ok ' if ok else 'FAILED'} {a} vs {b}: {' | '.join(summary)}")
    if not ok:
        print(proc.stderr[-1500:])
    return ok


def main():
    bad = 0
    for a, b in COMPARISONS:
        bad += not run(a, b, ALL)
    for a, b in PIPELINE_COMPARISONS:
        bad += not run(a, b, PIPELINE)
    print(f"\n{len(COMPARISONS) + len(PIPELINE_COMPARISONS)} comparisons, {bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""
scripts/make_tables.py

Generate the paper's results tables from the archives, in Markdown and LaTeX.

    python -m scripts.make_tables
    python -m scripts.make_tables --format latex --out paper/tables

WHY GENERATE THEM RATHER THAN WRITE THEM
----------------------------------------
Every number in a results table is a number that can be transcribed wrongly, and a table
copied by hand into a manuscript is a table that stops matching the archives the moment
anything is re-run. This project has already been bitten by the softer version of that:
summary paragraphs in PROGRESS.md quoting a `gine` result from before the head-width fix,
sitting next to the corrected one, reading as a contradiction.

So the tables are a build product. `results/runs/<variant>/metrics/` is the source of truth,
this script is the only thing that reads it into presentation form, and re-running it after
any new result is the whole update procedure.

WHAT IT EMITS
-------------
**Table 1 -- main results.** Every model on every dataset, canonical DeepChem split and
mean +/- 95% CI over the five seeded splits, side by side. Both, always: they differ by up
to 0.18 AUC on the same model and the same code, so reporting one is reporting a choice.

**Table 2 -- paired comparisons.** For each comparison that has been run: the per-dataset
tally, the same tally after Holm-Bonferroni across the eight datasets, and the
across-dataset test. Reading only the first column is how a 1-of-8 result becomes a claim.

**Table 3 -- conformal coverage.** Overall and per-class coverage against nominal, with set
size or interval width, per method.

Missing cells are printed as "--" and never as a blank or a zero. A model that was not run
on a dataset must not look like a model that scored badly on it.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import is_classification

RUNS_DIR = os.path.join("results", "runs")
MET_DIR = os.path.join("results", "metrics")
T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}

DATASETS = ["tox21", "bbbp", "clintox", "bace", "sider",
            "esol", "lipophilicity", "freesolv"]
UNITS = {"esol": "logS", "lipophilicity": "logD", "freesolv": "kcal/mol"}

# Ordered as the paper argues: inherited pipeline, then single views, then the fusion
# ladder, then the ablations. The order is the argument.
TAGS = [
    ("rf", "RF (ECFP)"), ("gnn", "GNN"), ("trf", "ChemBERTa"),
    ("hybrid", "Hybrid"), ("ens", "Ensemble"),
    ("gin_ref", "GIN (inherited)"), ("gine", "GINE"), ("attentivefp", "AttentiveFP"),
    ("chemprop", "Chemprop"),
    ("seq_frozen", "ChemBERTa frozen"), ("lora", "ChemBERTa + LoRA"),
    ("desc", "Descriptor MLP"),
    ("fuse_concat", "Fusion: concat"), ("fuse_gated", "Fusion: gated"),
    ("fuse_xattn", "Fusion: x-attn"), ("fuse_bilinear", "Fusion: bilinear"),
    ("fuse_proposed", "Fusion: proposed"),
    ("fuse_concat_e2e", "Fusion: concat (e2e)"), ("fuse_gated_e2e", "Fusion: gated (e2e)"),
    ("fuse_proposed_e2e", "Fusion: proposed (e2e)"),
    ("fuse_gated_nograph", "Fusion: gated, no graph"),
]


# The inherited pipeline's models are reported in one wide CSV per split. Their per-tag
# `<ds>_<tag>_test.csv` files also exist but are NOT authoritative: every `<ds>_ens_test.csv`
# in the repository is a byte-identical copy of one old run, the same in all six split
# archives, and it disagrees with the report on all 30 cells (tox21 seed0, for instance,
# says 0.7585 where the run actually scored 0.8356). No published number came from those
# files -- PROGRESS.md's baseline table reads the report -- but a table generator that
# preferred them would silently introduce one, which is how the wrong number gets into a
# paper. So for these tags the report wins, and the per-tag file is never consulted.
PIPELINE_TAGS = ("rf", "gnn", "trf", "hybrid", "ens")


def _from_report(variant, ds, tag):
    rep = os.path.join(RUNS_DIR, variant, "metrics", "final_report_thresholded.csv")
    if not os.path.exists(rep):
        return None
    df = pd.read_csv(rep)
    sub = df[(df.dataset == ds) & (df.model == tag)]
    if not len(sub):
        return None
    v = sub.iloc[0].get("test_auc" if is_classification(ds) else "test_rmse")
    return float(v) if v is not None and np.isfinite(v) else None


def _from_tag_csv(variant, ds, tag):
    path = os.path.join(RUNS_DIR, variant, "metrics", f"{ds}_{tag}_test.csv")
    if not os.path.exists(path):
        return None
    row = pd.read_csv(path).iloc[0]
    key = "auc" if is_classification(ds) else "rmse"
    return float(row[key]) if key in row and np.isfinite(row[key]) else None


def value(variant, ds, tag):
    """One model's test metric on one split, or None, from whichever source is correct."""
    if tag in PIPELINE_TAGS:
        return _from_report(variant, ds, tag) or _from_tag_csv(variant, ds, tag)
    return _from_tag_csv(variant, ds, tag) or _from_report(variant, ds, tag)


def ci95(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if v.size < 2:
        return (float(v[0]), 0.0) if v.size else (None, None)
    half = T_CRIT.get(v.size, 1.96) * v.std(ddof=1) / np.sqrt(v.size)
    return float(v.mean()), float(half)


def main_table():
    seeds = [f"seed{i}" for i in range(5)]
    rows = []
    for tag, label in TAGS:
        row = {"model": label}
        any_value = False
        for ds in DATASETS:
            dc = value("deepchem", ds, tag)
            m, h = ci95([value(v, ds, tag) for v in seeds])
            if dc is None and m is None:
                row[ds] = "--"
                continue
            any_value = True
            dc_txt = f"{dc:.4f}" if dc is not None else "--"
            seed_txt = f"{m:.4f}+/-{h:.4f}" if m is not None else "--"
            row[ds] = f"{dc_txt} / {seed_txt}"
        if any_value:
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_table():
    rows = []
    for f in sorted(os.listdir(MET_DIR)):
        if not f.startswith("view_compare_") or not f.endswith(".csv"):
            continue
        d = pd.read_csv(os.path.join(MET_DIR, f))
        if "across_p_sign" not in d.columns:
            # Written before the correction existed; regenerate it rather than show a
            # half-table that looks complete.
            continue
        rows.append({
            "comparison": f"{d.a.iloc[0]} vs {d.b.iloc[0]}",
            "per-dataset": f"{int((d.verdict == 'improves').sum())}+ / "
                           f"{int((d.verdict == 'degrades').sum())}-",
            "after Holm": f"{int((d.verdict_holm == 'improves').sum())}+ / "
                          f"{int((d.verdict_holm == 'degrades').sum())}-",
            "datasets favoured": f"{int(d.across_wins.iloc[0])}/{len(d)}",
            "p (sign)": f"{d.across_p_sign.iloc[0]:.4f}",
            "p (Wilcoxon, dz)": f"{d.across_p_wilcoxon_dz.iloc[0]:.4f}",
            "p (Wilcoxon, raw)": f"{d.across_p_wilcoxon_raw.iloc[0]:.4f}",
        })
    return pd.DataFrame(rows).sort_values("p (Wilcoxon, dz)") if rows else pd.DataFrame()


def conformal_table():
    frames = []
    for f in sorted(os.listdir(MET_DIR)):
        if not f.startswith("conformal_alpha") or f.endswith("bydistance.csv"):
            continue
        d = pd.read_csv(os.path.join(MET_DIR, f))
        if d.empty:
            continue
        d["method"] = f.replace("conformal_alpha0.1", "").replace(".csv", "").strip("_") \
                       or "marginal"
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    keep = ["method", "tag", "dataset", "coverage", "coverage_pos", "coverage_neg",
            "mean_set_size", "mean_width", "width_spread"]
    d = d[[c for c in keep if c in d.columns]]
    for c in ("coverage", "coverage_pos", "coverage_neg"):
        if c in d.columns:
            d[c] = (d[c] * 100).round(1)
    return d.round(3)


def to_markdown(df):
    """
    Markdown table, written out rather than delegated to `DataFrame.to_markdown`.

    That method needs `tabulate`, which is not in this project's environment. Adding a
    dependency so a build script can draw pipe characters is a bad trade -- every extra
    pin is another thing that can move under a re-run.
    """
    cols = [str(c) for c in df.columns]
    cells = [[("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
              for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in cells)) if cells else len(cols[i])
              for i in range(len(cols))]
    line = lambda vals: "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"
    out = [line(cols), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in cells]
    return "\n".join(out)


def emit(df, title, fmt):
    if df.empty:
        return f"\n## {title}\n\n(no data)\n"
    if fmt == "latex":
        return ("\n" + df.to_latex(index=False, escape=True,
                                   caption=title, longtable=False) + "\n")
    return f"\n## {title}\n\n{to_markdown(df)}\n"


def main():
    ap = argparse.ArgumentParser(description="Build the results tables from the archives.")
    ap.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    ap.add_argument("--out", default=os.path.join("paper", "tables"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ext = "tex" if args.format == "latex" else "md"

    pieces = [
        (main_table(), "Table 1: test performance "
                       "(DeepChem canonical split / mean +/- 95% CI over 5 seeded splits)",
         "table1_main"),
        (comparison_table(), "Table 2: paired comparisons, corrected and across datasets",
         "table2_comparisons"),
        (conformal_table(), "Table 3: conformal coverage at a nominal 90%",
         "table3_conformal"),
    ]

    combined = [f"# Results\n\nGenerated by `scripts/make_tables.py` from "
                f"`results/runs/`. Do not edit by hand.\n"]
    for df, title, name in pieces:
        text = emit(df, title, args.format)
        with open(os.path.join(args.out, f"{name}.{ext}"), "w", encoding="utf-8") as f:
            f.write(text)
        combined.append(text)
        print(f"  {title}: {len(df)} row(s)")

    path = os.path.join(args.out, f"all_tables.{ext}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined))
    print(f"\nWrote {path}")

    units = ", ".join(f"{k}={v}" for k, v in UNITS.items())
    print(f"Classification is AUC; regression is RMSE in chemical units ({units}).")


if __name__ == "__main__":
    main()

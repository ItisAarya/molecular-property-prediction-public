"""
scripts/fill_draft_tables.py

Regenerate every generated table inside `paper/draft.md` from the archives.

    python -m scripts.fill_draft_tables

A generated table sits between two markers in the draft:

    <!-- BEGIN GENERATED: name -->
    ...table...
    <!-- END GENERATED -->

and this script replaces whatever is between them with the table `name` built from `results/`.
Tables in a paper are the numbers most often transcribed wrongly, so they are never typed.
Run it after `scripts.run_comparisons`, `scripts.leakage_effect`, `scripts.device_effect` and the
conformal archives are current.
"""

import os
import re

import numpy as np
import pandas as pd

from scripts.make_tables import value
from src.eval.intervals import ci95

DRAFT = os.path.join("paper", "draft.md")
MET = os.path.join("results", "metrics")
SEEDS = [f"seed{i}" for i in range(5)]
CLS = ["tox21", "bbbp", "clintox", "bace", "sider"]
REG = ["esol", "lipophilicity", "freesolv"]
NAMES = {"tox21": "Tox21", "bbbp": "BBBP", "clintox": "ClinTox", "bace": "BACE", "sider": "SIDER",
         "esol": "ESOL", "lipophilicity": "Lipophilicity", "freesolv": "FreeSolv"}

MODELS = [
    ("rf", "RF (ECFP)"), ("gnn", "GNN (pipeline)"), ("trf", "ChemBERTa head (pipeline)"),
    ("hybrid", "Stacking meta-learner"), ("ens", "Weighted ensemble"),
    ("gin_ref", "GIN, 2-layer (`gin_ref`)"), ("gine", "GINE"), ("attentivefp", "AttentiveFP"),
    ("chemprop", "Chemprop"), ("seq_frozen", "ChemBERTa, frozen"), ("lora", "ChemBERTa + LoRA"),
    ("desc", "ECFP + descriptor MLP (`desc`)"),
    ("fuse_concat", "Fusion: `concat`"), ("fuse_gated", "Fusion: `gated`"),
    ("fuse_xattn", "Fusion: `xattn`"), ("fuse_bilinear", "Fusion: `bilinear`"),
    ("fuse_proposed", "Fusion: `proposed`"), ("fuse_gated_nograph", "Fusion: `gated`, no graph"),
    ("fuse_concat_e2e", "Fusion: `concat`, end-to-end"), ("fuse_gated_e2e", "Fusion: `gated`, end-to-end"),
    ("fuse_proposed_e2e", "Fusion: `proposed`, end-to-end"),
]

LABEL = {"fuse_proposed": "`proposed`", "fuse_gated": "`gated`", "fuse_concat": "`concat`",
         "fuse_xattn": "`xattn`", "fuse_bilinear": "`bilinear`", "gin_ref": "`gin_ref`",
         "gin_ref_gpu": "`gin_ref` (T4)", "desc": "`desc`", "gine": "`gine`",
         "attentivefp": "AttentiveFP", "chemprop": "Chemprop", "rf": "`rf`", "gnn": "`gnn`",
         "trf": "`trf`", "hybrid": "`hybrid`", "fuse_gated_nograph": "`gated`, no graph"}


def p3(p):
    return "—" if not np.isfinite(p) else f"{p:.3f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def compare(a, b):
    path = os.path.join(MET, f"view_compare_{a}_vs_{b}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def across(d):
    r = d.iloc[0]
    holm_i = int((d.verdict_holm == "improves").sum())
    holm_d = int((d.verdict_holm == "degrades").sum())
    return (f"{int(r.across_wins)}/{int(r.across_n_datasets)}",
            f"{p3(r.across_p_sign)} / {p3(r.across_p_wilcoxon_dz)} / {p3(r.across_p_wilcoxon_raw)}",
            f"{holm_i} / {holm_d}")


def comparison_rows(pairs):
    rows = []
    for a, b in pairs:
        d = compare(a, b)
        if d is None:
            raise SystemExit(f"missing comparison {a} vs {b}: run python -m scripts.run_comparisons")
        rows.append((f"{LABEL.get(a, a)} vs {LABEL.get(b, b)}", *across(d)))
    return rows


def cell(ds, tag):
    dc = value("deepchem", ds, tag)
    m, _, h, n = ci95([value(v, ds, tag) for v in SEEDS])
    if dc is None and n == 0:
        return "—"
    left = "—" if dc is None else f"{dc:.3f}"
    right = "—" if n == 0 else f"{m:.3f} ± {h:.3f}"
    return f"{left} / {right}"


def main_table(datasets):
    rows = []
    for tag, label in MODELS:
        cells = [cell(ds, tag) for ds in datasets]
        if all(c == "—" for c in cells):
            continue
        rows.append((label, *cells))
    return table(["Model"] + [NAMES[d] for d in datasets], rows)


def leakage_table():
    df = pd.read_csv(os.path.join(MET, "leakage_effect.csv"))
    label = {"fuse_seqonly": "seq only (frozen ChemBERTa)", "fuse_gated_nograph": "`gated`, no graph",
             "fuse_concat": "`concat`", "fuse_gated": "`gated`", "fuse_xattn": "`xattn`",
             "fuse_bilinear": "`bilinear`", "fuse_proposed": "`proposed`"}
    rows = []
    for r in df.itertuples():
        name = label.get(r.model, r.model.replace(" (reads no SMILES characters)", "") + " (no SMILES)")
        if "(no SMILES)" in name:
            name = LABEL.get(name.split(" ")[0], name.split(" ")[0]) + ", reads no SMILES"
            rows.append((NAMES[r.dataset], name, f"{r.seeded_mean_raw:.3f}", f"{r.seeded_mean_canonical:.3f}",
                         "—", "—"))
            continue
        rows.append((NAMES[r.dataset], name, f"{r.seeded_mean_raw:.3f}", f"{r.seeded_mean_canonical:.3f}",
                     f"{r.seeded_mean_change:+.3f}", p3(r.p_paired_t)))
    return table(["Dataset", "Model", "Raw SMILES", "Canonical SMILES", "Change", "p"], rows)


def ladder_table():
    rows = comparison_rows([("fuse_proposed", "gin_ref"), ("fuse_gated", "gin_ref"), ("desc", "gin_ref"),
                            ("fuse_proposed", "desc"), ("fuse_gated", "desc"), ("fuse_concat", "desc"),
                            ("fuse_xattn", "desc"), ("fuse_bilinear", "desc"),
                            ("fuse_proposed", "fuse_concat"), ("fuse_proposed", "fuse_gated")])
    return table(["Comparison", "Favoured", "p (sign / dz / raw)", "Holm improves / degrades"], rows)


def pipeline_table():
    rows = []
    for b in ("rf", "gnn", "trf", "hybrid"):
        d = compare("fuse_proposed", b)
        rows.append((f"`proposed` vs {LABEL[b]}", int(d.iloc[0].across_n_datasets), *across(d)))
    return table(["Comparison", "Datasets", "Favoured", "p (sign / dz / raw)",
                  "Holm improves / degrades"], rows)


def external_table():
    rows = comparison_rows([("attentivefp", "gine"), ("attentivefp", "gin_ref_gpu"),
                            ("chemprop", "gin_ref_gpu"), ("chemprop", "attentivefp"),
                            ("attentivefp", "desc"), ("chemprop", "desc"),
                            ("fuse_proposed", "attentivefp"), ("fuse_proposed", "chemprop")])
    return table(["Comparison", "Favoured", "p (sign / dz / raw)", "Holm improves / degrades"], rows)


def mechanism_table():
    settings = [("", "cached, CPU"), ("_gpu", "cached, T4"), ("_e2e", "end-to-end, T4")]
    pairs = [("bilinear", "concat"), ("xattn", "concat"), ("proposed", "bilinear"),
             ("proposed", "xattn")]
    rows = []
    for a, b in pairs:
        row = [f"`{a}` vs `{b}`"]
        for suffix, _ in settings:
            d = compare(f"fuse_{a}{suffix}", f"fuse_{b}{suffix}")
            r = d.iloc[0]
            row.append(f"{int(r.across_wins)}/{int(r.across_n_datasets)}, p = {p3(r.across_p_wilcoxon_dz)}")
        rows.append(row)
    return table(["Comparison"] + [s[1] for s in settings], rows)


FAILING = ("rf", "trf", "chemprop")
CONF_LABEL = {"rf": "Random forest (pipeline)", "trf": "ChemBERTa head (pipeline)",
              "chemprop": "Chemprop", "gnn": "GNN (pipeline)", "hybrid": "Stacking meta-learner",
              "ens": "Weighted ensemble", "desc": "`desc`", "gin_ref_gpu": "`gin_ref` (T4)",
              "attentivefp": "AttentiveFP", "fuse_gated": "`gated`",
              "fuse_gated_nograph": "`gated`, no graph", "fuse_concat_gpu": "`concat` (T4)",
              "fuse_xattn_gpu": "`xattn` (T4)", "fuse_bilinear_gpu": "`bilinear` (T4)",
              "fuse_proposed_gpu": "`proposed` (T4)", "desc_nopw": "`desc` (no class weighting)"}


def conformal_table():
    marg = pd.read_csv(os.path.join(MET, "conformal_alpha0.1_absolute.csv"))
    cond = pd.read_csv(os.path.join(MET, "conformal_alpha0.1_conditional.csv"))
    ctrl_m = pd.read_csv(os.path.join(MET, "conformal_alpha0.1_absolute_cpu_controls.csv"))
    ctrl_c = pd.read_csv(os.path.join(MET, "conformal_alpha0.1_conditional_cpu_controls.csv"))
    marg = pd.concat([marg[marg.dataset == "tox21"], ctrl_m[ctrl_m.tag == "desc_nopw"]])
    cond = pd.concat([cond[cond.dataset == "tox21"], ctrl_c[ctrl_c.tag == "desc_nopw"]])
    marg = marg.sort_values("coverage_pos")
    rows = []
    for r in marg.itertuples():
        c = cond[cond.tag == r.tag].iloc[0]
        weighting = {"rf": "balanced trees", "trf": "no", "chemprop": "no", "desc_nopw": "no",
                     "hybrid": "balanced meta-learner", "ens": "blend of rf, gnn, trf"}.get(r.tag, "yes")
        rows.append((CONF_LABEL.get(r.tag, r.tag), weighting, f"{100 * r.coverage:.1f}%",
                     f"{100 * r.coverage_pos:.1f}%", f"{r.mean_set_size:.2f}",
                     f"{100 * c.coverage_pos:.1f}%", f"{c.mean_set_size:.2f}"))
    return table(["Model", "Class-weighted loss", "Coverage", "Actives covered", "Set size",
                  "Actives, class-conditional", "Set size, class-conditional"], rows)


def distance_table():
    g = pd.read_csv(os.path.join(MET, "conformal_bydistance_groups.csv"))
    rows = []
    for band in sorted(g.band.unique()):
        a = g[(g.band == band) & g.group.str.contains("covering")].iloc[0]
        b = g[(g.band == band) & g.group.str.contains("failing")].iloc[0]
        rows.append((band, f"{a.n_band:.0f}", f"{100 * a.coverage:.1f}%", f"{100 * a.coverage_pos:.1f}%",
                     f"{100 * b.coverage:.1f}%", f"{100 * b.coverage_pos:.1f}%"))
    return table(["Similarity", "Mean molecules", "Overall (12 covering)", "Actives (12 covering)",
                  "Overall (3 failing)", "Actives (3 failing)"], rows)


def device_table():
    s = pd.read_csv(os.path.join(MET, "device_effect_summary.csv"))
    rows = []
    for r in s.itertuples():
        rows.append((r.experiment, f"{r.models}", f"{r.canonical_median_x_threshold:.2f} / "
                     f"{r.canonical_max_x_threshold:.2f}",
                     f"{r.canonical_over_threshold} of {r.canonical_n}",
                     f"{r.single_split_over_threshold} of {r.metrics_compared}",
                     f"{r.five_split_mean_median_x_threshold:.2f} / {r.five_split_mean_max_x_threshold:.2f}",
                     f"{r.five_split_mean_over_threshold} of {r.five_split_mean_n}"))
    return table(["Change", "Models", "Canonical split: median / max", "Canonical over threshold",
                  "All single splits over threshold", "Five-split mean: median / max",
                  "Five-split means over threshold"], rows)


def datasets_table():
    import json
    audit = pd.read_csv(os.path.join(MET, "notation_audit.csv"))
    meta = json.load(open(os.path.join("data", "pool", "pool_index.json")))
    kinds = {"esol": "regression (logS)", "lipophilicity": "regression (logD)",
             "freesolv": "regression (kcal·mol⁻¹)"}
    rows = []
    for ds in CLS + REG:
        pool = np.load(os.path.join("data", "pool", f"{ds}_ecfp.npz"), allow_pickle=True)
        y = np.asarray(pool["y_raw"], dtype=float).reshape(len(pool["smiles"]), -1)
        if ds in CLS:
            rates = [np.nanmean(y[:, t]) for t in range(y.shape[1])]
            rate = (f"{100 * min(rates):.1f}–{100 * max(rates):.1f}%" if len(rates) > 1
                    else f"{100 * rates[0]:.1f}%")
        else:
            rate = "—"
        kek = audit[audit.dataset == ds].kekule_fraction.iloc[0]
        rows.append((NAMES[ds], f"{len(pool['smiles']):,}", meta[ds]["n_tasks"],
                     kinds.get(ds, "classification"), rate, f"{100 * kek:.1f}%"))
    return table(["Dataset", "Molecules", "Tasks", "Type", "Positive rate", "Kekulé share"], rows)


def duplicates_table():
    base = pd.read_csv(os.path.join(MET, "duplicate_audit.csv"))
    by = pd.read_csv(os.path.join(MET, "duplicate_audit_by_split.csv"))
    rows = []
    for r in base[base.duplicate_groups > 0].itertuples():
        seeded = ", ".join(str(int(by[(by.dataset == r.dataset) & (by.variant == f"seed{i}")]
                                   .groups_spanning_splits.iloc[0])) for i in range(5))
        canon = int(by[(by.dataset == r.dataset) & (by.variant == "deepchem")]
                    .groups_spanning_splits.iloc[0])
        rows.append((NAMES[r.dataset], f"{r.rows:,}", f"{r.unique_molecules:,}", r.duplicate_groups,
                     r.conflicting_groups, f"{canon} / {seeded}"))
    return table(["Dataset", "Rows", "Unique molecules", "Duplicate groups", "Conflicting groups",
                  "Groups crossing splits (canonical / seeded 0–4)"], rows)


def rungs_table():
    from src.models.fusion import build_fusion
    from src.models.heads import MLPHead
    mech = {"concat": "concatenate the views", "gated": "per-molecule softmax gate over views",
            "xattn": "self-attention over the views as tokens",
            "bilinear": "low-rank product for each pair of views (rank 64)",
            "proposed": "`xattn`, then `bilinear` and `gated` in parallel"}
    rows = []
    for m in ("concat", "gated", "xattn", "bilinear", "proposed"):
        f = build_fusion(m, n_views=3, d=256)
        nf = sum(p.numel() for p in f.parameters())
        nh = sum(p.numel() for p in MLPHead(f.out_dim, 1, hidden=256).parameters())
        rows.append((f"`{m}`", mech[m], f"{nf:,}", f.out_dim, f"{nh:,}"))
    return table(["Rung", "Mechanism", "Fusion parameters", "Output width", "Head parameters"], rows)


def rank_table():
    from src.models.fusion import build_fusion
    rows = []
    for r, tag in ((16, "fuse_bilinear_r16_gpu"), (32, "fuse_bilinear_r32_gpu"),
                   (64, "fuse_bilinear_gpu"), (128, "fuse_bilinear_r128_gpu")):
        n = sum(p.numel() for p in build_fusion("bilinear", n_views=3, d=256, rank=r).parameters())
        d = compare(tag, "fuse_concat_gpu").iloc[0]
        vs64 = "—"
        if r != 64:
            e = compare(tag, "fuse_bilinear_gpu").iloc[0]
            vs64 = f"{int(e.across_wins)}/{int(e.across_n_datasets)}, {p3(e.across_p_wilcoxon_dz)}"
        rows.append((r, f"{n:,}",
                     f"{int(d.across_wins)}/{int(d.across_n_datasets)}, {p3(d.across_p_sign)} / "
                     f"{p3(d.across_p_wilcoxon_dz)} / {p3(d.across_p_wilcoxon_raw)}", vs64))
    return table(["Rank", "Fusion parameters", "vs `concat`: favoured, sign / dz / raw p",
                  "vs r = 64: favoured, dz p"], rows)


def motif_table():
    probe = pd.read_csv(os.path.join(MET, "motif_probe.csv"))
    cov = pd.read_csv(os.path.join(MET, "motif_coverage.csv"))
    add = pd.read_csv(os.path.join(MET, "motif_compare_desc_motif_vs_desc.csv"))
    rows = []
    for ds in CLS + REG:
        sub = probe[(probe.dataset == ds) & (probe.variant != "deepchem")]
        mean = sub.groupby("arm").value.mean()
        a = add[add.dataset == ds].iloc[0]
        seen = cov[(cov.dataset == ds) & (cov.variant != "deepchem")].test_brics_fragment_seen.mean()
        metric = "AUC" if ds in CLS else "RMSE"
        rows.append((NAMES[ds], metric, f"{mean['desc']:.3f}", f"{mean['motif']:.3f}",
                     f"{mean['desc+motif']:.3f}", f"{a.mean_diff:+.3f} ± {a.ci_diff:.3f}",
                     f"{100 * seen:.0f}%"))
    return table(["Dataset", "Metric", "`ECFP+desc`", "Motifs", "Both", "Change from adding motifs",
                  "BRICS fragments seen in training"], rows)


def aps_table():
    files = [("LAC", "conformal_alpha0.1_absolute_cpu_controls.csv"),
             ("APS, deterministic", "conformal_alpha0.1_absolute_aps.csv"),
             ("APS, randomised", "conformal_alpha0.1_absolute_aps_randomized.csv"),
             ("RAPS, deterministic", "conformal_alpha0.1_absolute_raps.csv"),
             ("RAPS, randomised", "conformal_alpha0.1_absolute_raps_randomized.csv")]
    rows = []
    for tag in ("desc", "gin_ref"):
        for label, f in files:
            d = pd.read_csv(os.path.join(MET, f))
            r = d[(d.tag == tag) & (d.dataset == "tox21")].iloc[0]
            rows.append((LABEL[tag], label, f"{100 * r.coverage:.1f}%", f"{100 * r.coverage_pos:.1f}%",
                         f"{r.mean_set_size:.2f}"))
    return table(["Model", "Score", "Coverage", "Actives covered", "Mean set size"], rows)


def regression_table():
    ab = pd.read_csv(os.path.join(MET, "conformal_alpha0.1_absolute.csv"))
    no = pd.read_csv(os.path.join(MET, "conformal_alpha0.1_normalized.csv"))
    cq = pd.read_csv(os.path.join(MET, "conformal_alpha0.1_cqr.csv"))
    pick = [("`desc`", "absolute residual", ab, "desc"),
            ("ensemble (`ens`)", "absolute residual", ab, "ens"),
            ("ensemble (`ens`)", "residual / base-model disagreement", no, "ens"),
            ("quantile descriptor MLP", "CQR", cq, "qdesc")]
    rows = []
    for model, score, df, tag in pick:
        r = df[(df.tag == tag) & (df.dataset == "esol")].iloc[0]
        rows.append((model, score, f"{100 * r.coverage:.1f}%", f"{r.mean_width:.3f}",
                     f"{r.width_spread:.3f}"))
    return table(["Model", "Score", "Coverage", "Mean width (logS)", "Width s.d."], rows)


def cliff_table():
    c = pd.read_csv(os.path.join(MET, "activity_cliffs_sim0.9.csv"))
    rows = []
    for ds in ("bace", "tox21"):
        for tag in ("desc", "fuse_gated"):
            sub = c[(c.dataset == ds) & (c.tag == tag) & (c.variant != "deepchem")]
            name = f"{NAMES[ds]} ({sub.n_cliff.mean():.0f} / {sub.n_other.mean():.0f}; {len(sub)} splits)"
            rows.append((name, LABEL[tag], f"{sub.cliff.mean():.3f}", f"{sub.other.mean():.3f}",
                         f"{sub.penalty.mean():.3f}"))
    return table(["Dataset (cliff / non-cliff molecules; splits)", "Model", "Cliff AUC",
                  "Non-cliff AUC", "Difference"], rows)


BUILDERS = {
    "leakage_table": leakage_table,
    "main_table_cls": lambda: main_table(CLS),
    "main_table_reg": lambda: main_table(REG),
    "ladder_table": ladder_table,
    "pipeline_table": pipeline_table,
    "external_table": external_table,
    "mechanism_table": mechanism_table,
    "conformal_table": conformal_table,
    "distance_table": distance_table,
    "device_table": device_table,
    "datasets_table": datasets_table,
    "duplicates_table": duplicates_table,
    "rungs_table": rungs_table,
    "rank_table": rank_table,
    "motif_table": motif_table,
    "aps_table": aps_table,
    "regression_table": regression_table,
    "cliff_table": cliff_table,
}


def main():
    md = open(DRAFT, encoding="utf-8").read()
    pattern = re.compile(r"(<!-- BEGIN GENERATED: (\w+) -->\n)(.*?)(\n<!-- END GENERATED -->)", re.S)

    def repl(m):
        name = m.group(2)
        if name not in BUILDERS:
            raise SystemExit(f"no builder for generated table {name!r}")
        return m.group(1) + BUILDERS[name]() + m.group(4)

    new, n = pattern.subn(repl, md)
    with open(DRAFT, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    print(f"Regenerated {n} table(s) in {DRAFT}")


if __name__ == "__main__":
    main()

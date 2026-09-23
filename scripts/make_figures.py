"""
scripts/make_figures.py

Generate every figure in `paper/draft.md` from the archives.

    python -m scripts.make_figures

Writes `paper/figures/fig<N>_*.pdf` and a matching `.png` for each, numbered in the order the
figures appear in the paper, plus the two archives the figures are drawn from and the prose
quotes: `results/metrics/split_gap.csv` and `results/metrics/conformal_bydistance_groups.csv`.

WHAT EACH FIGURE IS FOR
-----------------------
1  canonical split against the seeded mean, for every (model, classification dataset) pair with
   both -- the two conventions called "scaffold split" are not interchangeable.
2  fusion-block-plus-head parameters against mean AUC for the cached CPU ladder -- capacity and
   accuracy, with the paired statistics left to the tables where they belong.
3  mean prediction-set size against coverage of actives on Tox21, with the class-weighting
   control marked -- the same architecture moves between the two groups.
4  coverage against similarity to the training set, for the two groups of figure 3 separately.

Results whose inputs leaked the label through SMILES notation are excluded throughout
(src/eval/leakage.py).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval.intervals import ci95
from src.eval.leakage import excluded
from src.eval.metrics import is_classification

MET = os.path.join("results", "metrics")
RUNS = os.path.join("results", "runs")
OUT = os.path.join("paper", "figures")
SEEDS = [f"seed{i}" for i in range(5)]
CLS = ["tox21", "bbbp", "clintox", "bace", "sider"]

INK = "#14304F"
ACCENT = "#2A6099"
WARN = "#C1553B"
GOOD = "#2E7D5B"
GREY = "#8A94A0"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#4A5560",
    "figure.dpi": 150, "savefig.bbox": "tight",
})

# The inherited pipeline's per-tag files are not authoritative (scripts/make_tables.py), so
# these tags are read from the per-split report.
PIPELINE_TAGS = ("rf", "gnn", "trf", "hybrid", "ens")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf and .png")


def auc(variant, ds, tag):
    if excluded(ds, tag):
        return None
    if tag in PIPELINE_TAGS:
        rep = os.path.join(RUNS, variant, "metrics", "final_report_thresholded.csv")
        if not os.path.exists(rep):
            return None
        df = pd.read_csv(rep)
        sub = df[(df.dataset == ds) & (df.model == tag)]
        return float(sub.iloc[0]["test_auc"]) if len(sub) else None
    path = os.path.join(RUNS, variant, "metrics", f"{ds}_{tag}_test.csv")
    return float(pd.read_csv(path).iloc[0]["auc"]) if os.path.exists(path) else None


def all_tags():
    tags = set()
    for f in os.listdir(os.path.join(RUNS, "deepchem", "metrics")):
        for ds in CLS:
            if f.startswith(ds + "_") and f.endswith("_test.csv"):
                tags.add(f[len(ds) + 1:-len("_test.csv")])
    return sorted(t for t in tags if "rawsmiles" not in t and "seed43" not in t
                  and t not in ("deploy_proposed", "desc_nopw"))


def fig1_split_gap():
    rows = []
    for tag in all_tags():
        for ds in CLS:
            dc = auc("deepchem", ds, tag)
            seeded = [auc(v, ds, tag) for v in SEEDS]
            if dc is None or any(s is None for s in seeded):
                continue
            rows.append({"model": tag, "dataset": ds, "canonical": dc,
                         "seeded_mean": float(np.mean(seeded)),
                         "gap": float(np.mean(seeded)) - dc})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(MET, "split_gap.csv"), index=False)

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lim = (min(df.canonical.min(), df.seeded_mean.min()) - 0.03,
           max(df.canonical.max(), df.seeded_mean.max()) + 0.03)
    ax.plot(lim, lim, ls="--", lw=1, color=GREY, zorder=1)
    big = df.gap.abs() > 0.02
    ax.scatter(df.canonical[~big], df.seeded_mean[~big], s=22, color=ACCENT, zorder=3,
               label=f"within $\\pm$0.02 AUC ({int((~big).sum())})")
    ax.scatter(df.canonical[big], df.seeded_mean[big], s=22, color=WARN, zorder=3,
               label=f"differ by more than 0.02 AUC ({int(big.sum())})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("canonical DeepChem scaffold split, test AUC")
    ax.set_ylabel("mean of five seeded scaffold splits, test AUC")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig1_canonical_vs_seeded")
    print(f"    {len(df)} pairs; seeded higher on {int((df.gap > 0).sum())}; "
          f"median gap {df.gap.median():+.3f}; max {df.gap.max():.3f}")


def _seed_mean(ds, tag):
    vals = [auc(v, ds, tag) for v in SEEDS]
    vals = [v for v in vals if v is not None]
    return ci95(vals) if vals else (np.nan, np.nan, np.nan, 0)


# Label placement for the rungs that sit close together on the log axis.
LABEL_AT = {"gated": (-7, 6, "right", "bottom"), "bilinear": (0, 8, "center", "bottom"),
            "concat": (7, -7, "left", "top"), "xattn": (0, -8, "center", "top")}


def fig2_params_vs_accuracy():
    from src.models.fusion import build_fusion
    from src.models.heads import MLPHead
    rungs = ["concat", "gated", "xattn", "bilinear", "proposed"]
    rows = []
    for m in rungs:
        fusion = build_fusion(m, n_views=3, d=256)
        n_fusion = sum(p.numel() for p in fusion.parameters())
        n_head = sum(p.numel() for p in MLPHead(fusion.out_dim, 1, hidden=256).parameters())
        per = [_seed_mean(ds, f"fuse_{m}") for ds in CLS]
        vals = [v[0] for v in per if v[3] == 5]
        if len(vals) == len(CLS):
            rows.append((m, n_fusion, n_head, float(np.mean(vals))))
    if not rows:
        return print("  fig2 skipped: no archived ladder metrics")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, n_fusion, n_head, y in rows:
        x = n_fusion + n_head
        colour = GOOD if name == "gated" else (WARN if name == "proposed" else ACCENT)
        ax.scatter([x], [y], s=60, color=colour, zorder=3)
        dx, dy, ha, va = LABEL_AT.get(name, (0, 8, "center", "bottom"))
        ax.annotate(f"{name}\n{n_fusion:,} + {n_head:,}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=7, color="#3A4550")
    ys = [r[3] for r in rows]
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters in the fusion block + prediction head (log scale)")
    ax.set_ylabel("mean test AUC, 5 classification sets")
    pad = max(0.02, (max(ys) - min(ys)) * 1.5)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlim(3e4, 3e6)
    save(fig, "fig2_params_vs_accuracy")


NAMED = {"chemprop": (8, -4), "rf": (-14, -4), "trf": (7, -3), "desc": (-4, 8),
         "attentivefp": (6, 4), "fuse_proposed_gpu": (6, -10)}


def fig3_setsize_vs_coverage():
    path = os.path.join(MET, "conformal_alpha0.1_absolute.csv")
    if not os.path.exists(path):
        return print("  fig3 skipped: no conformal archive")
    t = pd.read_csv(path)
    t = t[t.dataset == "tox21"].copy()
    t["actives"] = 100 * t.coverage_pos
    ctrl_path = os.path.join(MET, "conformal_alpha0.1_absolute_cpu_controls.csv")
    ctrl = pd.read_csv(ctrl_path) if os.path.exists(ctrl_path) else pd.DataFrame()
    ctrl = ctrl[(ctrl.get("tag") == "desc_nopw")] if len(ctrl) else ctrl

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    unweighted = t.tag.isin(["rf", "trf", "chemprop"])
    ax.scatter(t.mean_set_size[~unweighted], t.actives[~unweighted], s=42, color=ACCENT, zorder=3,
               label=f"class-weighted loss, or a blend with one (n={int((~unweighted).sum())})")
    ax.scatter(t.mean_set_size[unweighted], t.actives[unweighted], s=42, color=WARN, zorder=3,
               label=f"unweighted loss or random forest (n={int(unweighted.sum())})")
    if len(ctrl):
        r = ctrl.iloc[0]
        ax.scatter([r.mean_set_size], [100 * r.coverage_pos], s=70, facecolor="white",
                   edgecolor=WARN, linewidth=1.6, zorder=4,
                   label="control: descriptor MLP without class weighting")
        desc = t[t.tag == "desc"].iloc[0]
        ax.annotate("", xy=(r.mean_set_size, 100 * r.coverage_pos),
                    xytext=(desc.mean_set_size, 100 * desc.coverage_pos),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax.axhline(90, ls="--", lw=1, color=GREY, zorder=1)
    ax.text(1.27, 91, "nominal 90%", color=GREY, fontsize=8, ha="right")
    for _, r in t.iterrows():
        if r.tag in NAMED:
            ax.annotate(r.tag.replace("fuse_", "").replace("_gpu", ""),
                        (r.mean_set_size, r.actives), textcoords="offset points",
                        xytext=NAMED[r.tag], fontsize=7, color="#3A4550")
    ax.set_xlabel("mean prediction-set size (max 2)")
    ax.set_ylabel("coverage of active compounds (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig3_setsize_vs_minority_coverage")


FAILING = ("rf", "trf", "chemprop")


def fig4_coverage_by_distance():
    path = os.path.join(MET, "conformal_alpha0.1_bydistance.csv")
    if not os.path.exists(path):
        return print("  fig4 skipped: no by-distance archive")
    d = pd.read_csv(path)
    d = d[d.dataset == "tox21"].copy()
    d["group"] = np.where(d.tag.isin(FAILING), "3 failing models",
                          "12 covering models")
    g = d.groupby(["group", "band"], as_index=False)[["n_band", "coverage", "coverage_pos"]].mean()
    g.to_csv(os.path.join(MET, "conformal_bydistance_groups.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    bands = sorted(d.band.unique())
    x = np.arange(len(bands))
    for group, colour in (("12 covering models", ACCENT),
                          ("3 failing models", WARN)):
        sub = g[g.group == group].set_index("band").loc[bands]
        ax.plot(x, 100 * sub.coverage, "o--", color=colour, lw=1.2, label=f"all molecules, {group}")
        ax.plot(x, 100 * sub.coverage_pos, "s-", color=colour, lw=1.8, label=f"actives, {group}")
    ax.axhline(90, ls=":", lw=1, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=7.5)
    ax.set_xlabel("Tanimoto similarity to the nearest training molecule")
    ax.set_ylabel("conformal coverage (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=6.8, loc="center right")
    save(fig, "fig4_coverage_by_distance")


def main():
    print("Building figures from the archives...")
    for old in ("fig1_setsize_vs_minority_coverage", "fig2_coverage_by_distance",
                "fig3_params_vs_accuracy", "fig4_canonical_vs_seeded"):
        for ext in ("pdf", "png"):
            p = os.path.join(OUT, f"{old}.{ext}")
            if os.path.exists(p) and old != "fig3_setsize_vs_minority_coverage":
                os.remove(p)
    fig1_split_gap()
    fig2_params_vs_accuracy()
    fig3_setsize_vs_coverage()
    fig4_coverage_by_distance()
    print(f"\nDone. Figures in {OUT}/")


if __name__ == "__main__":
    main()

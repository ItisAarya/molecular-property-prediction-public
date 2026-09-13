"""
scripts/make_figures.py

Generate every figure in `paper/draft.md` from the archives.

    python -m scripts.make_figures

Writes `paper/figures/fig<N>_*.pdf` and a matching `.png` for each.

WHY THIS EXISTS
---------------
Figures are the one part of a paper that readers trust most and verify least: a scatter plot
is taken on faith in a way a number in a sentence is not. So no figure here is drawn by hand
or from transcribed values. Each reads the same archived per-split metrics that
`scripts/check_paper.py` asserts the prose against, which means a figure and the sentence
beside it cannot disagree without one of them failing a check.

Vector PDF is the submission format; the PNG is for the README and the explainer decks.

WHAT EACH FIGURE IS FOR
-----------------------
1  set size vs minority coverage -- the paper's central uncertainty claim, and the one that
   is much more convincing as a picture than as a table: the fifteen models fall into two
   clusters with nothing between them.
2  coverage against distance to the training set -- the exchangeability assumption failing
   monotonically, which is the mechanism behind figure 1.
3  fusion parameters against accuracy -- the efficiency result. A log-scaled x-axis over two
   orders of magnitude with a flat y-axis is the whole argument.
4  canonical split against seeded mean, per model -- why one split is not a measurement.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.materialize import dataset_names
from src.eval.intervals import ci95
from src.eval.metrics import is_classification

MET = os.path.join("results", "metrics")
RUNS = os.path.join("results", "runs")
OUT = os.path.join("paper", "figures")
SEEDS = [f"seed{i}" for i in range(5)]

# One restrained palette, colour-blind safe, used across every figure.
INK = "#14304F"
ACCENT = "#2A6099"
WARN = "#C1553B"
GOOD = "#2E7D5B"
GREY = "#8A94A0"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4A5560",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {OUT}/{name}.pdf and .png")


def fig1_setsize_vs_coverage():
    """The bimodal split that mean set size predicts."""
    path = os.path.join(MET, "conformal_alpha0.1_absolute.csv")
    if not os.path.exists(path):
        return print("  fig1 skipped: no conformal archive")
    t = pd.read_csv(path)
    t = t[t.dataset == "tox21"].copy()
    t["actives"] = 100 * t.coverage_pos

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fails = t.mean_set_size <= 1.05
    ax.scatter(t.mean_set_size[fails], t.actives[fails], s=46, color=WARN,
               zorder=3, label=f"mean set size $\\leq$ 1.05  (n={int(fails.sum())})")
    ax.scatter(t.mean_set_size[~fails], t.actives[~fails], s=46, color=ACCENT,
               zorder=3, label=f"mean set size > 1.05  (n={int((~fails).sum())})")

    ax.axhline(90, ls="--", lw=1, color=GREY, zorder=1)
    ax.text(1.255, 91, "nominal 90%", color=GREY, fontsize=8, ha="right")
    ax.axvspan(t.mean_set_size[fails].max(), t.mean_set_size[~fails].min(),
               color="#F0F2F5", zorder=0)

    # Only the models worth naming. Labelling all fifteen made the upper cluster
    # unreadable and the three that matter are the ones on the left.
    NAMED = {"chemprop", "rf", "trf", "desc", "attentivefp", "fuse_proposed_gpu"}
    for _, r in t.iterrows():
        if r.tag not in NAMED:
            continue
        ax.annotate(r.tag.replace("fuse_", "").replace("_gpu", ""),
                    (r.mean_set_size, r.actives), textcoords="offset points",
                    xytext=(6, -3), fontsize=7, color="#3A4550")

    ax.set_xlabel("mean prediction-set size  (max 2)")
    ax.set_ylabel("coverage of active compounds (%)")
    ax.set_title("Set size predicts which models abandon the minority class\n"
                 "Tox21, marginal split conformal at a nominal 90%", loc="left")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    save(fig, "fig1_setsize_vs_minority_coverage")


def fig2_coverage_by_distance():
    """Coverage decaying with Tanimoto distance to the nearest training molecule."""
    path = os.path.join(MET, "conformal_by_distance_alpha0.1_absolute.csv")
    if not os.path.exists(path):
        return print("  fig2 skipped: no by-distance archive "
                     "(python -m src.eval.conformal --by-distance ...)")
    d = pd.read_csv(path)
    d = d[d.dataset == "tox21"]
    if d.empty:
        return print("  fig2 skipped: no tox21 rows")
    g = d.groupby("band", as_index=False)[["coverage", "coverage_pos"]].mean()

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(g))
    ax.plot(x, 100 * g.coverage, "o-", color=ACCENT, lw=1.8, label="all molecules")
    ax.plot(x, 100 * g.coverage_pos, "s-", color=WARN, lw=1.8, label="active compounds")
    ax.axhline(90, ls="--", lw=1, color=GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(g.band, fontsize=7.5)
    ax.set_xlabel("Tanimoto similarity to nearest training molecule")
    ax.set_ylabel("conformal coverage (%)")
    ax.set_title("The guarantee decays exactly where novel chemistry lives\n"
                 "Tox21 test molecules, binned by distance to training", loc="left")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig2_coverage_by_distance")


def _seed_mean(ds, tag):
    vals = []
    for s in SEEDS:
        p = os.path.join(RUNS, s, "metrics", f"{ds}_{tag}_test.csv")
        if os.path.exists(p):
            r = pd.read_csv(p).iloc[0]
            vals.append(float(r["auc" if is_classification(ds) else "rmse"]))
    return ci95(vals) if vals else (np.nan, np.nan, np.nan, 0)


def fig3_params_vs_accuracy():
    """Fusion-block parameters against accuracy: the efficiency result."""
    from src.models.fusion import build_fusion
    rungs = ["concat", "gated", "xattn", "bilinear", "proposed"]
    params = {m: sum(p.numel() for p in build_fusion(m, n_views=3, d=256).parameters())
              for m in rungs}

    cls = [d for d in dataset_names() if is_classification(d)]
    rows = []
    for m in rungs:
        per = [_seed_mean(ds, f"fuse_{m}") for ds in cls]
        vals = [v[0] for v in per if v[3] == 5]
        if vals:
            rows.append((m, max(params[m], 1), float(np.mean(vals))))
    if not rows:
        return print("  fig3 skipped: no archived ladder metrics")

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    best = max(ys)
    ax.axhspan(best - 0.02, best + 0.02, color="#EDF1F5", zorder=0)
    ax.text(1.4, best + 0.021, "$\pm$0.02 AUC: the minimum detectable effect (§3.5)",
            fontsize=7.5, color="#55606B")
    for name, x, y in rows:
        colour = GOOD if name == "gated" else (WARN if name == "proposed" else ACCENT)
        ax.scatter([x], [y], s=64, color=colour, zorder=3)
        ax.annotate(f"{name}\n{x:,}" if x > 1 else f"{name}\n0",
                    (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7, color="#3A4550")

    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters in the fusion block (log scale)")
    ax.set_ylabel("mean test AUC over 5 seeded splits")
    ax.set_title("Two orders of magnitude of fusion capacity, one flat line\n"
                 "averaged over the five classification datasets", loc="left")
    lo, hi = min(ys), max(ys)
    pad = max(0.02, (hi - lo) * 2)
    ax.set_ylim(lo - pad, hi + pad)
    save(fig, "fig3_params_vs_accuracy")


def fig4_split_disagreement():
    """The canonical split against the seeded mean, per model."""
    tags = ["rf", "gnn", "trf", "hybrid", "gin_ref", "desc", "gine", "fuse_gated",
            "fuse_proposed"]
    pts = []
    for tag in tags:
        for ds in dataset_names():
            dc_path = os.path.join(RUNS, "deepchem", "metrics", f"{ds}_{tag}_test.csv")
            if not os.path.exists(dc_path) or not is_classification(ds):
                continue
            dc = float(pd.read_csv(dc_path).iloc[0]["auc"])
            m, _, _, n = _seed_mean(ds, tag)
            if n == 5:
                pts.append((dc, m))
    if not pts:
        return print("  fig4 skipped: no paired canonical/seeded metrics")

    dc = np.array([p[0] for p in pts])
    sd = np.array([p[1] for p in pts])
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lim = (min(dc.min(), sd.min()) - 0.03, max(dc.max(), sd.max()) + 0.03)
    ax.plot(lim, lim, ls="--", lw=1, color=GREY, zorder=1)
    big = np.abs(dc - sd) > 0.02
    ax.scatter(dc[~big], sd[~big], s=28, color=ACCENT, zorder=3,
               label=f"within $\\pm$0.02 AUC ({int((~big).sum())})")
    ax.scatter(dc[big], sd[big], s=28, color=WARN, zorder=3,
               label=f"beyond the MDE ({int(big.sum())})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("canonical scaffold split, test AUC")
    ax.set_ylabel("mean of five seeded scaffold splits")
    ax.set_title("One split is not a measurement\n"
                 "each point is one (model, dataset)", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    save(fig, "fig4_canonical_vs_seeded")


def main():
    print("Building figures from the archives...")
    fig1_setsize_vs_coverage()
    fig2_coverage_by_distance()
    fig3_params_vs_accuracy()
    fig4_split_disagreement()
    print(f"\nDone. Figures in {OUT}/")


if __name__ == "__main__":
    main()

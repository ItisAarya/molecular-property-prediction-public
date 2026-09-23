"""
scripts/check_paper.py

Assert that `paper/draft.md` still matches `results/`.

    python -m scripts.check_paper

Three kinds of check, and the script exits non-zero if any fails:

1. **Generated tables are current.** Every `<!-- BEGIN GENERATED: name -->` block is rebuilt in
   memory by `scripts/fill_draft_tables.py` and compared with the draft. A table can only be out
   of date if someone edited it by hand or forgot to regenerate it.
2. **No placeholder survives.** A `[[...]]` left in the draft is a number nobody filled in.
3. **Every number in the prose is recomputed.** Each claim below is a sentence fragment with its
   numbers left as `{}`; the numbers are recomputed from the archives, formatted, substituted and
   searched for in the draft. A claim whose number changed fails, and so does a claim whose
   sentence was reworded or deleted -- dropping an inconvenient number must not pass silently.

Paper prose outlives the runs that produced it, which is why this exists.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

DRAFT = os.path.join("paper", "draft.md")
MET = os.path.join("results", "metrics")
RUNS = os.path.join("results", "runs")
SEEDS = [f"seed{i}" for i in range(5)]


def cmp(a, b):
    return pd.read_csv(os.path.join(MET, f"view_compare_{a}_vs_{b}.csv"))


def f3(x):
    return f"{x:.3f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def across(a, b):
    r = cmp(a, b).iloc[0]
    return int(r.across_wins), int(r.across_n_datasets), r.across_p_sign, r.across_p_wilcoxon_dz, \
        r.across_p_wilcoxon_raw


ORDER = ["tox21", "bbbp", "clintox", "bace", "sider", "esol", "lipophilicity", "freesolv"]
DISPLAY = dict(zip(ORDER, ["Tox21", "BBBP", "ClinTox", "BACE", "SIDER", "ESOL", "Lipophilicity", "FreeSolv"]))


def names(datasets):
    """Datasets in paper order and spelling, joined as prose: "ESOL", "ESOL and FreeSolv", "A, B and C"."""
    shown = [DISPLAY[d] for d in sorted(datasets, key=ORDER.index)]
    if not shown:
        return "no dataset"
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def improved(a, b):
    d = cmp(a, b)
    return list(d[d.verdict_holm == "improves"].dataset)


def row(a, b, dataset):
    return cmp(a, b).set_index("dataset").loc[dataset]


def holm_counts(a, b):
    d = cmp(a, b)
    return int((d.verdict_holm == "improves").sum()), int((d.verdict_holm == "degrades").sum())


def conformal(name):
    return pd.read_csv(os.path.join(MET, name))


def claims():
    out = []

    def claim(label, template, *values):
        out.append((label, template.format(*values)))

    # ---- abstract and section 4.1: the notation leak ------------------------------------------
    leak = pd.read_csv(os.path.join(MET, "leakage_effect.csv")).set_index(["dataset", "model"])
    seq_c = leak.loc[("clintox", "fuse_seqonly")]
    prop_c = leak.loc[("clintox", "fuse_proposed")]
    audit = pd.read_csv(os.path.join(MET, "notation_audit.csv"))
    ct = audit[(audit.dataset == "clintox")]
    bb = audit[(audit.dataset == "bbbp")].iloc[0]
    claim("abstract: seq-only ClinTox leak", "frozen ChemBERTa model from {} to {} AUC on ClinTox",
          f3(seq_c.seeded_mean_raw), f3(seq_c.seeded_mean_canonical))
    claim("abstract: proposed ClinTox leak", "three-view fusion model from {} to {}",
          f3(prop_c.seeded_mean_raw), f3(prop_c.seeded_mean_canonical))
    claim("abstract: ClinTox toxicity notation AUC", "with AUC {} for ClinTox toxicity",
          f3(ct.notation_auc.max()))
    claim("4.1: ClinTox aromatic molecules and AUCs",
          "Among its {} aromatic molecules the single notation bit predicts CT_TOX with AUC {} and "
          "FDA_APPROVED with AUC {}", f"{int(ct.aromatic_molecules.iloc[0]):,}",
          f3(ct[ct.task == 1].notation_auc.iloc[0]), f3(ct[ct.task == 0].notation_auc.iloc[0]))
    claim("4.1: BBBP notation", "BBBP mixes both conventions ({}% Kekulé), and the bit predicts "
          "permeability with AUC {}", pct(bb.kekule_fraction), f3(bb.notation_auc))
    seq_b = leak.loc[("bbbp", "fuse_seqonly")]
    claim("4.1: seq-only drops", "lowered the frozen ChemBERTa view by {} AUC on ClinTox and {} on BBBP",
          f3(-seq_c.seeded_mean_change), f3(-seq_b.seeded_mean_change))
    fusion = [m for m in ("fuse_gated_nograph", "fuse_concat", "fuse_gated", "fuse_xattn",
                          "fuse_bilinear", "fuse_proposed")]
    drops = [-leak.loc[(ds, m)].seeded_mean_change for ds in ("clintox", "bbbp") for m in fusion]
    claim("4.1: fusion drop range", "and each fusion model by {} to {} AUC", f3(min(drops)), f3(max(drops)))
    gin = {ds: leak.loc[(ds, "gin_ref (reads no SMILES characters)")].seeded_mean_canonical
           for ds in ("clintox", "bbbp")}
    canon = {ds: [leak.loc[(ds, m)].seeded_mean_canonical for m in fusion] for ds in ("clintox", "bbbp")}
    below = all(v < gin["clintox"] for v in canon["clintox"])
    claim("4.1: canonical fusion against GIN", "On ClinTox every canonicalised fusion model scores {} the "
          "two-layer GIN ({}–{} against {}); on BBBP they land within {} of it ({}–{} against {})",
          "below" if below else "[NOT ALL BELOW]", f3(min(canon["clintox"])), f3(max(canon["clintox"])),
          f3(gin["clintox"]), f3(max(abs(v - gin["bbbp"]) for v in canon["bbbp"])),
          f3(min(canon["bbbp"])), f3(max(canon["bbbp"])), f3(gin["bbbp"]))

    # ---- section 3.1: missing labels ----------------------------------------------------------
    pool = np.load(os.path.join("data", "pool", "tox21_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join("data", "splits", "tox21_deepchem.json")))
    y = np.asarray(pool["y_raw"], dtype=float)
    claim("3.1: Tox21 missing labels", "{}% of training labels and {}% of test labels on the canonical "
          "split are missing", pct(np.isnan(y[idx["train"]]).mean()), pct(np.isnan(y[idx["test"]]).mean()))

    # ---- section 4.3: split conventions -------------------------------------------------------
    gap = pd.read_csv(os.path.join(MET, "split_gap.csv"))
    claim("4.3: split gap", "Over all {} (model, classification dataset) pairs with results under both "
          "conventions, the seeded mean exceeded the canonical split in {} cases (median difference {} "
          "AUC, maximum {}), and {} pairs differed by more than 0.02 AUC", len(gap), int((gap.gap > 0).sum()),
          f"{gap.gap.median():+.3f}", f3(gap.gap.max()), int((gap.gap.abs() > 0.02).sum()))
    claim("8: split gap max", "differ by up to {} AUC on the same model", f3(gap.gap.max()))

    # ---- section 5.1: single views ------------------------------------------------------------
    w, n, ps, pdz, praw = across("desc", "gin_ref")
    claim("5.1: desc vs gin_ref", "favoured on {} of {} by mean; across datasets p = {}/{}/{}",
          w, n, f3(ps), f3(pdz), f3(praw))
    w, n, ps, _, _ = across("seq_frozen", "gin_ref")
    claim("5.1: frozen vs gin_ref", "was favoured on none of the {} datasets it can be scored on "
          "(sign test p = {})", "six" if n == 6 else n, f3(ps))
    lora = cmp("lora", "seq_frozen").set_index("dataset")
    claim("5.1: LoRA ESOL", "LoRA fine-tuning improved ESOL (Holm p = {})", f"{lora.loc['esol'].p_holm:.4f}")

    # ---- abstract and section 5.3: the cached ladder --------------------------------------------
    w, n, ps, pdz, praw = across("fuse_proposed", "gin_ref")
    claim("abstract: proposed vs gin_ref", "no longer significant across datasets (favoured on {} of {}, p ≥ {})",
          w, n, f3(min(ps, pdz, praw)))
    claim("5.3: proposed vs gin_ref", "`proposed` was favoured over the two-layer GIN on {} of {} datasets, but no "
          "across-dataset statistic reached 0.05 (p = {} / {} / {}), and only {} improved after correction ({} "
          "against {} kcal·mol⁻¹)", w, n, f3(ps), f3(pdz), f3(praw), names(improved("fuse_proposed", "gin_ref")),
          f3(row("fuse_proposed", "gin_ref", "freesolv").mean_a), f3(row("fuse_proposed", "gin_ref", "freesolv").mean_b))
    w, n, ps, pdz, praw = across("desc", "gin_ref")
    claim("5.3: desc vs gin_ref", "The descriptor MLP had the same record against the GIN ({} of {}; p = {} / {} / {})",
          w, n, f3(ps), f3(pdz), f3(praw))
    w, n, ps, pdz, praw = across("fuse_proposed", "desc")
    claim("5.3: proposed vs desc", "`proposed` was favoured over it on {} of {} datasets (p ≥ {}) and `xattn` on {}",
          w, n, f3(min(ps, pdz, praw)), across("fuse_xattn", "desc")[0])
    low = {m: across(f"fuse_{m}", "desc") for m in ("concat", "gated", "bilinear")}
    one = sorted({v[0] for v in low.values()})
    claim("5.3: concat/gated/bilinear vs desc", "`concat`, `gated` and `bilinear` were each favoured on {} of 8",
          one[0] if len(one) == 1 else one)
    wil = sorted({f3(low[m][k]) for m in ("gated", "bilinear") for k in (3, 4)})
    sign = sorted({f3(low[m][2]) for m in ("gated", "bilinear")})
    holm_none = all(holm_counts(f"fuse_{m}", "desc") == (0, 0) for m in ("gated", "bilinear"))
    claim("5.3: gated/bilinear worse than desc", "were significantly worse than the MLP on both Wilcoxon statistics "
          "(p = {}; sign test p = {}), with {}", "/".join(wil), "/".join(sign),
          "no single dataset surviving correction" if holm_none else "[HOLM CHANGES]")
    claim("abstract: gated/bilinear vs desc", "the gated and bilinear variants were worse than it on {} of 8 datasets "
          "(Wilcoxon p = {})", "/".join(sorted({str(8 - low[m][0]) for m in ("gated", "bilinear")})), "/".join(wil))
    wc, _, psc, pdzc, prawc = across("fuse_proposed", "fuse_concat")
    wg, _, psg, pdzg, prawg = across("fuse_proposed", "fuse_gated")
    claim("5.3: proposed vs concat/gated", "`proposed` was favoured over `concat` and over `gated` on {} of 8 "
          "datasets each. Against `gated` both Wilcoxon statistics were significant (p = {}) but the sign test was "
          "not (p = {}) and {}; against `concat` no across-dataset statistic was significant (p ≥ {}) and {} "
          "improved after correction", wc if wc == wg else f"{wc}/{wg}", "/".join(sorted({f3(pdzg), f3(prawg)})),
          f3(psg), "no dataset survived correction" if holm_counts("fuse_proposed", "fuse_gated") == (0, 0)
          else "[HOLM CHANGES]", f3(min(psc, pdzc, prawc)), names(improved("fuse_proposed", "fuse_concat")))

    # ---- section 5.3: the earlier pipeline ------------------------------------------------------
    rf, gnn, trf, hyb = (cmp("fuse_proposed", b) for b in ("rf", "gnn", "trf", "hybrid"))
    claim("5.3: pipeline", "`proposed` was favoured over the random forest on {} five datasets ({} after "
          "correction), over the earlier GNN on {} of {} ({} after correction; the GNN led on {}), and over the "
          "SMILES transformer and the stacking meta-learner on {} three comparable datasets ({} and {}, "
          "respectively, after correction)",
          "all" if (rf.mean_diff > 0).all() and len(rf) == 5 else "[NOT ALL]", names(improved("fuse_proposed", "rf")),
          int((gnn.mean_diff > 0).sum()), len(gnn), names(improved("fuse_proposed", "gnn")),
          names(list(gnn[gnn.mean_diff < 0].dataset)),
          "all" if (trf.mean_diff > 0).all() and (hyb.mean_diff > 0).all() and len(trf) == len(hyb) == 3 else "[NOT ALL]",
          "all three" if len(improved("fuse_proposed", "trf")) == 3 else names(improved("fuse_proposed", "trf")),
          names(improved("fuse_proposed", "hybrid")))
    w, n, ps, pdz, praw = across("fuse_proposed_e2e", "fuse_proposed")
    claim("5.3: end-to-end vs cached", "(the end-to-end model was favoured on {} of {}; p = {} on all three "
          "statistics, the smallest attainable at *n* = 6; {})", w, n,
          f3(ps) if f3(ps) == f3(pdz) == f3(praw) else f"{f3(ps)}/{f3(pdz)}/{f3(praw)}",
          "no dataset differed after correction" if holm_counts("fuse_proposed_e2e", "fuse_proposed") == (0, 0)
          else "[HOLM CHANGES]")

    # ---- section 5.4: graph view and gate weights ------------------------------------------------
    ga = pd.read_csv(os.path.join(MET, "gate_attribution_fuse_gated.csv")).set_index("dataset")
    gn = pd.read_csv(os.path.join(MET, "gate_attribution_fuse_gated_nograph.csv")).set_index("dataset")
    claim("5.4: graph gate weight", "assigned the graph view {}–{}% of its weight", pct(ga.graph.min()),
          pct(ga.graph.max()))
    w, n, ps, pdz, praw = across("fuse_gated_nograph", "fuse_gated")
    claim("5.4: nograph vs gated", "(the model without the graph view was favoured on {} of {} datasets; "
          "p = {} / {} / {}; {})", w, n, f3(ps), f3(pdz),
          f3(praw), "no dataset differed after correction" if holm_counts("fuse_gated_nograph", "fuse_gated") == (0, 0)
          else "[HOLM CHANGES]")
    eq = pd.read_csv(os.path.join(MET, "equivalence.csv"))
    eq = eq[(eq.a == "fuse_gated_nograph") & (eq.b == "fuse_gated") & eq.equivalent]
    claim("5.4: nograph equivalence", "established equivalence on only {} of 8 datasets ({}; 90% interval",
          len(eq), names(list(eq.dataset)))
    claim("5.4: gate at most", "Removing a view that carried at most {}% of the gate weight", pct(ga.graph.max()))
    seq_pref = [ds for ds in ga.index if ga.loc[ds].gate_prefers == "seq"]
    rose = all(gn.loc[ds].desc > ga.loc[ds].desc for ds in ga.index)
    claim("5.4: gate shift", "the descriptor weight rose on {} eight datasets, from {}–{} to {}–{}, and the three "
          "datasets on which the three-view gate had favoured the sequence view ({}) switched to the descriptor view",
          "all" if rose else "[NOT ALL]", f"{ga.desc.min():.2f}", f"{ga.desc.max():.2f}", f"{gn.desc.min():.2f}",
          f"{gn.desc.max():.2f}", names(sorted(seq_pref, key=lambda d: ORDER.index(d))) if all(
              gn.loc[d].gate_prefers == "desc" for d in seq_pref) else "[NOT SWITCHED]")

    # ---- section 5.5: proposed against the external baselines ------------------------------------
    afp_c, chem_c = across("fuse_proposed", "attentivefp"), across("fuse_proposed", "chemprop")
    claim("5.5: proposed vs external", "`proposed` was favoured over AttentiveFP and over Chemprop on {} of 8 "
          "datasets each, with no across-dataset statistic significant (p ≥ {}); against Chemprop, {} improved "
          "after correction", afp_c[0] if afp_c[0] == chem_c[0] else f"{afp_c[0]}/{chem_c[0]}",
          f3(min(afp_c[2:] + chem_c[2:])), names(improved("fuse_proposed", "chemprop")))

    # ---- section 5.6: mechanisms ------------------------------------------------------------------
    t4 = across("fuse_bilinear_gpu", "fuse_concat_gpu")
    e2e = across("fuse_bilinear_e2e", "fuse_concat_e2e")
    claim("5.6: bilinear T4", "favoured on {} six datasets with the smallest p-value attainable at *n* = 6 ({})",
          "all" if t4[0] == e2e[0] == 6 else "[NOT ALL]", "/".join(sorted({f3(t4[3]), f3(e2e[3])})))
    cpu = across("fuse_bilinear", "fuse_concat")
    bc = cmp("fuse_bilinear", "fuse_concat").set_index("dataset")
    shared = list(cmp("fuse_bilinear_gpu", "fuse_concat_gpu").dataset)
    claim("5.6: bilinear CPU", "It did not separate on the CPU ladder once ClinTox and BBBP were canonicalised ({} of "
          "{}, p = {}): it lost to `concat` on ClinTox in {} five splits ({} against {} AUC) and was favoured on {} of "
          "the {} datasets that the T4 settings cover", cpu[0], cpu[1], f3(cpu[3]),
          "all" if bc.loc["clintox"].a_wins == 0 else "[NOT ALL]", f3(bc.loc["clintox"].mean_a),
          f3(bc.loc["clintox"].mean_b), int((bc.loc[shared].mean_diff > 0).sum()), len(shared))

    pb = across("fuse_proposed", "fuse_bilinear")
    claim("5.6: proposed vs bilinear CPU", "(on the CPU ladder only the raw-difference Wilcoxon reached p = {}; "
          "sign p = {}, *dz* p = {})", f3(pb[4]), f3(pb[2]), f3(pb[3]))

    # ---- section 6.1: calibration maps ------------------------------------------------------------
    ece = pd.read_csv(os.path.join(MET, "ece_multiseed_leakfree.csv"))
    hurt = ece[ece.hurts]
    claim("6.1: ECE", "Over {} (model, classification dataset) pairs, fitting a calibration map on the validation "
          "split reduced test ECE by more than its interval on {}, increased it on {} (Chemprop on SIDER, {} to {})",
          len(ece), int(ece.helps.sum()), len(hurt),
          f3(hurt.ece_raw.iloc[0]) if len(hurt) == 1 and hurt.tag.iloc[0] == "chemprop" else "[CHANGED]",
          f3(hurt.ece_cal.iloc[0]) if len(hurt) == 1 else "[CHANGED]")
    claim("6.1: ECE medians", "(median raw ECE {} where the map helped, {} elsewhere)",
          f3(ece[ece.helps].ece_raw.median()), f3(ece[~ece.helps].ece_raw.median()))

    # ---- section 5.5: AttentiveFP parameters ---------------------------------------------------
    from src.models.encoders.graph import build_graph_encoder
    count = lambda enc, **kw: sum(p.numel() for p in build_graph_encoder(enc, in_dim=34, hidden=256, **kw).parameters())
    afp = count("attentivefp", layers=4, timesteps=2, dropout=0.3)
    gine = count("gine", n_layers=4, readout="mean+sum")
    gin = count("gin")
    claim("5.5: AttentiveFP parameters", "AttentiveFP ({} encoder parameters, {} times GINE and {} times "
          "the two-layer GIN)", f"{afp:,}", f"{afp / gine:.1f}", f"{afp / gin:.1f}")
    claim("3.2: encoder parameters", "(`gin_ref`, {} encoder parameters)", f"{gin:,}")

    # ---- section 5.6: rank sweep --------------------------------------------------------------
    ranks = {16: "fuse_bilinear_r16_gpu", 32: "fuse_bilinear_r32_gpu", 64: "fuse_bilinear_gpu",
             128: "fuse_bilinear_r128_gpu"}
    pairs = [(a, b) for a in ranks for b in ranks if a > b]
    ps = []
    for a, b in pairs:
        path = os.path.join(MET, f"view_compare_{ranks[a]}_vs_{ranks[b]}.csv")
        if not os.path.exists(path):
            path = os.path.join(MET, f"view_compare_{ranks[b]}_vs_{ranks[a]}.csv")
        r = pd.read_csv(path).iloc[0]
        ps += [r.across_p_sign, r.across_p_wilcoxon_dz, r.across_p_wilcoxon_raw]
    claim("5.6: ranks indistinguishable", "No pair of ranks differed on any statistic (all p ≥ {}; *n* = 6)",
          f3(min(ps)))
    other = [across(ranks[r], "fuse_concat_gpu") for r in (16, 32, 128)]
    claim("5.6: other ranks vs concat", "were favoured on {}, {} and {} of 6 datasets with p between {} and {}",
          other[0][0], other[1][0], other[2][0], f3(min(min(o[2:]) for o in other)),
          f"{max(max(o[2:]) for o in other):.2f}")

    # ---- section 5.7: motif probe ---------------------------------------------------------------
    mot = pd.read_csv(os.path.join(MET, "motif_compare_motif_vs_desc.csv"))
    add = pd.read_csv(os.path.join(MET, "motif_compare_desc_motif_vs_desc.csv")).set_index("dataset")
    claim("5.7: motifs alone", "they lost to `ECFP+desc` on {} of 8 datasets (sign test p = {}), with {} "
          "datasets surviving correction", 8 - int(mot.across_wins.iloc[0]), f"{mot.across_p_sign.iloc[0]:.4f}",
          "five" if int((mot.p_holm < 0.05).sum()) == 5 else int((mot.p_holm < 0.05).sum()))
    claim("5.7: adding motifs", "the combination was favoured on {} of 8 datasets (p = {})",
          int(add.across_wins.iloc[0]), f"{add.across_p_sign.iloc[0]:.2f}")
    lipo = add.loc["lipophilicity"]
    claim("5.7: Lipophilicity loss", "a loss of {} logD on Lipophilicity (p = {}; Holm p = {})",
          f3(-lipo.mean_diff), f"{lipo.p_ttest:.4f}", f3(lipo.p_holm))
    canon = pd.read_csv(os.path.join(MET, "motif_canonical.csv")).set_index("dataset")
    claim("5.7: canonical motifs", "motifs alone beat `ECFP+desc` on BACE ({} against {}) and SIDER ({} "
          "against {}), and the combination improved {} of 8 numbers",
          f"{canon.loc['bace'].motif:.4f}", f"{canon.loc['bace'].desc:.4f}", f"{canon.loc['sider'].motif:.4f}",
          f"{canon.loc['sider'].desc:.4f}", int(canon.both_beats_desc.sum()))
    cov = pd.read_csv(os.path.join(MET, "motif_coverage.csv"))
    claim("5.7: fragment coverage", "from {}% of a molecule's fragments seen in training on the best BACE "
          "split to {}% on the worst FreeSolv split", f"{100 * cov[cov.dataset == 'bace'].test_brics_fragment_seen.max():.0f}",
          f"{100 * cov[cov.dataset == 'freesolv'].test_brics_fragment_seen.min():.0f}")
    claim("5.7: coverage correlation", "(*r* = {} over 48 dataset–split pairs)",
          f"{np.corrcoef(cov.mean_brics_per_molecule, cov.test_brics_fragment_seen)[0, 1]:.2f}")

    # ---- section 6.2: conformal on Tox21 -------------------------------------------------------
    m = conformal("conformal_alpha0.1_absolute.csv")
    m = m[m.dataset == "tox21"]
    c = conformal("conformal_alpha0.1_conditional.csv")
    c = c[c.dataset == "tox21"]
    ctrl_m = conformal("conformal_alpha0.1_absolute_cpu_controls.csv").set_index("tag")
    ctrl_c = conformal("conformal_alpha0.1_conditional_cpu_controls.csv").set_index("tag")
    fail = m[m.tag.isin(["rf", "trf", "chemprop"])]
    ok = m[~m.tag.isin(["rf", "trf", "chemprop"])]
    claim("6.2: overall coverage range", "overall coverage ranged from {}% to {}% across the fifteen models",
          pct(m.coverage.min()), pct(m.coverage.max()))
    claim("6.2: failing group", "three models covered {}–{}% of actives with mean set sizes of {}–{}",
          pct(fail.coverage_pos.min()), pct(fail.coverage_pos.max()),
          f"{fail.mean_set_size.min():.2f}", f"{fail.mean_set_size.max():.2f}")
    claim("6.2: covering group", "the other twelve covered {}–{}% with set sizes of {}–{}",
          pct(ok.coverage_pos.min()), pct(ok.coverage_pos.max()),
          f"{ok.mean_set_size.min():.2f}", f"{ok.mean_set_size.max():.2f}")
    auc_pw = np.mean([float(pd.read_csv(os.path.join(RUNS, v, "metrics", "tox21_desc_test.csv")).iloc[0].auc) for v in SEEDS])
    auc_np = np.mean([float(pd.read_csv(os.path.join(RUNS, v, "metrics", "tox21_desc_nopw_test.csv")).iloc[0].auc) for v in SEEDS])
    claim("6.2: control", "Its AUC was unchanged ({} against {}), but active coverage fell from {}% to {}% and "
          "mean set size from {} to {}", f3(auc_np), f3(auc_pw), pct(ctrl_m.loc["desc"].coverage_pos),
          pct(ctrl_m.loc["desc_nopw"].coverage_pos), f"{ctrl_m.loc['desc'].mean_set_size:.2f}",
          f"{ctrl_m.loc['desc_nopw'].mean_set_size:.2f}")
    claim("abstract: control", "reproduced that failure ({}% to {}%)",
          pct(ctrl_m.loc["desc"].coverage_pos), pct(ctrl_m.loc["desc_nopw"].coverage_pos))
    claim("6.2: class-conditional", "restored active coverage to {}–{}% for every model, including the control "
          "({}%), at larger mean set sizes ({}–{})", pct(c.coverage_pos.min()), pct(c.coverage_pos.max()),
          pct(ctrl_c.loc["desc_nopw"].coverage_pos), f"{c.mean_set_size.min():.2f}", f"{c.mean_set_size.max():.2f}")
    claim("abstract: failing range", "three of fifteen models covered only {}–{}% of active compounds",
          f"{100 * fail.coverage_pos.min():.0f}", f"{100 * fail.coverage_pos.max():.0f}")

    # ---- sections 6.2 and 6.3: scripts/validate_conformal.py ---------------------------------------
    val = pd.read_csv(os.path.join(MET, "conformal_validation.csv"))
    syn = val[val.check == "synthetic"].set_index("item").value
    claim("6.2: synthetic validation", "at a 90% target, coverage was {}% for regression intervals, {}% for "
          "marginal binary sets and {}% for actives under class-conditional sets", pct(syn["regression"]),
          pct(syn["binary_marginal"]), pct(syn["binary_conditional_actives"]))
    tmp = val[val.check == "temperature"]
    claim("6.3: temperature invariance", "the sets were {} for every task and split, while individual "
          "probabilities moved by up to {}", "identical" if len(tmp) == 9 and tmp.identical.astype(bool).all()
          else "[NOT IDENTICAL OR MISSING]", f"{tmp.value.max():.2f}")

    # ---- section 6.3: Platt ---------------------------------------------------------------------
    lg = conformal("conformal_alpha0.1_logistic_absolute.csv")
    claim("6.3: Platt", "raised active coverage of the random forest to {}%",
          pct(lg[(lg.tag == "rf") & (lg.dataset == "tox21")].coverage_pos.iloc[0]))

    # ---- section 6.5: regression intervals ----------------------------------------------------
    ab = conformal("conformal_alpha0.1_absolute.csv")
    no = conformal("conformal_alpha0.1_normalized.csv")
    cq = conformal("conformal_alpha0.1_cqr.csv").set_index("dataset")
    ens_ab = ab[(ab.tag == "ens") & (ab.dataset == "esol")].mean_width.iloc[0]
    ens_no = no[(no.tag == "ens") & (no.dataset == "esol")].mean_width.iloc[0]
    claim("6.5: normalised widening", "widened the ensemble's mean interval {}-fold", f"{ens_no / ens_ab:.1f}")
    claim("6.5: CQR under-coverage", "under-covered by {} points on ESOL, {} on Lipophilicity and {} on FreeSolv",
          f"{90 - 100 * cq.loc['esol'].coverage:.1f}", f"{90 - 100 * cq.loc['lipophilicity'].coverage:.1f}",
          f"{90 - 100 * cq.loc['freesolv'].coverage:.1f}")

    # ---- section 6.6: distance ----------------------------------------------------------------
    g = pd.read_csv(os.path.join(MET, "conformal_bydistance_groups.csv"))
    wt = g[g.group.str.contains("covering")].set_index("band")
    un = g[g.group.str.contains("failing")]
    claim("6.6: weighted group", "overall coverage rose from {}% for the most novel molecules to {}% for close "
          "analogues, and active coverage from {}% to a peak of {}% at similarity 0.5–0.7",
          pct(wt.loc["0.0-0.3"].coverage), pct(wt.loc["0.7-1.0"].coverage), pct(wt.loc["0.0-0.3"].coverage_pos),
          pct(wt.loc["0.5-0.7"].coverage_pos))
    claim("6.6: unweighted group", "covered {}–{}% of actives in every band",
          f"{100 * un.coverage_pos.min():.0f}", f"{100 * un.coverage_pos.max():.0f}")

    # ---- section 7: device and seed -----------------------------------------------------------
    dev = pd.read_csv(os.path.join(MET, "device_effect_summary.csv")).set_index("experiment")
    ints = ["metrics_compared", "identical", "single_split_over_threshold", "canonical_over_threshold",
            "canonical_n", "five_split_mean_over_threshold", "five_split_mean_n"]
    # A row of mixed int/float columns comes back as floats, so read rows as objects and cast counts.
    rows = {k: {c: (int(v) if c in ints else v) for c, v in dev.loc[k].items()} for k in dev.index}
    d = pd.Series(rows["device: CPU vs T4"], dtype=object)
    s = pd.Series(rows["seed: 42 vs 43, same CPU"], dtype=object)
    claim("7: device counts", "Of {} test metrics, {} were identical; the median absolute difference was {} and "
          "the mean {} (maximum {} RMSE)", d.metrics_compared, d.identical, f3(d.median_abs_diff),
          f3(d.mean_abs_diff), f3(d.max_abs_diff))
    claim("7: device threshold", "{} of {} single-split results ({}%) and {} of {} canonical-split results moved "
          "by more than the threshold, but only {} of {} five-split means did", d.single_split_over_threshold,
          d.metrics_compared, f"{100 * d.single_split_fraction_over:.0f}", d.canonical_over_threshold,
          d.canonical_n, d.five_split_mean_over_threshold, d.five_split_mean_n)
    claim("7: seed threshold", "{} of {} single-split results ({}%) and {} of {} canonical-split results exceeded "
          "the threshold (median absolute difference {}), and none of the {} five-split means did",
          s.single_split_over_threshold, s.metrics_compared, f"{100 * s.single_split_fraction_over:.0f}",
          s.canonical_over_threshold, s.canonical_n, f3(s.median_abs_diff), s.five_split_mean_n)
    return out


def main():
    from scripts import fill_draft_tables as T

    md = open(DRAFT, encoding="utf-8").read()
    failures = []

    for m in re.finditer(r"<!-- BEGIN GENERATED: (\w+) -->\n(.*?)\n<!-- END GENERATED -->", md, re.S):
        if T.BUILDERS[m.group(1)]().strip() != m.group(2).strip():
            failures.append(f"generated table `{m.group(1)}` is stale: run python -m scripts.fill_draft_tables")

    body = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    for p in sorted(set(re.findall(r"\[\[[^\]]+\]\]", body))):
        failures.append(f"unfilled placeholder {p}")

    flat = " ".join(body.split())
    results = claims()
    for label, expected in results:
        if " ".join(expected.split()) not in flat:
            failures.append(f"{label}: expected to find\n      \"{expected}\"")

    n_tables = len(re.findall(r"BEGIN GENERATED", md))
    if failures:
        print(f"{len(failures)} problem(s) in {DRAFT}:\n")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"{DRAFT}: {n_tables} generated tables current, no placeholders, "
          f"{len(results)} prose claims match the archives.")


if __name__ == "__main__":
    main()

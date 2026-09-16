"""
scripts/probe_motifs.py

Would a motif view have been worth building?

    python -m scripts.probe_motifs
    python -m scripts.probe_motifs --datasets bace --variants deepchem

Writes five archives under `results/metrics/`: `motif_probe.csv` (every arm, every split),
`motif_coverage.csv`, `motif_compare_desc_motif_vs_desc.csv`, `motif_compare_motif_vs_desc.csv`
and `motif_canonical.csv`. A run narrowed by `--datasets` or `--variants` writes none of them.

WHY THIS EXISTS
---------------
The study plan for this work named a motif view -- BRICS fragments and Murcko scaffolds as
a fourth representation -- as an optional Phase 5 item, taken from the atom-plus-motif idea
in the AMCT line of work that §2 cites. It was never built, and "we ran out of time" is not an
answer to the reviewer who asks why.

This is the answer instead: a measurement of whether an explicit motif representation carries
predictive signal that the views already in the ladder do not. It is a probe, not a view. It
does not train the motif encoder the plan described; it asks the cheaper question that has to
come first -- *is there anything there to encode?* -- under a readout held identical across
arms, so the only thing that differs between them is the feature set.

THREE ARMS, ONE READOUT
-----------------------
`desc`        1024-bit ECFP + the 217 RDKit 2-D descriptors: the same *features* the
              descriptor view of §5.1 is built on.
`motif`       Motif indicators alone. Reported because "adds nothing" and "contains nothing"
              are different findings and the first is only interesting given the second.
`desc+motif`  Both. The arm the plan's motif view would have had to beat.

The readout is a linear model -- logistic regression for classification, ridge for regression,
one fixed setting, no tuning, no seed. That mirrors §3.6 and makes the comparison paired and
exactly reproducible. It also bounds the claim: this measures what motifs add *to a linear
readout*, which is a lower bound on what a trained motif encoder could extract. The bound is
worth having anyway, because of what the coverage table below says about why the answer comes
out the way it does.

THE `desc` ARM IS NOT THE DESCRIPTOR VIEW, AND MUST NOT BE QUOTED AS IT
-----------------------------------------------------------------------
It shares the view's features and not its model. The trained MLP beats this linear stand-in by
0.069 AUC on Tox21 and 0.181 RMSE on Lipophilicity; on BACE the two land within 0.0004 of each
other, which is coincidence and not reassurance. Every comparison this script reports is
*within* the table, where all three arms share the readout and only the feature set moves.
Comparing a number here against a number in §5.1 compares two different models.

THE VOCABULARY IS FITTED ON TRAINING MOLECULES ONLY
---------------------------------------------------
A motif vocabulary is a fitted object: which fragments exist, and which are frequent enough to
keep, is read off the data. Phase 0 of this project removed four stages fitted on data they
were later scored against, and a vocabulary built over the whole pool would be a fifth -- test
molecules would be choosing the feature columns. So it is built per split, from training rows
only, and a test fragment the training split never saw is simply absent. That is not a
limitation of the probe; it is the thing the probe measures.

WHY THE COVERAGE TABLE IS THE POINT
-----------------------------------
A scaffold split partitions molecules *by Murcko scaffold*. So a scaffold-level feature is
guaranteed by construction to be unseen at test time -- not usually, not mostly: always, for
the scaffold column itself. BRICS fragments are smaller than a scaffold and do transfer in
part, and `motif_coverage.csv` records how much. Whatever the predictive arms say, the
coverage numbers say why, and they would hold for any motif encoder rather than only for a
linear one.

The descriptor normalisation is `DescriptorEncoder`'s own, imported rather than reimplemented:
if the probe standardised descriptors differently from the view it is standing in for, the
`desc` arm would not be the baseline it claims to be.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression, Ridge

from src.data.materialize import dataset_names
from src.eval.intervals import ci95
from src.eval.metrics import cls_metrics, is_classification, reg_metrics
from src.eval.view_stats import across_datasets, holm
from src.models.encoders.descriptor import DescriptorEncoder

RDLogger.DisableLog("rdApp.*")

POOL_DIR = os.path.join("data", "pool")
SPLIT_DIR = os.path.join("data", "splits")
MET_DIR = os.path.join("results", "metrics")
OUT = os.path.join(MET_DIR, "motif_probe.csv")
COVERAGE_OUT = os.path.join(MET_DIR, "motif_coverage.csv")

VARIANTS = ["deepchem", "seed0", "seed1", "seed2", "seed3", "seed4"]
ARMS = ["desc", "motif", "desc+motif"]

# Prefixes keep the three fragment families separable in the coverage table. BRICS fragments
# carry their own dummy-atom notation and need no prefix to be unambiguous.
MURCKO = "MURCKO:"
GENERIC = "GENERIC:"

# Bumped whenever fragments() changes what it returns, so data/pool/*_motifs.npz written
# by an older definition is rebuilt instead of silently reused.
CACHE_VERSION = 2

# One fixed setting, as everywhere else in this project.
C_LOGREG = 1.0
ALPHA_RIDGE = 1.0
MAX_ITER = 2000


# --------------------------------------------------------------------------------------
# Fragments
# --------------------------------------------------------------------------------------
def fragments(smiles):
    """
    The motif multiset of one molecule: BRICS fragments, its Murcko scaffold, and the
    generic (element- and bond-order-flattened) form of that scaffold.

    Returns a set, not a list. A fragment occurring twice in a molecule is still one
    feature here -- the plan's motif view would have pooled over a multiset, but a count
    feature and an indicator feature cannot both be tested by a probe this size, and the
    indicator is the weaker assumption.

    WHY BreakBRICSBonds AND NOT BRICSDecompose
    ------------------------------------------
    `BRICSDecompose` returns every fragment at every level of a *recursive* decomposition,
    so its cost grows combinatorially in the number of breakable bonds. SIDER's largest
    molecules run to 476 and 484 atoms with 108 and 132 BRICS bonds, and the first version of
    this script spent roughly twenty-five minutes on SIDER without finishing it -- which
    molecule it was stuck on was never established, only that the dataset did not complete.
    Breaking every BRICS bond once and taking the connected components is linear, does those
    same molecules in 0.05 s, and yields the leaf fragments -- which is what a
    fragment-embedding view would have consumed anyway, not the hierarchy above them.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    try:
        pieces = Chem.GetMolFrags(BRICS.BreakBRICSBonds(mol), asMols=True)
        out = {Chem.MolToSmiles(p) for p in pieces}
    except Exception:
        out = set()
    try:
        # An acyclic molecule has an *empty* Murcko scaffold. `src/data/splits.py` gives each
        # of those its own singleton group rather than grouping them together, on the grounds
        # that they share no scaffold and there is nothing to leak between them. Emitting a
        # shared `MURCKO:` feature for all of them would do exactly what that code refuses to
        # do -- and would report the resulting column as transferable scaffold signal.
        smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if smi:
            out.add(MURCKO + smi)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            out.add(GENERIC + Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold)))
    except Exception:
        # Murcko fails on a handful of molecules (no ring system, or a valence RDKit will
        # not flatten). BRICS fragments still stand; dropping the whole molecule would be
        # the larger distortion.
        pass
    return out


def motif_lists(ds, refresh=False):
    """
    Per-molecule fragment sets for a dataset, cached under `data/pool/`.

    Fragments are a property of a molecule, not of a split, so this is computed once for all
    six splits. The cache stores the pool's SMILES alongside and is rebuilt if they differ,
    so a re-prepped pool cannot be scored against a stale vocabulary. It also stores
    `CACHE_VERSION`, which is bumped whenever `fragments()` changes what it returns -- a
    cache written by an older definition is silently wrong rather than missing, and relying
    on someone remembering `--refresh-cache` is not a guarantee.
    """
    pool = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    smiles = [str(s) for s in pool["smiles"]]
    cache = os.path.join(POOL_DIR, f"{ds}_motifs.npz")

    if os.path.exists(cache) and not refresh:
        z = np.load(cache, allow_pickle=True)
        fresh = ("version" in z.files and int(z["version"]) == CACHE_VERSION
                 and list(z["smiles"]) == smiles)
        if fresh:
            flat, offsets = z["flat"], z["offsets"]
            return [set(flat[offsets[i]:offsets[i + 1]]) for i in range(len(smiles))]

    t0 = time.time()
    sets = [fragments(s) for s in smiles]
    flat, offsets = [], [0]
    for s in sets:
        flat.extend(sorted(s))
        offsets.append(len(flat))
    os.makedirs(POOL_DIR, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(smiles, dtype=object),
                        flat=np.array(flat, dtype=object),
                        offsets=np.array(offsets, dtype=np.int64),
                        version=np.array(CACHE_VERSION, dtype=np.int64))
    print(f"    fragmented {len(smiles)} molecules in {time.time() - t0:.1f}s")
    return sets


def vocabulary(sets, train_rows, min_count):
    """Fragments occurring in at least `min_count` *training* molecules, in a fixed order."""
    counts = {}
    for r in train_rows:
        for f in sets[r]:
            counts[f] = counts.get(f, 0) + 1
    return sorted(f for f, n in counts.items() if n >= min_count)


def indicators(sets, rows, vocab):
    """A (len(rows), len(vocab)) binary matrix. Fragments outside the vocabulary vanish."""
    index = {f: j for j, f in enumerate(vocab)}
    x = np.zeros((len(rows), len(vocab)), dtype=np.float32)
    for i, r in enumerate(rows):
        for f in sets[r]:
            j = index.get(f)
            if j is not None:
                x[i, j] = 1.0
    return x


# --------------------------------------------------------------------------------------
# Coverage: what a scaffold split leaves a motif feature to work with
# --------------------------------------------------------------------------------------
def coverage(ds, variant, sets, splits, vocab):
    """How much of a test molecule's motif content the training split had ever seen."""
    train, test = splits["train"], splits["test"]
    seen = set()
    for r in train:
        seen |= sets[r]
    invocab = set(vocab)

    murcko_hits, n_cyclic, brics_frac, no_hit, n_brics = 0, 0, [], 0, []
    for r in test:
        fs = sets[r]
        murcko = {f for f in fs if f.startswith(MURCKO)}
        brics = {f for f in fs if not f.startswith((MURCKO, GENERIC))}
        # Only ring-bearing molecules have a scaffold to have seen or not seen.
        if murcko:
            n_cyclic += 1
            if murcko <= seen:
                murcko_hits += 1
        if brics:
            brics_frac.append(len(brics & seen) / len(brics))
            n_brics.append(len(brics))
        if not (fs & invocab):
            no_hit += 1

    return {
        "dataset": ds,
        "variant": variant,
        "train_molecules": len(train),
        "test_molecules": len(test),
        "vocab_size": len(vocab),
        "test_cyclic_molecules": n_cyclic,
        "test_murcko_scaffold_seen": murcko_hits / max(n_cyclic, 1),
        "test_brics_fragment_seen": float(np.mean(brics_frac)) if brics_frac else np.nan,
        "test_no_vocab_hit": no_hit / max(len(test), 1),
        "mean_brics_per_molecule": float(np.mean(n_brics)) if n_brics else np.nan,
    }


# --------------------------------------------------------------------------------------
# The readout
# --------------------------------------------------------------------------------------
def scorable(y, train, test, cls):
    """
    Task columns that both splits can support, chosen once and used by every arm.

    A task whose training rows are all one class cannot be fitted, and one whose test rows
    are all one class has no AUC. Deciding that per arm would let the arms be scored on
    different tasks and then compared as though they were not.
    """
    if not cls:
        return list(range(y.shape[1]))
    keep = []
    for t in range(y.shape[1]):
        tr = y[train, t]
        te = y[test, t]
        tr = tr[np.isfinite(tr)]
        te = te[np.isfinite(te)]
        if tr.size and te.size and np.unique(tr).size > 1 and np.unique(te).size > 1:
            keep.append(t)
    return keep


def predict(x_train, y_train, x_test, cls):
    """One column of predictions from the fixed-setting linear readout."""
    mask = np.isfinite(y_train)
    xt, yt = x_train[mask], y_train[mask]
    if cls:
        model = LogisticRegression(C=C_LOGREG, max_iter=MAX_ITER, solver="liblinear",
                                   random_state=0)
        model.fit(xt, yt.astype(int))
        return model.predict_proba(x_test)[:, 1]
    model = Ridge(alpha=ALPHA_RIDGE)
    model.fit(xt, yt)
    return model.predict(x_test)


def run_split(ds, variant, sets, min_count):
    """Score all three arms on one dataset and one split. Returns (metric rows, coverage)."""
    cls = is_classification(ds)
    ecfp_z = np.load(os.path.join(POOL_DIR, f"{ds}_ecfp.npz"), allow_pickle=True)
    desc_z = np.load(os.path.join(POOL_DIR, f"{ds}_desc.npz"), allow_pickle=True)
    ecfp = np.asarray(ecfp_z["X"], dtype=np.float32)
    desc_raw = np.asarray(desc_z["X"], dtype=np.float32)
    y = np.asarray(ecfp_z["y"], dtype=np.float64)

    with open(os.path.join(SPLIT_DIR, f"{ds}_{variant}.json")) as f:
        splits = json.load(f)
    train = np.asarray(splits["train"], dtype=int)
    test = np.asarray(splits["test"], dtype=int)

    # Descriptor normalisation from the training rows, via the view's own code.
    enc = DescriptorEncoder.fit(ecfp[train], desc_raw[train], hidden=8)
    desc = enc.normalise(desc_raw).detach().numpy().astype(np.float32)

    vocab = vocabulary(sets, train, min_count)
    motif_train = indicators(sets, train, vocab)
    motif_test = indicators(sets, test, vocab)

    blocks = {
        "desc": (np.hstack([ecfp[train], desc[train]]), np.hstack([ecfp[test], desc[test]])),
        "motif": (motif_train, motif_test),
        "desc+motif": (np.hstack([ecfp[train], desc[train], motif_train]),
                       np.hstack([ecfp[test], desc[test], motif_test])),
    }

    tasks = scorable(y, train, test, cls)
    rows = []
    for arm in ARMS:
        xa, xb = blocks[arm]
        if xa.shape[1] == 0:
            # An empty vocabulary is a real outcome on a small dataset, not a crash.
            rows.append({"dataset": ds, "variant": variant, "arm": arm,
                         "metric": "auc" if cls else "rmse", "value": np.nan,
                         "n_tasks": 0, "n_features": 0})
            continue
        preds = np.column_stack([predict(xa, y[train, t], xb, cls) for t in tasks])
        truth = y[np.ix_(test, tasks)]
        if cls:
            m = cls_metrics(truth, preds)
            value, key = m["auc"], "auc"
        else:
            m = reg_metrics(truth, preds, dataset=ds)
            value, key = m["rmse"], "rmse"
        rows.append({"dataset": ds, "variant": variant, "arm": arm, "metric": key,
                     "value": float(value), "n_tasks": len(tasks),
                     "n_features": int(xa.shape[1])})

    return rows, coverage(ds, variant, sets, splits, vocab)


# --------------------------------------------------------------------------------------
# Paired summary, under the same statistics as every other comparison in the paper
# --------------------------------------------------------------------------------------
def compare(df, a, b, variants):
    """Paired `a` minus `b` across the seeded splits, per dataset, then across datasets."""
    seeded = [v for v in variants if v.startswith("seed")]
    rows = []
    for ds in df.dataset.unique():
        cls = is_classification(ds)
        higher = cls
        sub = df[(df.dataset == ds) & (df.variant.isin(seeded))]
        va, vb = [], []
        for v in seeded:
            xa = sub[(sub.variant == v) & (sub.arm == a)].value
            xb = sub[(sub.variant == v) & (sub.arm == b)].value
            if len(xa) and len(xb) and np.isfinite(xa.iloc[0]) and np.isfinite(xb.iloc[0]):
                va.append(float(xa.iloc[0]))
                vb.append(float(xb.iloc[0]))
        if len(va) < 3:
            continue
        va, vb = np.asarray(va), np.asarray(vb)
        diff = (va - vb) if higher else (vb - va)   # positive = `a` is better
        md, sd, hd, n = ci95(diff)
        p_t = float(sps.ttest_rel(va, vb).pvalue) if not np.allclose(diff, 0) else 1.0
        dz = float(md / sd) if sd > 0 else np.nan
        mde = 0.02 if cls else 0.10
        rows.append({
            "dataset": ds, "metric": "auc" if cls else "rmse", "a": a, "b": b,
            "n_splits": n, "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
            "mean_diff": md, "ci_diff": hd, "a_wins": int((diff > 0).sum()),
            "p_ttest": p_t, "cohens_dz": dz,
            "decisive": bool(abs(md) > hd), "clears_mde": bool(abs(md) > mde),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out, {}
    out["p_holm"] = holm(out.p_ttest.values)
    return out, across_datasets(out.mean_diff.values, out.cohens_dz.values)


def canonical(df, archive=True):
    """
    The same three arms on the DeepChem canonical split, which §3.2 requires reporting.

    A single split carries no interval, so this can only count wins by mean -- and that is
    the point of reporting it beside the seeded table rather than instead of it. It is also
    where the two conventions disagree: the motif arm loses every seeded dataset and wins two
    canonical ones. Leaving it out would be choosing the convention that agreed with us,
    which is the specific failure §3.2 exists to name.
    """
    rows = []
    sub = df[df.variant == "deepchem"]
    for ds in df.dataset.unique():
        d = sub[sub.dataset == ds]
        if d.empty:
            continue
        vals = {a: float(d[d.arm == a].value.iloc[0]) for a in ARMS if len(d[d.arm == a])}
        if len(vals) != len(ARMS):
            continue
        higher = d.metric.iloc[0] == "auc"

        def wins(a, b):
            return bool(vals[a] > vals[b]) if higher else bool(vals[a] < vals[b])

        rows.append({
            "dataset": ds, "metric": d.metric.iloc[0],
            "desc": vals["desc"], "motif": vals["motif"], "desc_motif": vals["desc+motif"],
            "motif_beats_desc": wins("motif", "desc"),
            "both_beats_desc": wins("desc+motif", "desc"),
        })
    out = pd.DataFrame(rows)
    if archive and not out.empty:
        os.makedirs(MET_DIR, exist_ok=True)
        out.to_csv(os.path.join(MET_DIR, "motif_canonical.csv"), index=False)
    return out


def report(df, variants, archive=True):
    """
    Print the two comparisons the reviewer question actually asks about, and archive them.

    Archived as `motif_compare_<a>_vs_<b>.csv`, in the same shape `src/eval/view_stats.py`
    writes, so `scripts/check_paper.py` can read a CSV instead of importing this module and
    recomputing. A checker that re-derives the number it is checking is checking itself.
    """
    for a, b in (("desc+motif", "desc"), ("motif", "desc")):
        cmp_df, ad = compare(df, a, b, variants)
        if cmp_df.empty:
            continue
        if archive:
            out = cmp_df.copy()
            for k, v in ad.items():
                out[f"across_{k}"] = v
            os.makedirs(MET_DIR, exist_ok=True)
            out.to_csv(os.path.join(
                MET_DIR, f"motif_compare_{a.replace('+', '_')}_vs_{b}.csv"), index=False)
        print(f"\n{'=' * 86}")
        print(f"{a}  vs  {b}   (paired across the seeded scaffold splits)")
        print(f"{'=' * 86}")
        show = cmp_df[["dataset", "metric", "mean_a", "mean_b", "mean_diff", "ci_diff",
                       "a_wins", "p_ttest", "p_holm", "decisive", "clears_mde"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        n_clear = int((cmp_df.p_holm < 0.05).sum())
        n_mde = int((cmp_df.decisive & cmp_df.clears_mde).sum())
        print(f"\n  Holm-corrected wins for {a}: "
              f"{int(((cmp_df.p_holm < 0.05) & (cmp_df.mean_diff > 0)).sum())} of "
              f"{len(cmp_df)}   (significant either way: {n_clear})")
        print(f"  Differences that are decisive AND clear the practical threshold: {n_mde}")
        print(f"  Across datasets: favoured on {ad['wins']}/{ad['n_datasets']}, "
              f"median {ad['median_diff']:+.4f}, sign p={ad['p_sign']:.4f}, "
              f"Wilcoxon(dz) p={ad['p_wilcoxon_dz']:.4f}, "
              f"Wilcoxon(raw) p={ad['p_wilcoxon_raw']:.4f}")

    # Section 3.2: report both conventions, never one.
    canon = canonical(df, archive=archive)
    if not canon.empty:
        print(f"\n{'=' * 86}")
        print("CANONICAL DeepChem split (one split, so wins by mean only -- no interval)")
        print(f"{'=' * 86}")
        print(canon.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n  motif beats ECFP+desc on {int(canon.motif_beats_desc.sum())} of {len(canon)}"
              f"   |   adding motifs beats it on {int(canon.both_beats_desc.sum())} of "
              f"{len(canon)}")
        print("  The seeded table above disagrees with both counts. That disagreement is a "
              "result,\n  not a reason to quote whichever convention is kinder.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--min-count", type=int, default=5,
                    help="a fragment must occur in this many TRAINING molecules to be kept")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="re-fragment every molecule instead of reading data/pool/*_motifs.npz")
    ap.add_argument("--out", default=OUT,
                    help="where to write. Defaults to the full-set archive (five "
                         "results/metrics/motif_*.csv files), none of which a run narrowed by "
                         "--datasets or --variants will touch; a narrowed run with --out writes "
                         "the subset and its coverage beside that path instead.")
    args = ap.parse_args()

    everything = dataset_names()
    datasets = args.datasets or everything

    rows, cov = [], []
    for ds in datasets:
        print(f"{ds}")
        sets = motif_lists(ds, refresh=args.refresh_cache)
        for variant in args.variants:
            t0 = time.time()
            r, c = run_split(ds, variant, sets, args.min_count)
            rows.extend(r)
            cov.append(c)
            scores = "  ".join(f"{x['arm']}={x['value']:.4f}" for x in r
                               if np.isfinite(x["value"]))
            print(f"    {variant:<9} {scores}   ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    cov_df = pd.DataFrame(cov)

    print(f"\n{'=' * 86}")
    print("COVERAGE: what fraction of a test molecule's motifs the training split had seen")
    print(f"{'=' * 86}")
    print(cov_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # A narrowed run does NOT overwrite the archive -- the same guard scripts/audit_duplicates.py
    # carries, and for the same reason: a partial re-run once replaced a full result file and
    # the paper disagreed with it until someone noticed.
    #
    # "The archive" is FIVE files, not one: the probe CSV, the coverage CSV, both comparison
    # CSVs and the canonical-split CSV. The first version of this guard protected only the
    # probe CSV. A `--datasets freesolv --out elsewhere.csv` run wrote the subset where it was
    # asked to -- and also cut the other four archives down to FreeSolv alone, which is the
    # exact failure the guard's own comment says it prevents. So a partial run now writes
    # nothing to results/metrics/ at all, whether or not --out is given.
    partial = set(datasets) != set(everything) or set(args.variants) != set(VARIANTS)
    if partial:
        report(df, args.variants, archive=False)
        if args.out == OUT:
            print(f"\nNOT written: this run covers {len(datasets)} of {len(everything)} "
                  f"datasets and {len(args.variants)} of {len(VARIANTS)} splits, and "
                  f"{MET_DIR}/motif_*.csv is the full-set archive that scripts/check_paper.py "
                  f"asserts against.\nRe-run without --datasets/--variants to refresh it, or "
                  f"pass --out to write this subset somewhere else.")
            return
        stem, ext = os.path.splitext(args.out)
        cov_out = f"{stem}_coverage{ext or '.csv'}"
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        df.to_csv(args.out, index=False)
        cov_df.to_csv(cov_out, index=False)
        print(f"\nWrote the subset to {args.out} and {cov_out}.\n"
              f"Nothing under {MET_DIR}/ was touched.")
        return

    report(df, args.variants)

    os.makedirs(MET_DIR, exist_ok=True)
    df.to_csv(args.out, index=False)
    cov_df.to_csv(COVERAGE_OUT, index=False)
    print(f"\nWrote {args.out}\nWrote {COVERAGE_OUT}")


if __name__ == "__main__":
    main()

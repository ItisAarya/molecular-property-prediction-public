"""
src/data/splits.py

Who is allowed to look at which rows.

THE PROBLEM THIS SOLVES
-----------------------
The inherited pipeline used one validation split for six different jobs: early stopping,
fitting the stacking meta-learner, choosing ensemble weights, fitting the probability
calibrators, tuning per-task decision thresholds, and reporting "validation" performance.

Fitting something on a set of rows and then scoring it on those same rows measures how
well it memorised them, not how well it generalises. BBBP showed the symptom plainly:
validation AUC 0.99 against test AUC 0.75.

THE PROTOCOL WE USE
-------------------
Out-of-fold (OOF) prediction over the training set, nested twice:

    train  --5-fold-->  oof_base   base predictions for every training molecule, each
                                   made by a model that never trained on it
    oof_base --5-fold-->  oof_meta meta-learner predictions, same guarantee
    oof_meta ---------->  calibrators and decision thresholds are fitted here
    valid  ------------>  early stopping and model selection only; never fitted on
    test   ------------>  evaluated once, at the very end

`kfold_indices` provides the folds. The nesting matters: the meta-learner consumes base
predictions, so it needs base predictions that are out-of-sample; the calibrators consume
meta predictions, so they need meta predictions that are out-of-sample. Skipping the
second level would just move the leak up a layer.

WHY NOT SIMPLY SPLIT THE VALIDATION SET
---------------------------------------
That was the first design, and the numbers ruled it out. Partitioning validation three
ways leaves Tox21 with ~235 rows per part and ClinTox with ~45. ClinTox's CT_TOX task
carries only about 7 positives in the whole validation split, so a part holds two or three
-- a calibrator fitted on two positive examples is noise, not calibration. Several Tox21
assays land in the same position.

OOF over `train` gives the meta level the entire training set instead: ~95 CT_TOX
positives rather than 2, and 6258 Tox21 molecules rather than 235. The cost is K times the
base-model training, which is affordable here only because the transformer encoder is
frozen and its embeddings are cached (see scripts/cache_embeddings.py).

The validation-partition helpers below (`partition_indices`, `valid_partition`,
`enough_positives`) are kept: they are the fallback for any future dataset too small for
K-fold, and `enough_positives` is the guard every consumer should use before fitting a
per-task calibrator or threshold.

STRATIFICATION AND DETERMINISM
------------------------------
Folds and parts are stratified on how many positive labels each molecule carries, because
Tox21 tasks run as low as 2.5% positives and an unstratified fold could leave a task with
no positives at all. Regression targets are binned by quantile instead.

Everything is deterministic and derived from the dataset name, so every stage is fitted on
exactly the same rows on every run and on every machine.
"""

import hashlib
import os

import numpy as np

DATA_DIR = "data"

# meta / cal / sel. The meta-learner gets the largest share because it fits the most
# parameters; selection needs only enough rows to rank a handful of models.
DEFAULT_FRACTIONS = (0.40, 0.30, 0.30)

PARTS = ("meta", "cal", "sel")

# Below this many positives, a per-task calibrator or threshold is not worth fitting.
MIN_POSITIVES = 10


def _dataset_seed(dataset, base_seed):
    """
    Stable per-dataset seed.

    Hashing the name gives each dataset a different partition, so a quirk of one split
    does not repeat everywhere, while staying reproducible across runs and machines.
    The built-in hash() is randomised per process, hence md5.
    """
    digest = hashlib.md5(dataset.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + base_seed) % (2**31)


def _strata(y, n_bins=4):
    """
    Stratum label per row, so every part gets a similar label mix.

    Classification: the number of positive labels a molecule carries (capped), which stops
    rare actives from clustering in one part. Regression: quantile bins of the target.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    labelled = y[~np.isnan(y)]
    is_binary = labelled.size > 0 and np.all(np.isin(labelled, [0.0, 1.0]))

    if is_binary:
        with np.errstate(invalid="ignore"):
            pos = np.nansum(y == 1.0, axis=1)
        return np.minimum(pos, 3).astype(int)

    col = y[:, 0]
    filled = np.where(np.isnan(col), np.nanmedian(col), col)
    edges = np.unique(np.quantile(filled, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.digitize(filled, edges)


def partition_indices(dataset, y, fractions=DEFAULT_FRACTIONS, base_seed=0):
    """
    Split the validation rows into the meta / cal / sel parts.

    `y` is the validation label matrix, used only for stratification. Returns a dict of
    sorted index arrays that together cover every row exactly once.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    n = len(y)
    rng = np.random.default_rng(_dataset_seed(dataset, base_seed))
    strata = _strata(y)

    buckets = {p: [] for p in PARTS}
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        rng.shuffle(members)
        # Cut points from the cumulative fractions, so each stratum is divided in the
        # same proportions and no part is systematically starved.
        cuts = np.round(np.cumsum(fractions)[:-1] * len(members)).astype(int)
        for part, chunk in zip(PARTS, np.split(members, cuts)):
            buckets[part].append(chunk)

    out = {}
    for p in PARTS:
        idx = np.concatenate(buckets[p]) if buckets[p] else np.array([], dtype=int)
        out[p] = np.sort(idx.astype(int))

    covered = np.sort(np.concatenate([out[p] for p in PARTS]))
    assert np.array_equal(covered, np.arange(n)), "partition must cover every row exactly once"
    return out


def load_valid_y(dataset):
    """Load the validation label matrix (NaN = never measured)."""
    y = np.load(os.path.join(DATA_DIR, f"{dataset}_valid_ecfp.npz"))["y"].astype(np.float32)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def valid_partition(dataset, fractions=DEFAULT_FRACTIONS, base_seed=0):
    """Convenience wrapper: load the validation labels and return the index dict."""
    return partition_indices(dataset, load_valid_y(dataset), fractions, base_seed)


def enough_positives(y_part, task, minimum=MIN_POSITIVES):
    """
    Is this task worth fitting a calibrator or threshold on, in this part?

    Requires both classes present and at least `minimum` positives. A caller that gets
    False must fall back to a documented default and record that it did.
    """
    col = np.asarray(y_part)[:, task]
    col = col[~np.isnan(col)]
    if col.size == 0 or len(np.unique(col)) < 2:
        return False
    return int((col == 1).sum()) >= minimum


def kfold_indices(dataset, y, n_folds=5, base_seed=0):
    """
    Stratified, deterministic K-fold partition of a split (used on `train`).

    Returns a list of index arrays, one per fold, covering every row exactly once.

    This is what makes correct stacking possible. A meta-learner must be fitted on base
    predictions for molecules the base models did not train on; otherwise it learns to
    trust predictions that are in-sample and over-confident. Training on fold k-complement
    and predicting on fold k, for every k, produces exactly that: an out-of-fold prediction
    for every training molecule.

    Stratification matters more here than usual. Tox21 tasks run as low as 2.5% positives,
    so an unstratified fold can easily end up with a task that has no positives at all,
    which makes that task unfittable for that fold.
    """
    n = len(y)
    rng = np.random.default_rng(_dataset_seed(dataset, base_seed + 977))
    strata = _strata(y)

    buckets = [[] for _ in range(n_folds)]
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        rng.shuffle(members)
        # Deal the stratum round-robin across folds so each fold gets a near-equal share
        # of it, rather than slicing (which biases the last fold when sizes do not divide).
        for i, row in enumerate(members):
            buckets[i % n_folds].append(row)

    folds = [np.sort(np.array(b, dtype=int)) for b in buckets]
    covered = np.sort(np.concatenate(folds))
    assert np.array_equal(covered, np.arange(n)), "folds must cover every row exactly once"
    return folds


def load_y(dataset, split):
    """Load any split's label matrix (NaN = never measured)."""
    y = np.load(os.path.join(DATA_DIR, f"{dataset}_{split}_ecfp.npz"))["y"].astype(np.float32)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def load_smiles(dataset, split):
    """Load the SMILES strings for a split."""
    d = np.load(os.path.join(DATA_DIR, f"{dataset}_{split}_ecfp.npz"))
    return [str(x) for x in d["smiles"]]


def _murcko_scaffold(smiles):
    """Bemis-Murcko scaffold as a canonical SMILES string; empty for unparseable input."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def scaffold_kfold_indices(dataset, smiles, n_folds=5):
    """
    Scaffold-disjoint K-fold: molecules sharing a Bemis-Murcko scaffold always land in the
    same fold. Returns a list of index arrays covering every row exactly once.

    WHY THIS RATHER THAN RANDOM FOLDS
    ---------------------------------
    These benchmarks use a scaffold split, so `test` contains structurally different
    molecules from `train` on purpose -- that is the point of the benchmark. With random
    K-fold, a held-out molecule usually has close relatives in the folds the model trained
    on, so out-of-fold predictions come out easier than test predictions.

    Measured on ESOL: random-fold OOF RMSE was 1.15 logS against a test RMSE of 1.69 -- the
    meta-learner would have been fitted on predictions that were far better than the ones
    it meets at test time, and would trust them too much.

    Keeping whole scaffold groups inside a fold reproduces the train/test relationship
    inside the training set, so the meta level is fitted under the distribution shift it
    actually faces.

    Groups are assigned largest-first to whichever fold is currently smallest, which is the
    standard greedy balance. Assignment is deterministic: no RNG, and ties are broken by
    the lowest row index.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    singletons = []
    for i, smi in enumerate(smiles):
        scaf = _murcko_scaffold(smi)
        if scaf == "":
            # An acyclic molecule has no ring system, so Murcko returns an empty string.
            # Grouping all of those together would be wrong twice over: they share no
            # scaffold, so there is nothing to leak, and on a dataset like ESOL they are
            # numerous enough to swamp one fold (measured: 317 vs 110 rows). Treat each as
            # its own group so they spread evenly.
            singletons.append([i])
        else:
            groups[scaf].append(i)

    # Largest groups first, so the big scaffolds are placed while folds are still empty
    # enough to balance around them; singletons last, as filler that evens the sizes out.
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0])) + singletons

    folds = [[] for _ in range(n_folds)]
    for g in ordered:
        smallest = min(range(n_folds), key=lambda k: (len(folds[k]), k))
        folds[smallest].extend(g)

    out = [np.sort(np.array(f, dtype=int)) for f in folds]
    covered = np.sort(np.concatenate(out))
    assert np.array_equal(covered, np.arange(len(smiles))), "folds must cover every row once"
    return out


def describe(dataset, fractions=DEFAULT_FRACTIONS, base_seed=0):
    """Human-readable summary, for logging and for the verification script."""
    y = load_valid_y(dataset)
    parts = partition_indices(dataset, y, fractions, base_seed)

    sizes = ", ".join(f"{p}={len(parts[p])}" for p in PARTS)
    lines = [f"{dataset}: {len(y)} validation rows -> {sizes}"]

    labelled = y[~np.isnan(y)]
    if labelled.size and np.all(np.isin(labelled, [0.0, 1.0])):
        for t in range(y.shape[1]):
            counts = []
            for p in PARTS:
                col = y[parts[p], t]
                n_pos = int(np.nansum(col == 1))
                flag = "" if enough_positives(y[parts[p]], t) else "!"
                counts.append(f"{p}={n_pos}{flag}")
            lines.append(f"    task {t:<2} positives: " + ", ".join(counts))
        lines.append("    ('!' marks too few positives to fit on; consumer uses a default)")
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    for ds in meta:
        print(describe(ds))
        print()

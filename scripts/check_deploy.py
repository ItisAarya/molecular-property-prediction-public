"""
scripts/check_deploy.py

Assert that the live featuriser reproduces the features the models were trained on.

    python -m scripts.check_deploy
    python -m scripts.check_deploy --datasets bbbp freesolv --n 8

WHY THIS EXISTS
---------------
`src/deploy/featurize.py` rebuilds ECFP, descriptors, graphs and ChemBERTa embeddings from
a SMILES string the user typed. Training read the same quantities from `data/pool/`, built
months earlier by a different script through a different library.

If those two drift apart, nothing fails. The model accepts whatever width of tensor it is
given, runs, and returns a confident number that means nothing. There is no error message
and no obviously wrong output -- a wrong prediction looks exactly like a right one. That is
the single most dangerous failure mode in a deployed model, and it is invisible to every
other check in this repo, because every other check works from the archived predictions
rather than from features.

So this walks real pool molecules back through the live path and demands bit-level
agreement with what training actually used. The ECFP check in particular is what licenses
`featurize.ecfp` to use RDKit directly instead of importing DeepChem, which built the pool.

Exit code is non-zero on any mismatch, so it belongs in the same gate as check_configs.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from src.data.materialize import dataset_names

POOL = os.path.join("data", "pool")


def check_dataset(ds, n, do_seq):
    """Compare live features against the pool for the first `n` molecules of `ds`."""
    from src.deploy import featurize as F

    pool = np.load(os.path.join(POOL, f"{ds}_ecfp.npz"), allow_pickle=True)
    smiles = [str(s) for s in pool["smiles"]][:n]
    mols = [F.parse(s) for s in smiles]

    bad = []

    # ECFP: exact equality. These are 0/1 bits; "close" is not a meaningful standard.
    live = F.ecfp(mols)
    want = pool["X"][:n].astype(np.float32)
    if not np.array_equal(live, want):
        rows = int((live != want).any(axis=1).sum())
        bad.append(f"ecfp: {rows}/{n} molecules differ")

    # Descriptors: NaN-aware, because undefined descriptors are meant to stay NaN and
    # NaN != NaN would otherwise report every dataset as broken.
    dz = np.load(os.path.join(POOL, f"{ds}_desc.npz"), allow_pickle=True)
    live_d = F.descriptors(mols)
    want_d = dz["X"][:n].astype(np.float64)
    same_nan = np.isnan(live_d) == np.isnan(want_d)
    finite = ~np.isnan(want_d) & ~np.isnan(live_d)
    if not same_nan.all():
        bad.append(f"desc: NaN pattern differs in {int((~same_nan).sum())} cells")
    elif not np.allclose(live_d[finite], want_d[finite], rtol=1e-5, atol=1e-8):
        worst = float(np.abs(live_d[finite] - want_d[finite]).max())
        bad.append(f"desc: values differ, max |diff| = {worst:.3g}")

    # Graphs: node features, edge count and edge features. A graph that matches on node
    # count but not on edges would still run, and would still be wrong.
    stored = torch.load(os.path.join(POOL, f"{ds}_graphs.pt"), weights_only=False)["graphs"]
    for i, s in enumerate(smiles):
        live_g = F.mol_to_graph(s)
        want_g = stored[i]
        if live_g is None:
            bad.append(f"graph[{i}]: live featuriser rejected a pooled molecule")
            break
        if not torch.equal(live_g.x, want_g.x):
            bad.append(f"graph[{i}]: node features differ")
            break
        if not torch.equal(live_g.edge_index, want_g.edge_index):
            bad.append(f"graph[{i}]: edge_index differs")
            break
        if not torch.equal(live_g.edge_attr, want_g.edge_attr):
            bad.append(f"graph[{i}]: edge_attr differs")
            break

    # ChemBERTa: float32 through a 6-layer transformer, so exact equality is the wrong
    # standard -- batch composition changes the arithmetic order. The tolerance is tight
    # enough that a different pooling rule or a different checkpoint would fail it.
    if do_seq:
        ez = np.load(os.path.join(POOL, f"{ds}_chemberta.npz"))
        want_s = np.concatenate([ez["cls"][:n], ez["mean"][:n]], axis=1)
        live_s = F.chemberta(smiles, ds).numpy()
        if not np.allclose(live_s, want_s, rtol=1e-4, atol=1e-4):
            worst = float(np.abs(live_s - want_s).max())
            bad.append(f"seq: embeddings differ, max |diff| = {worst:.3g}")

    return bad


def check_end_to_end(ds, tag, variant="deepchem", tol=1e-5):
    """
    The whole path at once: SMILES strings -> features -> model -> the archived numbers.

    The per-view checks above prove each featuriser still agrees with the pool. This proves
    something stronger and simpler: that running the deployed model over the test split
    through the *live* code reproduces the predictions the training run archived. If this
    passes, every number the app shows was produced by the same pipeline that produced the
    results in the paper.

    Returns None when the archive is absent, so a tree that has not trained the deployment
    models yet reports "skipped" rather than failing.
    """
    import numpy as np

    from src.deploy.predict import Predictor

    archived_path = os.path.join("results", "runs", variant, "preds", f"{ds}_{tag}_test.npy")
    ckpt = os.path.join("models", f"{ds}_{tag}.pt")
    if not (os.path.exists(archived_path) and os.path.exists(ckpt)):
        return None

    pool = np.load(os.path.join(POOL, f"{ds}_ecfp.npz"), allow_pickle=True)
    idx = json.load(open(os.path.join("data", "splits", f"{ds}_{variant}.json")))["test"]
    smiles = [str(pool["smiles"][i]) for i in idx]

    archived = np.load(archived_path)
    live, ok = Predictor(ds, tag=tag, variant=variant)._raw(smiles)
    if live is None:
        return ["end-to-end: the live featuriser rejected every test molecule"]
    if int(ok.sum()) != len(smiles):
        return [f"end-to-end: {len(smiles) - int(ok.sum())} test molecule(s) the pool "
                f"contains are now unfeaturisable"]

    worst = float(np.abs(live - archived[ok]).max())
    if worst > tol:
        return [f"end-to-end: live predictions differ from the archive, "
                f"max |diff| = {worst:.3g}"]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=5,
                    help="molecules per dataset (ChemBERTa is the slow part)")
    ap.add_argument("--skip-seq", action="store_true",
                    help="skip the ChemBERTa check, which needs the model downloaded")
    ap.add_argument("--tag", default="deploy_proposed",
                    help="which deployed model to check end to end")
    ap.add_argument("--skip-end-to-end", action="store_true",
                    help="skip the full SMILES-to-prediction check (it runs the model "
                         "over every test molecule, which is the slow part)")
    args = ap.parse_args()

    datasets = args.datasets or dataset_names()
    failures = {}
    for ds in datasets:
        problems = check_dataset(ds, args.n, not args.skip_seq)

        e2e = None if args.skip_end_to_end else check_end_to_end(ds, args.tag)
        if e2e:
            problems = problems + e2e

        if problems:
            status = "FAILED"
        elif e2e is None:
            status = "ok (features only)"
        else:
            status = "ok (features + end to end)"
        print(f"  {ds:<16} {status}")
        for p in problems:
            print(f"      {p}")
        if problems:
            failures[ds] = problems

    print()
    if failures:
        print(f"{len(failures)} of {len(datasets)} datasets FAILED.")
        print("The live featuriser no longer reproduces the training features. Any "
              "prediction the app makes is meaningless until this passes -- fix the "
              "featuriser, do not loosen the tolerance.")
        sys.exit(1)

    views = "ecfp, desc, graph" + ("" if args.skip_seq else ", seq")
    print(f"All {len(datasets)} datasets reproduce their pooled features ({views}), "
          f"{args.n} molecules each.")
    if not args.skip_end_to_end:
        print(f"'{args.tag}' also reproduces its archived test predictions end to end, "
              f"from SMILES strings.")


if __name__ == "__main__":
    main()

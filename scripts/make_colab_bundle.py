"""
scripts/make_colab_bundle.py

Pack everything a Colab session needs to run the sequence view, and nothing else.

    python -m scripts.make_colab_bundle

Writes `mpp_colab_bundle.zip` in the project root. Upload that one file to Google Drive
and run `notebooks/phase1c_lora_colab.ipynb`.

WHY A BUNDLE RATHER THAN A GIT CLONE
------------------------------------
The repository is private, so cloning it on Colab means putting a GitHub token into a
notebook cell -- a credential in a file that syncs to Drive and is easy to share by
accident. A zip avoids that entirely.

WHY IT IS SMALL
---------------
The sequence view reads tokenised SMILES, and the labels and scaling constants that live
alongside the fingerprints. It never touches the molecular graphs or the cached ChemBERTa
embeddings, which are the bulk of `data/pool/`. Materialising a split with
`--artifacts tok ecfp` skips them, so the bundle carries roughly 50 MB instead of several
hundred.

The splits themselves are index arrays, so all six variants cost a few hundred kilobytes:
the same molecules, partitioned six ways.
"""

import argparse
import fnmatch
import os
import zipfile

BUNDLE = "mpp_colab_bundle.zip"

# Which pooled artifacts each kind of run actually reads. Carrying only these is what
# keeps the sequence-only bundle at ~4 MB instead of ~280 MB.
PRESETS = {
    # Phase 1c: the sequence views. Tokens for the input, fingerprints for the labels.
    "sequence": ["*_ecfp.npz", "*_tok.pt"],
    # Phase 2 end to end: adds molecular graphs and descriptors. No cached ChemBERTa
    # embeddings -- with --seq lora the encoder runs for real, so the cache is unused.
    "fusion": ["*_ecfp.npz", "*_tok.pt", "*_graphs.pt", "*_desc.npz"],
    # Phase 2 with frozen views: the cache replaces the tokens.
    "fusion-cached": ["*_ecfp.npz", "*_chemberta.npz", "*_graphs.pt", "*_desc.npz"],
    "all": ["*_ecfp.npz", "*_tok.pt", "*_graphs.pt", "*_desc.npz", "*_chemberta.npz"],
}

# Code the run imports, plus the data it actually reads.
CODE_DIRS = ["src", "scripts"]
CODE_SKIP = ["*__pycache__*", "*.pyc"]
DATA_FILES = [
    ("data/dataset_meta.json", True),
    ("data/pool/pool_index.json", True),
]



def code_files():
    for root_dir in CODE_DIRS:
        for root, _, files in os.walk(root_dir):
            for f in files:
                path = os.path.join(root, f)
                if any(fnmatch.fnmatch(path, pat) for pat in CODE_SKIP):
                    continue
                yield path


def data_files(pool_patterns):
    for path, required in DATA_FILES:
        if os.path.exists(path):
            yield path
        elif required:
            raise SystemExit(f"missing {path} -- run the prep pipeline first")

    for f in sorted(os.listdir("data/pool")):
        if any(fnmatch.fnmatch(f, pat) for pat in pool_patterns):
            yield os.path.join("data/pool", f)

    for f in sorted(os.listdir("data/splits")):
        if f.endswith(".json"):
            yield os.path.join("data/splits", f)


def main():
    ap = argparse.ArgumentParser(description="Pack a Colab bundle for one kind of run.")
    ap.add_argument("--preset", default="sequence", choices=list(PRESETS),
                    help="which pooled artifacts to include (default: sequence)")
    ap.add_argument("--out", default=BUNDLE)
    args = ap.parse_args()

    paths = list(code_files()) + list(data_files(PRESETS[args.preset]))

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.replace(os.sep, "/"))

    size = os.path.getsize(args.out) / 1e6
    n_code = sum(1 for p in paths if p.startswith(("src", "scripts")))
    print(f"Wrote {args.out}  [preset: {args.preset}]  ({size:.0f} MB, {len(paths)} files: "
          f"{n_code} code, {len(paths) - n_code} data)")
    print("\nNext:")
    print("  1. Upload this file to Google Drive (anywhere; the notebook will ask where).")
    print("  2. Open notebooks/phase1c_lora_colab.ipynb in Colab and run the cells in order.")


if __name__ == "__main__":
    main()

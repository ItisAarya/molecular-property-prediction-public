"""
scripts/merge_colab_results.py

Merge a results archive downloaded from Colab into results/runs/, safely.

    python -m scripts.merge_colab_results ~/Downloads/phase1c_results.zip
    python -m scripts.merge_colab_results ~/Downloads/phase1c_results.zip --dry-run

WHY THIS EXISTS
---------------
Unpacking the archive by hand is easy to get wrong, and the failure is quiet. Extracting
it into `results/runs/` with the wrong root nests the local working directories
(`results/metrics`, `results/preds`, `results/figs`) inside `results/runs/` instead, which
looks like every tracked result file has been deleted. That already happened once.

So this script never extracts in place. It unpacks to a temporary directory, looks only
for split-variant folders it recognises, copies just their `metrics/` and `preds/`
contents, and leaves everything else in the archive alone. Anything it does not recognise
is reported rather than moved.
"""

import argparse
import os
import shutil
import tempfile
import zipfile

RUNS_DIR = os.path.join("results", "runs")
COPY_SUBDIRS = ("metrics", "preds")


def variant_dirs(root):
    """Find split-variant folders in the extracted archive, at either nesting depth."""
    found = {}
    for base, dirs, _ in os.walk(root):
        for d in dirs:
            if d == "deepchem" or (d.startswith("seed") and d[4:].isdigit()):
                path = os.path.join(base, d)
                if any(os.path.isdir(os.path.join(path, s)) for s in COPY_SUBDIRS):
                    found[d] = path
    return found


def merge(zip_path, dry_run=False):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)

        variants = variant_dirs(tmp)
        if not variants:
            raise SystemExit(
                f"No split-variant folders found in {zip_path}. Expected directories "
                "named 'deepchem' or 'seed0'..'seed4' containing metrics/ or preds/."
            )

        total = 0
        for name in sorted(variants):
            src = variants[name]
            counts = []
            for sub in COPY_SUBDIRS:
                s = os.path.join(src, sub)
                if not os.path.isdir(s):
                    continue
                d = os.path.join(RUNS_DIR, name, sub)
                files = sorted(os.listdir(s))
                if not dry_run:
                    os.makedirs(d, exist_ok=True)
                    for f in files:
                        shutil.copy2(os.path.join(s, f), os.path.join(d, f))
                counts.append(f"{len(files)} {sub}")
                total += len(files)
            print(f"  {name:<10} {', '.join(counts)}")

        verb = "would copy" if dry_run else "copied"
        print(f"\n{verb} {total} files into {RUNS_DIR}/")
        if dry_run:
            print("Re-run without --dry-run to apply.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[2])
    ap.add_argument("zip_path", help="the archive downloaded from Colab")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be copied without writing anything")
    args = ap.parse_args()

    # Expand ~ ourselves. Neither cmd.exe nor PowerShell expands it in an argument to a
    # native command, so it arrives here as a literal directory name and the file is
    # reported missing when it is sitting right there.
    args.zip_path = os.path.expanduser(args.zip_path)

    if not os.path.exists(args.zip_path):
        hint = ""
        folder = os.path.dirname(args.zip_path) or "."
        if os.path.isdir(folder):
            near = sorted(f for f in os.listdir(folder) if f.endswith(".zip"))
            if near:
                hint = (chr(10) + chr(10) + "Zip files in that folder:" + chr(10)
                        + chr(10).join("  " + f for f in near))
        raise SystemExit(f"Not found: {args.zip_path}{hint}")

    if not os.path.isdir(RUNS_DIR):
        raise SystemExit(f"{RUNS_DIR} does not exist -- run this from the project root.")

    merge(args.zip_path, args.dry_run)


if __name__ == "__main__":
    main()

"""
scripts/verify_prep.py

Sanity checks on the files written by scripts/prep_moleculenet.py.

Run after regenerating data:
    python -m scripts.verify_prep

Every check must print PASS. This exists so that a data-prep regression is caught
immediately rather than showing up later as a mysteriously wrong metric.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = "data"
SPLITS = ["train", "valid", "test"]

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def main():
    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))

    for ds, info in meta.items():
        tasks = info["tasks"]
        n_tasks = len(tasks)
        y_mean = np.asarray(info["y_mean"], dtype=np.float32)
        y_std = np.asarray(info["y_std"], dtype=np.float32)
        is_cls = info["task_type"] == "classification"

        print(f"\n=== {ds}  ({info['task_type']}, {n_tasks} task(s)) ===")

        for split in SPLITS:
            d = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"))
            X, y, y_raw, w = d["X"], d["y"], d["y_raw"], d["w"]
            smiles = d["smiles"]
            n = X.shape[0]

            print(f"  -- {split} (n={n}) --")

            # 1. Shapes line up with each other and with the metadata.
            check(
                "shapes consistent",
                y.shape == (n, n_tasks) and w.shape == (n, n_tasks)
                and y_raw.shape == (n, n_tasks) and len(smiles) == n,
                f"X{X.shape} y{y.shape} w{w.shape}",
            )
            check("size matches metadata", n == info["sizes"][split])

            # 2. THE CORE FIX: a NaN label appears exactly where the mask says missing.
            nan_mask = np.isnan(y)
            check(
                "NaN in y occurs exactly where w == 0",
                np.array_equal(nan_mask, w == 0),
                f"{int(nan_mask.sum())} NaN / {int((w == 0).sum())} masked",
            )

            # 3. The mask really is binary (balancing weights were dropped).
            check("w is binary 0/1", np.all(np.isin(w, [0.0, 1.0])))

            # 4. Features must never be NaN — only labels can be missing.
            check("no NaN in X", not np.isnan(X).any())

            # 5. Classification labels are 0/1 (ignoring the missing entries).
            if is_cls:
                present = y[~nan_mask]
                check(
                    "present labels are 0/1",
                    np.all(np.isin(present, [0.0, 1.0])),
                    f"uniques={np.unique(present)}",
                )

            # 6. y_raw round-trips: y_raw == y * std + mean, wherever a label exists.
            expected = y * y_std[None, :] + y_mean[None, :]
            both = ~nan_mask
            check(
                "y_raw == y * y_std + y_mean",
                np.allclose(y_raw[both], expected[both], atol=1e-4),
            )

            # 7. The CSV agrees with the NPZ (same molecules, same missing pattern).
            df = pd.read_csv(os.path.join(DATA_DIR, f"{ds}_{split}.csv"))
            check("csv row count matches", len(df) == n)
            check("csv task columns match", list(df.columns) == ["smiles"] + tasks)
            check(
                "csv missing pattern matches w",
                np.array_equal(df[tasks].isna().values, w == 0),
                f"{int(df[tasks].isna().values.sum())} empty cells",
            )

            # 8. Report the label statistics that the metrics now see.
            if is_cls:
                rates = []
                for t in range(n_tasks):
                    col = y[:, t]
                    col = col[~np.isnan(col)]
                    rates.append(f"{col.mean() * 100:.1f}%" if col.size else "n/a")
                print(f"      positive rate per task (measured labels only): {', '.join(rates)}")
            else:
                col = y_raw[~np.isnan(y_raw)]
                print(
                    f"      y_raw: min={col.min():.2f} max={col.max():.2f} "
                    f"mean={col.mean():.2f} std={col.std():.2f}  (chemical units)"
                )

    print("\n" + "=" * 70)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()

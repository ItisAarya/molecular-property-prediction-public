"""
scripts/dataset_select.py

Shared handling of the `--datasets` flag.

Every stage of the prep pipeline discovers its work by iterating over whatever
datasets it finds in `data/dataset_meta.json` (or `data/pool/pool_index.json`).
Processing everything is the right default, but it makes *adding* a dataset
needlessly expensive and, more importantly, unsafe: re-running a stage across the
datasets that were already prepared regenerates files that finished results depend
on. If anything in the environment has drifted since -- a library version, a cached
download, a featuriser default -- the reported numbers quietly stop matching the
data they were computed from, and nothing announces it.

So every stage accepts an optional `--datasets`. Naming a subset processes only
those and leaves every other dataset's files byte-for-byte as they were.

Usage inside a stage:

    ap = argparse.ArgumentParser()
    add_datasets_arg(ap)
    args = ap.parse_args()
    ...
    for ds in resolve(args.datasets, meta):
        ...
"""


def add_datasets_arg(parser):
    """Attach the standard `--datasets` option to an existing argument parser."""
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DS",
        help="Process only these datasets. Default: every dataset found on disk. "
             "Naming a subset leaves the other datasets' files untouched.",
    )
    return parser


def resolve(requested, available):
    """
    Return the datasets to process, in `available`'s own order.

    `requested` is whatever came off the command line (None = all).
    `available` is any iterable of dataset names -- a dict of metadata works, since
    iterating a dict yields its keys.

    Raises SystemExit with a readable message if a requested dataset is not on disk,
    which is nearly always a typo or a missing earlier stage.
    """
    names = list(available)
    if requested is None:
        return names

    unknown = [d for d in requested if d not in names]
    if unknown:
        raise SystemExit(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(names)}. "
            "If this is a new dataset, run the earlier prep stages for it first."
        )

    selected = [d for d in names if d in requested]
    skipped = [d for d in names if d not in requested]
    if skipped:
        print(f"Processing {len(selected)} dataset(s): {', '.join(selected)}")
        print(f"Leaving untouched: {', '.join(skipped)}")
    return selected

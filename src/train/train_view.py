"""
src/train/train_view.py

Train a single-view encoder standalone and write predictions in the pipeline's format.

    python -m src.train.train_view --encoder gine
    python -m src.train.train_view --encoder gin --tag gin_ref   # the inherited reference
    python -m src.train.train_view --encoder desc --tag desc     # ECFP + 2-D descriptors
    python -m src.train.train_view --encoder lora --tag lora     # ChemBERTa + LoRA
    python -m src.train.train_view --encoder seq_frozen --tag seq_frozen   # frozen control

Writes results/preds/<ds>_<tag>_{valid,test}.npy and results/metrics/<ds>_<tag>_*.csv, so a
new view drops straight into the existing stacking, calibration and reporting stages
without any of them needing to know it exists.

WHY A SHARED TRAINER
--------------------
Every single-view baseline is a row the fusion model will be compared against in Phase 2.
If those rows came from different training recipes -- different early stopping, different
class weighting -- the comparison would measure the recipes rather than the
representations. So all views train through this one loop with one objective, and the only
thing that varies is the encoder.

WHAT IS HELD FIXED FROM PHASE 0
-------------------------------
Masked losses (Tox21's unmeasured labels are NaN and must never contribute), per-task
`pos_weight` computed over measured labels only, seeding via `src.utils.seed`, and early
stopping on validation -- which is legitimate because Phase 0 moved every *fitted* stage
off the validation split, leaving it free for exactly this.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TensorLoader, Dataset as TorchDataset, TensorDataset

from src.data.bucketing import LengthBucketSampler, make_token_collate
from src.eval.metrics import is_classification
from src.models.encoders.descriptor import DescriptorEncoder
from src.models.encoders.sequence import SequenceEncoder

# torch_geometric is imported inside the graph branch of `build_view`, not here. It is a
# heavy dependency that only the graph views need, and requiring it at import time would
# stop the sequence and descriptor views running anywhere it is not installed -- which is
# exactly the case on a stock Colab runtime.
from src.models.heads import EMBED_DIM, SingleViewModel
from src.train.loop import MET_DIR, fit_and_score
from src.utils.seed import set_seed

DATA_DIR = "data"
SPLITS = ("train", "valid", "test")

# Which representation each encoder reads. Adding a view means adding it here and a
# branch in `build_view`; the training loop below never changes.
GRAPH_ENCODERS = ("gine", "gin", "attentivefp")
DESCRIPTOR_ENCODERS = ("desc",)
SEQUENCE_ENCODERS = ("lora", "seq_frozen")


def pick_device(name="auto"):
    """'auto' uses the GPU when torch can actually see one, otherwise the CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_graphs(ds, split):
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_graphs.pt"), weights_only=False)
    return obj["graphs"]


def labels_of(graphs):
    return torch.stack([g.y for g in graphs])


def load_descriptors(ds, split):
    """ECFP fingerprint, raw 2-D descriptors and labels for one split."""
    ecfp = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"), allow_pickle=True)
    desc_path = os.path.join(DATA_DIR, f"{ds}_{split}_desc.npz")
    if not os.path.exists(desc_path):
        raise FileNotFoundError(
            f"{desc_path} not found -- run `python -m scripts.make_descriptors`, then "
            "re-materialise the active split so the per-split copy exists."
        )
    desc = np.load(desc_path, allow_pickle=True)
    return (
        torch.tensor(ecfp["X"], dtype=torch.float32),
        torch.tensor(desc["X"], dtype=torch.float32),
        torch.tensor(ecfp["y"], dtype=torch.float32),
    )


def load_tokens(ds, split):
    """ChemBERTa token ids, attention mask and labels for one split."""
    obj = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_tok.pt"), weights_only=False)
    return {
        "input_ids": obj["input_ids"],
        "attention_mask": obj["attention_mask"],
        "y": obj["y"].float(),
    }


class _IndexDataset(TorchDataset):
    """
    A dataset of row numbers.

    The batch sampler decides which rows go together and the collate function does the
    gathering, so the dataset itself only has to hand out indices. This avoids copying
    each row individually before the batch is assembled.
    """

    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def split_batch(batch):
    """
    Return (encoder input, labels, row indices) for a batch of any representation.

    A PyG batch of graphs carries its labels inside the graph objects; a tensor batch
    yields them as named positions. The training loop should not have to know which
    representation it is looking at, so the difference is confined to here.

    The row indices matter for the sequence view: length bucketing permutes the order
    molecules are visited in, and predictions are saved as arrays that must line up
    row-for-row with the labels. Graph batches keep dataset order and report None.
    """
    if hasattr(batch, "to_data_list"):
        return batch, torch.stack([g.y for g in batch.to_data_list()]), None
    *inputs, y, idx = batch
    return tuple(inputs), y, idx.numpy()


def _drop_last(n, batch_size):
    """
    Whether the training loader should discard a trailing batch of one.

    Both encoders use BatchNorm, which cannot compute a batch variance from a single
    sample and raises at training time. This only ever drops one molecule, and only when
    the split size happens to leave a remainder of exactly one.
    """
    return n % batch_size == 1


def build_view(encoder_name, ds, batch_size, enc_kwargs, seed=0):
    """
    Load one representation and build the encoder that reads it.

    Returns (loaders, labels per split, encoder). Keeping this separate from the training
    loop is what allows a new view to be added without touching the objective -- which is
    the property that makes the single-view baselines comparable to each other.
    """
    if encoder_name in GRAPH_ENCODERS:
        from torch_geometric.loader import DataLoader as GraphLoader

        from src.models.encoders.graph import build_graph_encoder

        graphs = {s: load_graphs(ds, s) for s in SPLITS}
        y = {s: labels_of(graphs[s]) for s in SPLITS}
        loaders = {
            s: GraphLoader(graphs[s], batch_size=batch_size, shuffle=(s == "train"),
                           drop_last=(s == "train" and _drop_last(len(graphs[s]), batch_size)))
            for s in SPLITS
        }
        encoder = build_graph_encoder(
            encoder_name, in_dim=graphs["train"][0].x.size(1), **enc_kwargs
        )
        return loaders, y, encoder, None

    if encoder_name in DESCRIPTOR_ENCODERS:
        parts = {s: load_descriptors(ds, s) for s in SPLITS}
        y = {s: parts[s][2] for s in SPLITS}
        loaders = {
            s: TensorLoader(
                TensorDataset(*parts[s], torch.arange(len(parts[s][2]))),
                batch_size=batch_size, shuffle=(s == "train"),
                drop_last=(s == "train" and _drop_last(len(parts[s][2]), batch_size)))
            for s in SPLITS
        }
        # Fitted on the training rows only. See the encoder's module docstring for why
        # fitting it over the whole pool would be a distributional leak.
        encoder = DescriptorEncoder.fit(parts["train"][0], parts["train"][1], **enc_kwargs)
        return loaders, y, encoder, None

    if encoder_name in SEQUENCE_ENCODERS:
        tok = {s: load_tokens(ds, s) for s in SPLITS}
        y = {s: tok[s]["y"] for s in SPLITS}
        loaders = {}
        samplers = {}
        for s in SPLITS:
            lengths = tok[s]["attention_mask"].sum(dim=1).numpy()
            samplers[s] = LengthBucketSampler(
                lengths, batch_size, shuffle=(s == "train"), seed=seed
            )
            loaders[s] = TensorLoader(
                _IndexDataset(len(lengths)),
                batch_sampler=samplers[s],
                collate_fn=make_token_collate(
                    tok[s]["input_ids"], tok[s]["attention_mask"], tok[s]["y"]
                ),
            )
        encoder = SequenceEncoder(frozen=(encoder_name == "seq_frozen"), **enc_kwargs)
        return loaders, y, encoder, samplers.get("train")

    raise ValueError(f"Unknown encoder: {encoder_name}")


def run_ds(ds, encoder_name, tag, epochs, patience, batch_size, lr, weight_decay,
           enc_kwargs, device=None, embed_dim=EMBED_DIM, head_hidden=EMBED_DIM):
    """Build one view's model and fit it through the shared loop."""
    seed = set_seed()
    cls = is_classification(ds)
    device = device or pick_device("cpu")

    loaders, y, encoder, train_sampler = build_view(
        encoder_name, ds, batch_size, enc_kwargs, seed=seed
    )
    model = SingleViewModel(encoder, n_tasks=y["train"].shape[1], embed_dim=embed_dim,
                            head_hidden=head_hidden)

    return fit_and_score(
        model, loaders, y, ds=ds, tag=tag, cls=cls, unpack=split_batch, device=device,
        epochs=epochs, patience=patience, lr=lr, weight_decay=weight_decay,
        train_sampler=train_sampler, seed=seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="gine",
                    choices=["gine", "gin", "attentivefp", "desc", "lora",
                             "seq_frozen"])
    ap.add_argument("--tag", default=None, help="output name; defaults to --encoder")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--jk", action="store_true")
    ap.add_argument("--readout", default="mean+sum", choices=["mean", "sum", "mean+sum"])
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--timesteps", type=int, default=2,
                    help="AttentiveFP: rounds of graph-level attention pooling")
    # Ablations for the descriptor view: which half of the input is doing the work.
    ap.add_argument("--no-ecfp", action="store_true",
                    help="desc only: drop the fingerprint, keep the 2-D descriptors")
    ap.add_argument("--no-descriptors", action="store_true",
                    help="desc only: drop the 2-D descriptors, keep the fingerprint "
                         "(this is the inherited ECFP input, under the shared trainer)")
    # Sequence view.
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--pooling", default="cls+mean", choices=["cls", "mean", "cls+mean"])
    # Held identical across views on purpose: see SingleViewModel. Changing either makes
    # the single-view rows non-comparable with each other and with the fusion model.
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM,
                    help="common width every view is projected to before the head")
    ap.add_argument("--head-hidden", type=int, default=EMBED_DIM)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    tag = args.tag or args.encoder
    enc_kwargs = dict(hidden=args.hidden)
    if args.encoder == "gine":
        enc_kwargs.update(n_layers=args.layers, residual=not args.no_residual,
                          jk=args.jk, readout=args.readout)
    elif args.encoder == "attentivefp":
        enc_kwargs.update(layers=args.layers, timesteps=args.timesteps,
                          dropout=args.dropout)
    elif args.encoder == "desc":
        enc_kwargs.update(dropout=args.dropout, use_ecfp=not args.no_ecfp,
                          use_desc=not args.no_descriptors)
    elif args.encoder in ("lora", "seq_frozen"):
        # The frozen control takes no adapter, so it must not be handed LoRA settings.
        enc_kwargs = dict(pooling=args.pooling)
        if args.encoder == "lora":
            enc_kwargs.update(lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_dropout)

    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    datasets = args.datasets or list(meta.keys())

    device = pick_device(args.device)
    rows = [run_ds(ds, args.encoder, tag, args.epochs, args.patience, args.batch_size,
                   args.lr, args.weight_decay, enc_kwargs, device,
                   args.embed_dim, args.head_hidden) for ds in datasets]
    pd.DataFrame(rows).to_csv(os.path.join(MET_DIR, f"{tag}_summary.csv"), index=False)
    print(f"\nWrote results/metrics/{tag}_summary.csv")


if __name__ == "__main__":
    main()

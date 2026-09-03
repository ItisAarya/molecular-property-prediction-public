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
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TensorLoader, Dataset as TorchDataset, TensorDataset

from src.data.bucketing import LengthBucketSampler, make_token_collate
from src.eval.metrics import cls_metrics, is_classification, reg_metrics
from src.models.encoders.descriptor import DescriptorEncoder
from src.models.encoders.sequence import SequenceEncoder

# torch_geometric is imported inside the graph branch of `build_view`, not here. It is a
# heavy dependency that only the graph views need, and requiring it at import time would
# stop the sequence and descriptor views running anywhere it is not installed -- which is
# exactly the case on a stock Colab runtime.
from src.models.heads import (
    SingleViewModel, masked_bce, masked_mse, pos_weight_from_labels,
)
from src.utils.seed import set_seed

DATA_DIR = "data"
PRED_DIR = "results/preds"
MET_DIR = "results/metrics"
MODELS_DIR = "models"
SPLITS = ("train", "valid", "test")

# Which representation each encoder reads. Adding a view means adding it here and a
# branch in `build_view`; the training loop below never changes.
GRAPH_ENCODERS = ("gine", "gin")
DESCRIPTOR_ENCODERS = ("desc",)
SEQUENCE_ENCODERS = ("lora", "seq_frozen")


def pick_device(name="auto"):
    """'auto' uses the GPU when torch can actually see one, otherwise the CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def to_device(obj, device):
    """Move a batch to the device, whichever representation it is."""
    if device.type == "cpu":
        return obj
    if isinstance(obj, tuple):
        return tuple(t.to(device) for t in obj)
    return obj.to(device)

for d in (PRED_DIR, MET_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


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
    return tuple(inputs), y, idx


def checkpoint_state(model):
    """
    The part of the model worth writing to disk: what training changed, plus fitted
    buffers (BatchNorm statistics, the descriptor scaler's constants).

    A LoRA run carries 44M frozen pretrained parameters that are byte-identical in every
    checkpoint. Saving the full state dict costs ~180 MB per fit and there are 96 fits in
    the full protocol -- roughly 17 GB of writes to store the same pretrained encoder
    ninety-six times. The encoder is reproducible from its name, so only the adapter, the
    head and the buffers need keeping.

    For the graph and descriptor views every parameter is trainable, so this saves exactly
    what it saved before.
    """
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    keep |= {n for n, _ in model.named_buffers()}
    return {k: v for k, v in model.state_dict().items() if k in keep}


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


@torch.no_grad()
def predict(model, loader, cls, device, n_rows):
    """
    Predictions for one split, always in dataset order.

    Length bucketing visits molecules out of order, and these arrays are saved to
    `results/preds/` where every downstream stage reads them positionally against the
    labels. Concatenating in arrival order would misalign predictions with their labels
    without raising anything -- the metrics would simply be wrong. So batches that report
    row indices are scattered back into place, and the result is checked for completeness.
    """
    model.eval()
    out, filled, cursor = None, None, 0

    for batch in loader:
        inputs, _, idx = split_batch(batch)
        p = model(to_device(inputs, device)).detach().cpu().numpy()
        if out is None:
            out = np.empty((n_rows, p.shape[1]), dtype=np.float64)
            filled = np.zeros(n_rows, dtype=bool)
        if idx is None:
            rows = np.arange(cursor, cursor + p.shape[0])
            cursor += p.shape[0]
        else:
            rows = idx.numpy()
        out[rows] = p
        filled[rows] = True

    if not filled.all():
        raise RuntimeError(
            f"predict covered {int(filled.sum())} of {n_rows} rows -- some molecules were "
            "never scored, so predictions would not line up with labels."
        )
    return 1.0 / (1.0 + np.exp(-out)) if cls else out


def run_ds(ds, encoder_name, tag, epochs, patience, batch_size, lr, weight_decay,
           enc_kwargs, device=None):
    seed = set_seed()
    cls = is_classification(ds)
    device = device or pick_device("cpu")

    loaders, y, encoder, train_sampler = build_view(
        encoder_name, ds, batch_size, enc_kwargs, seed=seed
    )
    n_tasks = y["train"].shape[1]
    n_train = int(y["train"].shape[0])

    model = SingleViewModel(encoder, n_tasks=n_tasks).to(device)
    # Optimise only what is actually trainable. With LoRA the pretrained encoder is
    # frozen, and handing frozen tensors to Adam would allocate optimiser state for 44M
    # parameters that never move.
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    pos_w = pos_weight_from_labels(y["train"]).to(device) if cls else None

    n_par = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    print(f"\n=== {ds} ({tag}) [seed={seed}, {device.type}] ===")
    print(f"  {n_train} train molecules, {n_tasks} task(s), "
          f"{n_par:,} params ({n_trainable:,} trainable), embedding dim {encoder.out_dim}")

    best_score, best_state, bad, best_ep = float("-inf"), None, 0, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        # Re-bucket with a different shuffle each epoch, reproducibly.
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        total = 0.0
        for batch in loaders["train"]:
            inputs, yb, _ = split_batch(batch)
            logits = model(to_device(inputs, device))
            yb = yb.to(device)
            loss = masked_bce(logits, yb, pos_w) if cls else masked_mse(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        p_val = predict(model, loaders["valid"], cls, device, int(y["valid"].shape[0]))
        m = cls_metrics(y["valid"].numpy(), p_val) if cls else reg_metrics(y["valid"].numpy(), p_val, ds)
        # One score, higher-is-better, so early stopping is identical for both task types.
        score = m["auc"] if cls else -m["rmse"]

        if score > best_score:
            best_score, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1:
            key = "val_auc" if cls else "val_rmse"
            val = m["auc"] if cls else m["rmse"]
            print(f"  epoch {ep:>3}  loss={total:8.3f}  {key}={val:.4f}")

        if bad >= patience:
            print(f"  early stop at epoch {ep} (best was {best_ep})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(checkpoint_state(model), os.path.join(MODELS_DIR, f"{ds}_{tag}.pt"))

    results = {}
    for split in ("valid", "test"):
        p = predict(model, loaders[split], cls, device, int(y[split].shape[0]))
        np.save(os.path.join(PRED_DIR, f"{ds}_{tag}_{split}.npy"), p)
        m = cls_metrics(y[split].numpy(), p) if cls else reg_metrics(y[split].numpy(), p, ds)
        pd.DataFrame([m]).to_csv(os.path.join(MET_DIR, f"{ds}_{tag}_{split}.csv"), index=False)
        results[split] = m

    key = "auc" if cls else "rmse"
    print(f"  valid {key}={results['valid'][key]:.4f}   test {key}={results['test'][key]:.4f}"
          f"   ({(time.time() - t0) / 60:.1f} min, best epoch {best_ep})")
    return {"dataset": ds, "tag": tag, f"valid_{key}": results["valid"][key],
            f"test_{key}": results["test"][key], "params": n_par, "best_epoch": best_ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="gine",
                    choices=["gine", "gin", "desc", "lora", "seq_frozen"])
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
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    tag = args.tag or args.encoder
    enc_kwargs = dict(hidden=args.hidden)
    if args.encoder == "gine":
        enc_kwargs.update(n_layers=args.layers, residual=not args.no_residual,
                          jk=args.jk, readout=args.readout)
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
                   args.lr, args.weight_decay, enc_kwargs, device) for ds in datasets]
    pd.DataFrame(rows).to_csv(os.path.join(MET_DIR, f"{tag}_summary.csv"), index=False)
    print(f"\nWrote results/metrics/{tag}_summary.csv")


if __name__ == "__main__":
    main()

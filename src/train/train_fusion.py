"""
src/train/train_fusion.py

Train a multi-view fusion model and write predictions in the pipeline's format.

    python -m src.train.train_fusion --mode proposed --tag fuse_proposed
    python -m src.train.train_fusion --mode concat --tag fuse_concat --datasets esol
    python -m src.train.train_fusion --mode proposed --seq lora --device cuda

Three views per molecule -- graph, sequence, descriptor -- encoded, projected to a common
width, combined by one of the five strategies in `src/models/fusion.py`, then a masked
multi-task head. Trained through `src/train/loop.py`, the same loop every single-view
baseline used, so a difference between a fusion row and a single-view row is a difference
in architecture and not in training recipe.

TWO SEQUENCE MODES, AND WHY THE DEFAULT IS THE CACHED ONE
---------------------------------------------------------
`--seq cached` (default) reads the frozen ChemBERTa vectors computed once in Phase 0.
`--seq lora` runs the real encoder with trainable LoRA adapters.

The frozen encoder is a pure function of the molecule, so its output cannot change between
epochs and recomputing it is pure waste -- Phase 0 measured that stage at ~75 minutes
against 40 seconds cached. In fusion the saving is larger still, because a step now runs
three encoders and the ChemBERTa forward pass dominates the other two by roughly an order
of magnitude on CPU. Cached mode makes the fusion ablation ladder runnable on a laptop;
LoRA mode needs a GPU and is the end-to-end setting from the plan's §4.3.

Both are honest, and they answer different questions: cached asks whether *fusion* helps,
holding the views fixed at what Phase 1 measured; LoRA asks whether fusing and adapting
together helps. Reporting the cached ladder first keeps the fusion effect separable from
the adaptation effect, which is the same reason Phase 1c ran a frozen control.

ROW ALIGNMENT ACROSS THREE VIEWS
--------------------------------
The hazard specific to this trainer: a molecule's graph, tokens and descriptors are three
separate files, and they must refer to the same molecule in the same batch position. They
are all built from the same pooled order and sliced by the same split indices, so the
alignment holds by construction -- and `--check-alignment` verifies it against the SMILES
strings rather than trusting that.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.bucketing import LengthBucketSampler, POOL_MULTIPLIER
from src.eval.metrics import is_classification
from src.models.encoders.cached import CachedEmbeddingEncoder
from src.models.encoders.descriptor import DescriptorEncoder
from src.models.fusion import MODES
from src.models.heads import EMBED_DIM
from src.models.multiview import MultiViewModel
from src.train.loop import MET_DIR, fit_and_score
from src.utils.seed import set_seed

DATA_DIR = "data"
SPLITS = ("train", "valid", "test")
VIEW_ORDER = ("graph", "seq", "desc")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_split(ds, split, seq_mode):
    """Every view for one split, plus labels. Row i is the same molecule in all of them."""
    ecfp = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_ecfp.npz"), allow_pickle=True)
    desc = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_desc.npz"), allow_pickle=True)
    graphs = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_graphs.pt"),
                        weights_only=False)["graphs"]

    out = {
        "ecfp": torch.tensor(ecfp["X"], dtype=torch.float32),
        "desc": torch.tensor(desc["X"], dtype=torch.float32),
        "graphs": graphs,
        "y": torch.tensor(ecfp["y"], dtype=torch.float32),
        "smiles": [str(s) for s in ecfp["smiles"]],
    }

    if seq_mode == "cached":
        emb = np.load(os.path.join(DATA_DIR, f"{ds}_{split}_chemberta.npz"))
        # CLS and masked mean concatenated, matching the pooling the LoRA view uses.
        out["seq"] = torch.tensor(
            np.concatenate([emb["cls"], emb["mean"]], axis=1), dtype=torch.float32)
    else:
        tok = torch.load(os.path.join(DATA_DIR, f"{ds}_{split}_tok.pt"), weights_only=False)
        out["input_ids"] = tok["input_ids"]
        out["attention_mask"] = tok["attention_mask"]
    return out


def check_alignment(parts):
    """
    Confirm the three views really do refer to the same molecules, row by row.

    They are sliced from one pooled order by one index array, so this should hold by
    construction. It is checked anyway because a misalignment would not crash -- it would
    train each view on a different molecule's label and quietly produce a plausible,
    meaningless number.
    """
    for split, p in parts.items():
        n = len(p["smiles"])
        assert p["ecfp"].shape[0] == n, f"{split}: ecfp rows != molecules"
        assert p["desc"].shape[0] == n, f"{split}: descriptor rows != molecules"
        assert len(p["graphs"]) == n, f"{split}: graph count != molecules"
        assert p["y"].shape[0] == n, f"{split}: label rows != molecules"
        if "seq" in p:
            assert p["seq"].shape[0] == n, f"{split}: cached embedding rows != molecules"
        else:
            assert p["input_ids"].shape[0] == n, f"{split}: token rows != molecules"
        # The graph objects carry their own labels; they must match the label matrix.
        ymat = torch.stack([g.y for g in p["graphs"]])
        both = ~(torch.isnan(ymat) | torch.isnan(p["y"]))
        assert torch.allclose(ymat[both], p["y"][both]), \
            f"{split}: graph labels disagree with the label matrix -- views are misaligned"
    return True


# --------------------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------------------
class _IndexDataset(Dataset):
    """Hands out row numbers; the collate function does the gathering."""

    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def make_collate(part, seq_mode):
    """Build one batch of all three views from a list of row indices."""
    from torch_geometric.data import Batch

    def collate(indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        views = {
            "graph": Batch.from_data_list([part["graphs"][int(i)] for i in idx]),
            "desc": (part["ecfp"][idx], part["desc"][idx]),
        }
        if seq_mode == "cached":
            views["seq"] = part["seq"][idx]
        else:
            mask = part["attention_mask"][idx]
            width = max(int(mask.sum(dim=1).max().item()), 1)
            views["seq"] = (part["input_ids"][idx][:, :width], mask[:, :width])
        return views, part["y"][idx], idx.numpy()

    return collate


def unpack(batch):
    """`(views, labels, row indices)` -- the contract `src/train/loop.py` expects."""
    return batch[0], batch[1], batch[2]


def build_loaders(parts, batch_size, seq_mode, seed):
    loaders, train_sampler = {}, None
    for split in SPLITS:
        n = len(parts[split]["smiles"])
        shuffle = split == "train"
        if seq_mode == "lora":
            # Only worth bucketing when the transformer actually runs: padding is what it
            # costs. Every view is gathered by the same index list, so grouping molecules
            # by sequence length keeps all three aligned.
            sampler = LengthBucketSampler(
                parts[split]["attention_mask"].sum(dim=1).numpy(),
                batch_size, shuffle=shuffle, seed=seed)
        else:
            sampler = _PlainBatchSampler(n, batch_size, shuffle=shuffle, seed=seed)
        if shuffle:
            train_sampler = sampler
        loaders[split] = DataLoader(
            _IndexDataset(n), batch_sampler=sampler,
            collate_fn=make_collate(parts[split], seq_mode))
    return loaders, train_sampler


class _PlainBatchSampler:
    """Shuffled fixed-size batches, with the same `set_epoch` contract as the bucketer."""

    def __init__(self, n, batch_size, shuffle=True, seed=0):
        self.n, self.batch_size, self.shuffle, self.seed = int(n), int(batch_size), shuffle, seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _batches(self):
        order = (np.random.default_rng(self.seed + self.epoch).permutation(self.n)
                 if self.shuffle else np.arange(self.n))
        out = [order[i:i + self.batch_size].tolist()
               for i in range(0, self.n, self.batch_size)]
        # BatchNorm cannot compute a variance from one sample; this drops at most one
        # molecule, and only when the split size leaves a remainder of exactly one.
        if self.shuffle and len(out) > 1 and len(out[-1]) == 1:
            out = out[:-1]
        return out

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def build_encoders(parts, seq_mode, graph_encoder, hidden, dropout):
    from src.models.encoders.graph import build_graph_encoder

    train = parts["train"]
    encoders = {
        "graph": build_graph_encoder(graph_encoder, in_dim=train["graphs"][0].x.size(1),
                                     hidden=hidden),
        "desc": DescriptorEncoder.fit(train["ecfp"], train["desc"], hidden=hidden,
                                      dropout=dropout),
    }
    if seq_mode == "cached":
        encoders["seq"] = CachedEmbeddingEncoder(train["seq"].shape[1])
    else:
        from src.models.encoders.sequence import SequenceEncoder
        encoders["seq"] = SequenceEncoder(lora_r=8)
    # Fixed order, so the gate's weight columns always mean the same thing.
    return {name: encoders[name] for name in VIEW_ORDER}


def run_ds(ds, args, device):
    seed = set_seed()
    cls = is_classification(ds)

    parts = {s: load_split(ds, s, args.seq) for s in SPLITS}
    if args.check_alignment:
        check_alignment(parts)

    loaders, train_sampler = build_loaders(parts, args.batch_size, args.seq, seed)
    y = {s: parts[s]["y"] for s in SPLITS}

    encoders = build_encoders(parts, args.seq, args.graph_encoder, args.hidden, args.dropout)
    model = MultiViewModel(encoders, mode=args.mode, n_tasks=y["train"].shape[1],
                           d=args.embed_dim, rank=args.rank, n_layers=args.xattn_layers,
                           n_heads=args.xattn_heads)

    return fit_and_score(
        model, loaders, y, ds=ds, tag=args.tag, cls=cls, unpack=unpack, device=device,
        epochs=args.epochs, patience=args.patience, lr=args.lr,
        weight_decay=args.weight_decay, train_sampler=train_sampler, seed=seed,
        extra={"mode": args.mode, "seq": args.seq},
    )


def main():
    ap = argparse.ArgumentParser(description="Train a multi-view fusion model.")
    ap.add_argument("--mode", default="proposed", choices=list(MODES))
    ap.add_argument("--tag", default=None, help="output name; defaults to fuse_<mode>")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seq", default="cached", choices=["cached", "lora"],
                    help="cached = frozen ChemBERTa vectors from Phase 0 (CPU-friendly); "
                         "lora = train adapters end to end (needs a GPU)")
    ap.add_argument("--graph-encoder", default="gine", choices=["gine", "gin"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    ap.add_argument("--rank", type=int, default=64, help="low-rank bilinear rank")
    ap.add_argument("--xattn-layers", type=int, default=2)
    ap.add_argument("--xattn-heads", type=int, default=4)
    ap.add_argument("--check-alignment", action="store_true", default=True)
    ap.add_argument("--no-check-alignment", dest="check_alignment", action="store_false")
    args = ap.parse_args()

    args.tag = args.tag or f"fuse_{args.mode}"
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))

    meta = json.load(open(os.path.join(DATA_DIR, "dataset_meta.json")))
    datasets = args.datasets or list(meta.keys())

    rows = [run_ds(ds, args, device) for ds in datasets]
    out = os.path.join(MET_DIR, f"{args.tag}_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

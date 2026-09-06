"""
src/data/bucketing.py

Length-bucketed batching for the tokenised SMILES view.

WHY
---
`tokenize_smiles.py` pads every SMILES in a dataset to one fixed width, because the token
files are rectangular tensors. The width is set by the longest molecule, but the *average*
molecule is far shorter -- 25.9 tokens against a padded 128 on Tox21, 12.0 against 67 on
FreeSolv. A transformer spends compute per token whether or not the token carries
information, so the fixed padding means roughly 3.8x of the work across these eight
datasets is spent on padding.

Bucketing removes that waste and nothing else. Molecules of similar length are grouped into
the same batch, and each batch is trimmed to the longest *real* sequence it actually
contains. No model, loss or split changes; the same molecules train on the same labels.

HOW THE SHUFFLING STAYS HONEST
------------------------------
Sorting the whole training set by length and slicing it into batches would group molecules
by size for the entire run -- every batch would be homogeneous, batch composition would be
identical each epoch, and small molecules would never share a gradient step with large
ones. That is a change to training, not just to speed.

So we shuffle first, cut the shuffled order into large pools (50 batches' worth), and sort
only *within* a pool. Batches are then drawn from the pool and their order shuffled again.
Padding within a batch stays small because a pool spans a narrow slice of the shuffled
data, while batch membership still varies from epoch to epoch.

THE ORDERING HAZARD
-------------------
Bucketing permutes the order in which molecules are visited. Predictions come back in
batch order, but they are saved as arrays that must line up row-for-row with the labels --
`results/preds/<ds>_<tag>_test.npy` is read positionally by every downstream stage.

Silently misaligning predictions and labels would not crash; it would produce plausible,
wrong metrics. So every batch carries the row indices it was built from, and the prediction
routine scatters results back to their original positions rather than concatenating them in
arrival order.
"""

import numpy as np
import torch
from torch.utils.data import Sampler

# A pool this many batches wide. Large enough that batch composition varies between
# epochs, narrow enough that sorting inside it leaves little padding.
POOL_MULTIPLIER = 50


class LengthBucketSampler(Sampler):
    """
    Yields lists of row indices, grouped so each batch holds similar-length sequences.

    Set `shuffle=False` for validation and test: the order is then deterministic. It is
    still permuted relative to the dataset (that is the point), which is why batches carry
    their indices.

    A trailing batch of one molecule is dropped while shuffling. FreeSolv has 513 training
    molecules against a batch size of 128, leaving exactly one in the last batch, and any
    model containing BatchNorm -- the descriptor view does -- raises on it. The cached
    fusion path never hit this because its sampler already dropped such a batch; the
    end-to-end path did, on the first dataset it touched.
    """

    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False, seed=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        """Vary the shuffle per epoch while keeping the whole run reproducible."""
        self.epoch = int(epoch)

    def _batches(self):
        n = len(self.lengths)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            order = rng.permutation(n)
        else:
            order = np.arange(n)

        pool = self.batch_size * POOL_MULTIPLIER
        batches = []
        for start in range(0, n, pool):
            chunk = order[start:start + pool]
            # Stable sort so equal lengths keep their shuffled relative order.
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for b in range(0, len(chunk), self.batch_size):
                batch = chunk[b:b + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                # A batch of one cannot be trained on when the model contains BatchNorm,
                # which cannot compute a variance from a single sample. Only shuffling
                # loaders are training loaders, and this drops at most one molecule per
                # pool. Evaluation loaders keep every row -- dropping one there would
                # leave a molecule unscored, which `predict` treats as a hard error.
                if self.shuffle and len(batch) == 1:
                    continue
                batches.append(batch.tolist())

        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch + 10_000).shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def make_token_collate(input_ids, attention_mask, y):
    """
    Build a collate function that gathers rows and trims each batch to its real width.

    Returns batches of (input_ids, attention_mask, y, row_indices). The indices are what
    let `predict` put results back in dataset order -- see the module docstring.
    """

    def collate(indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        mask = attention_mask[idx]
        # The tokenizer pads on the right, so the real content of every row in this batch
        # sits in the first `width` columns.
        width = int(mask.sum(dim=1).max().item())
        width = max(width, 1)
        return input_ids[idx][:, :width], mask[:, :width], y[idx], idx

    return collate

"""
src/models/heads.py

Prediction heads and the masked multi-task losses they are trained with.

Every view encoder in this project returns a single vector per molecule, so the head is
the only part that knows how many tasks a dataset has or whether it is classification or
regression. Keeping that in one place means an encoder can be swapped without touching
anything about the objective, which is what the Phase 2 fusion ablations need.

WHY THE LOSSES ARE MASKED
-------------------------
Tox21 is sparsely measured: ~15% of training labels and ~24% of validation and test labels
are for (molecule, assay) pairs that were never tested, and those are stored as NaN.

A plain `BCEWithLogitsLoss` on NaN returns NaN, and one NaN gradient step turns every
weight in the model to NaN -- training fails silently and the model predicts NaN forever
after. So the loss is computed elementwise, multiplied by a "this label exists" mask, and
averaged over the measured entries only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The common embedding width every view is projected to. Fixed across views so that a
# comparison between them is a comparison of representations and not of head sizes.
EMBED_DIM = 256


class MLPHead(nn.Module):
    """
    Two-layer MLP from a molecule embedding to per-task outputs.

    Returns raw logits for classification (the loss applies the sigmoid internally, which
    is numerically stabler than sigmoid-then-BCE) and raw values for regression.
    """

    def __init__(self, d_in, n_tasks, hidden=None, dropout=0.2):
        super().__init__()
        hidden = hidden or d_in
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, h):
        return self.net(h)


def masked_bce(logits, y, pos_weight=None):
    """Binary cross-entropy over measured labels only. NaN entries contribute nothing."""
    mask = ~torch.isnan(y)
    # nan_to_num keeps the elementwise op well defined; the mask removes the contribution.
    loss = F.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(y, nan=0.0), pos_weight=pos_weight, reduction="none"
    )
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def masked_mse(preds, y):
    """Mean squared error over measured targets only."""
    mask = ~torch.isnan(y)
    loss = F.mse_loss(preds, torch.nan_to_num(y, nan=0.0), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def pos_weight_from_labels(y):
    """
    Per-task negative/positive ratio, for re-weighting a rare positive class.

    Computed over measured labels only -- counting the NaN placeholders as negatives is
    exactly the bug Phase 0 removed, and it would inflate every weight here.
    """
    weights = []
    for t in range(y.shape[1]):
        col = y[:, t]
        col = col[~torch.isnan(col)]
        pos = (col == 1).sum().item()
        neg = (col == 0).sum().item()
        weights.append(neg / pos if pos > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


class SingleViewModel(nn.Module):
    """
    One encoder plus one head: the single-view baseline for a given representation.

    These are the rows the fusion model has to beat in the results table, so they are
    trained through exactly the same loop and objective the fusion model will use. A
    single-view number produced by a different training recipe would not be a fair
    comparison.
    """

    def __init__(self, encoder, n_tasks, embed_dim=EMBED_DIM, head_hidden=EMBED_DIM,
                 dropout=0.2):
        super().__init__()
        self.encoder = encoder

        # Every view is projected to one common width before the head.
        #
        # Without this the head is sized from whatever the encoder happens to emit -- 256
        # for GIN, 512 for GINE with mean+sum readout, 1536 for the sequence views with
        # CLS+mean pooling. The single-view baselines then differed in *head capacity* as
        # well as in representation, so the comparison measured both at once. On the
        # sequence view the head was 2.36M parameters against a 147k LoRA adapter: most of
        # what was being trained was the head, not the view.
        #
        # It is also a hard requirement for Phase 2. Cross-attention over views, and the
        # low-rank bilinear term between pairs of them, both need every view to arrive at
        # the same width.
        self.project = (nn.Identity() if encoder.out_dim == embed_dim
                        else nn.Linear(encoder.out_dim, embed_dim))
        self.out_dim = embed_dim
        self.head = MLPHead(embed_dim, n_tasks, hidden=head_hidden, dropout=dropout)

    def forward(self, batch):
        return self.head(self.encode(batch))

    def encode(self, batch):
        """The molecule embedding at the common width, for the fusion module to consume."""
        return self.project(self.encoder(batch))

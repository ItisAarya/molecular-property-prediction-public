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

    def __init__(self, encoder, n_tasks, head_hidden=None, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.head = MLPHead(encoder.out_dim, n_tasks, hidden=head_hidden, dropout=dropout)

    def forward(self, batch):
        return self.head(self.encoder(batch))

    def encode(self, batch):
        """The molecule embedding, for the fusion module to consume in Phase 2."""
        return self.encoder(batch)

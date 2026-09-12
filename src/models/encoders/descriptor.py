"""
src/models/encoders/descriptor.py

The descriptor view: 1024-bit ECFP fingerprint concatenated with normalised RDKit 2-D
physicochemical descriptors, encoded by a small MLP.

    from src.models.encoders.descriptor import DescriptorEncoder
    enc = DescriptorEncoder.fit(ecfp_train, desc_train, hidden=256)

WHAT THIS VIEW ADDS
-------------------
The inherited pipeline used ECFP alone, and only inside a Random Forest. A fingerprint is
a record of which substructures occur in a molecule: it is excellent at "does this contain
a sulfonamide" and carries nothing at all about bulk properties -- molecular weight, logP,
polar surface area, rotatable-bond count. Those govern solubility, membrane permeability
and hydration free energy, which is most of what these datasets measure. So the descriptor
block is not a restatement of the fingerprint; it is the part the fingerprint omits.

WHY THE SCALER LIVES INSIDE THE ENCODER
---------------------------------------
Descriptors span wildly different ranges -- a count of 0-6 rings sits next to a molecular
weight of 7,000 -- so they must be standardised before an MLP can use them. That makes the
scaler a *fitted* object, and every fitted object in this project has to be fitted on
training rows only. Phase 0 removed four separate stages that were fitted on data they
were later scored against; a scaler fitted over the whole pool would be a fifth, quieter
one: the test molecules would be setting the mean and standard deviation the model trains
against.

So the constants are fitted once from the training split via `DescriptorEncoder.fit`, and
then stored as buffers. Buffers travel inside `state_dict`, so a checkpoint carries the
exact normalisation it was trained with and cannot be reloaded against a different one.

THREE FITTED CONSTANTS, AND WHY EACH IS NEEDED
----------------------------------------------
`median`  Imputes descriptors that are undefined for a molecule (the BCUT2D and
          partial-charge families fail on atoms outside their parameter sets -- up to
          ~6% of molecules). The median, not zero: zero is a legitimate value for many of
          these columns and would be indistinguishable from a real measurement.
`keep`    Drops columns that are constant across the training set. They carry no signal,
          and their zero standard deviation would divide by (near) zero.
`mean`/`std`  The z-scoring itself.

Standardised values are then clipped to +/-`CLIP`. A single 7,000-dalton peptide sits tens
of standard deviations out on a dozen columns, and without a clip that one molecule would
dominate the first layer's gradients. Clipping bounds its influence without discarding it.
"""

import torch
import torch.nn as nn

# Standardised descriptors are clipped to this many standard deviations.
CLIP = 10.0
# A training-set standard deviation below this counts as constant.
CONSTANT_TOL = 1e-8


class DescriptorEncoder(nn.Module):
    """
    ECFP + normalised 2-D descriptors -> a fixed-width molecule embedding.

    Build with `DescriptorEncoder.fit(...)` rather than calling the constructor directly:
    the normalisation constants have to come from the training split, and `fit` is the
    only place that computes them.
    """

    def __init__(self, n_ecfp, median, keep, mean, std, hidden=256, dropout=0.3,
                 use_ecfp=True, use_desc=True):
        super().__init__()
        if not (use_ecfp or use_desc):
            raise ValueError("The descriptor view needs at least one of ECFP or descriptors.")

        self.use_ecfp = use_ecfp
        self.use_desc = use_desc
        self.n_ecfp = n_ecfp

        # Buffers, not parameters: fitted constants, not learned weights. Saved and
        # restored with the model so a checkpoint cannot drift from its own scaler.
        self.register_buffer("median", median)
        self.register_buffer("keep", keep)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        n_desc = int(keep.sum().item())
        d_in = (n_ecfp if use_ecfp else 0) + (n_desc if use_desc else 0)

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.out_dim = hidden

    @staticmethod
    def _stats(desc_train):
        """Fit impute/keep/scale constants from the training descriptor matrix."""
        x = torch.as_tensor(desc_train, dtype=torch.float32)
        missing = torch.isnan(x)

        # nanmedian over each column, ignoring the undefined entries.
        median = torch.zeros(x.shape[1], dtype=torch.float32)
        for j in range(x.shape[1]):
            col = x[:, j][~missing[:, j]]
            median[j] = col.median() if col.numel() else torch.tensor(0.0)

        filled = torch.where(missing, median.expand_as(x), x)
        mean = filled.mean(dim=0)
        std = filled.std(dim=0)
        keep = std > CONSTANT_TOL
        # Guard the division for dropped columns; they are masked out anyway.
        std = torch.where(keep, std, torch.ones_like(std))
        return median, keep, mean, std

    @classmethod
    def fit(cls, ecfp_train, desc_train, **kwargs):
        """Fit the normalisation on training rows only, then build the encoder."""
        median, keep, mean, std = cls._stats(desc_train)
        n_ecfp = int(torch.as_tensor(ecfp_train).shape[1])
        return cls(n_ecfp, median, keep, mean, std, **kwargs)

    def normalise(self, desc):
        """Impute, standardise, drop constant columns, clip. No fitting happens here."""
        x = torch.as_tensor(desc, dtype=torch.float32)
        x = torch.where(torch.isnan(x), self.median.expand_as(x), x)
        x = (x - self.mean) / self.std
        x = x.clamp(-CLIP, CLIP)
        return x[:, self.keep]

    def forward(self, batch):
        """`batch` is (ecfp, descriptors), both float tensors of shape (B, ...)."""
        ecfp, desc = batch
        parts = []
        if self.use_ecfp:
            parts.append(torch.as_tensor(ecfp, dtype=torch.float32))
        if self.use_desc:
            parts.append(self.normalise(desc))
        return self.net(torch.cat(parts, dim=1))

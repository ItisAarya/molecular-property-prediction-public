"""
src/models/encoders/cached.py

A view whose encoder has already been run: read the vector, do not recompute it.

WHY
---
The frozen sequence view is a pure function of the molecule -- the encoder's weights never
change, so the same SMILES produces the same 768-d CLS and mean vectors on every epoch of
every run. Phase 0 already exploited this once: caching those vectors cut the transformer
stage from ~75 minutes to 40 seconds.

Fusion training needs that saving far more than single-view training did. A fusion step
runs three encoders instead of one, and on CPU the ChemBERTa forward pass dominates
everything else by an order of magnitude -- roughly 19 minutes per epoch across the eight
datasets, against seconds for the graph and descriptor encoders. Reading the cached vector
instead makes frozen-mode fusion trainable on a laptop.

This is the `frozen encoders` mode from the plan's §4.3. The end-to-end mode, where LoRA
adapters train and the vectors genuinely change, runs the real encoder and needs a GPU.

WHAT IS AND IS NOT FROZEN
-------------------------
The encoder is frozen; the projection that follows it is not. So the fusion model still
learns how to use the sequence view, it just does not learn to re-encode SMILES. That is
the same arrangement the inherited pipeline used for its transformer, and the honest label
for it is "frozen encoder", not "no sequence view".
"""

import torch
import torch.nn as nn


class CachedEmbeddingEncoder(nn.Module):
    """
    Passes through a precomputed embedding.

    Holds no parameters at all: it exists so a cached view has the same
    `.out_dim` / `forward(batch)` interface as a real encoder, and can therefore sit in
    the same fusion module without special-casing anywhere else.
    """

    def __init__(self, dim):
        super().__init__()
        self.out_dim = int(dim)

    def forward(self, batch):
        # `batch` is the already-encoded (B, out_dim) tensor.
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return torch.as_tensor(x, dtype=torch.float32)

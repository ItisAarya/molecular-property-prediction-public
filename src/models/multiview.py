"""
src/models/multiview.py

The multi-view model: three encoders, one fusion module, one head.

    model = MultiViewModel(encoders, mode="proposed", n_tasks=12)
    logits = model(batch)          # batch is a dict keyed by view name

Kept separate from the trainer so the fusion ablations, the interpretability analysis and
any later inference path all construct the same object rather than three similar ones.

WHY THE PROJECTIONS LIVE HERE
-----------------------------
Each encoder emits its own natural width -- 256 for GIN, 512 for GINE with mean+sum
readout, 1536 for the sequence view with CLS+mean pooling, 256 for the descriptor MLP.
Cross-attention over views and the pairwise bilinear term both require a common width, so
every view is projected to `d` before fusion.

This mirrors `SingleViewModel` deliberately. The single-view baselines and the fusion model
must differ in *how views are combined* and in nothing else; if the fusion model gave its
views a wider projection than the baselines got, a win would partly be extra capacity.
"""

import torch.nn as nn

from src.models.fusion import build_fusion
from src.models.heads import EMBED_DIM, MLPHead


class MultiViewModel(nn.Module):
    """
    `encoders` is an ordered dict of {view name: encoder}. Every encoder must expose
    `.out_dim` and accept whatever `batch[view_name]` holds.

    The view order is fixed at construction and reused everywhere -- the fusion module's
    pair indices and the gate's weight columns both refer to it, so the interpretability
    output would be mislabelled if it varied.
    """

    def __init__(self, encoders, mode="proposed", n_tasks=1, d=EMBED_DIM, dropout=0.2,
                 **fusion_kwargs):
        super().__init__()
        self.view_names = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.project = nn.ModuleDict({
            name: (nn.Identity() if enc.out_dim == d else nn.Linear(enc.out_dim, d))
            for name, enc in encoders.items()
        })
        self.fusion = build_fusion(mode, n_views=len(self.view_names), d=d, **fusion_kwargs)
        self.head = MLPHead(self.fusion.out_dim, n_tasks, hidden=d, dropout=dropout)
        self.d = d

    def encode_views(self, batch):
        """Per-view embeddings at the common width, in the fixed view order."""
        return [self.project[n](self.encoders[n](batch[n])) for n in self.view_names]

    def forward(self, batch):
        return self.head(self.fusion(self.encode_views(batch)))

    def view_weights(self):
        """
        Per-molecule gate weights from the last forward pass, or None for variants
        without a gate. Columns follow `self.view_names`.
        """
        return self.fusion.view_weights()

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

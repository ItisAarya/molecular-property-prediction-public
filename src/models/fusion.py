"""
src/models/fusion.py

Five ways to combine the view embeddings, behind one interface.

    from src.models.fusion import build_fusion
    fuse = build_fusion("proposed", n_views=3, d=256)
    h = fuse([h_graph, h_seq, h_desc])       # each (B, 256) -> (B, fuse.out_dim)

WHAT THIS REPLACES
------------------
The inherited pipeline's "hybrid" was a per-task logistic regression over three scalar
probabilities. Three numbers per molecule, combined linearly. Everything the encoders had
learned was thrown away before the combination step, so the model could not represent
"trust the graph view when the molecule has an unusual scaffold" or any interaction
between what two views saw. This module combines the 256-d representations instead.

WHY FIVE VARIANTS AND NOT ONE
-----------------------------
`proposed` is the contribution, but a single number for it is not a result. Each variant
below removes one mechanism, so the ladder answers a specific question:

    concat     Does combining views help at all, beyond the best single view?
    gated      Does *weighting* views per molecule beat concatenating them?
    xattn      Does letting views attend to each other help?          (~MvMRL)
    bilinear   Do explicit second-order interactions help?            (~KROVEX)
    proposed   Do cross-attention and the bilinear term compose?

If `proposed` wins but `concat` wins by just as much, the contribution is "use more than
one view", which is not novel. The ladder is what distinguishes those outcomes.

Phase 1 is why this is worth building at all. No single encoder beat the inherited GIN
across eight datasets, but the *ordering reverses* between datasets -- the descriptor view
is best on FreeSolv and worst on ClinTox, the sequence views the other way round, with a
best-to-worst spread of 0.146 AUC on ClinTox. There is complementary signal. Whether a
learned combination can extract it is what these five variants measure.

A NOTE ON WHAT "FUSION" COSTS
-----------------------------
Every variant here adds parameters, and Phase 1's lesson was that more parameters have
repeatedly bought nothing on these datasets (GINE: 5x the parameters, 0 wins in 8). The
minimum detectable effect is ~0.02 AUC / ~0.10 RMSE. A fusion model that improves by less
than that has not improved.
"""

import torch
import torch.nn as nn

MODES = ("concat", "gated", "xattn", "bilinear", "proposed")

# Rank of the low-rank bilinear projection. Full bilinear interaction between two 256-d
# views would be a 256x256 matrix per pair; the low-rank form uses two d x r projections
# and an elementwise product, which is 2dr parameters instead of d^2.
BILINEAR_RANK = 64


class GateBlock(nn.Module):
    """
    Per-molecule softmax weights over views.

    Scores each view from its own embedding, then normalises across views, so a molecule
    can lean on the graph view while another leans on the sequence view. The weights are
    kept on `.last_weights` after a forward pass because they are the interpretability
    output the paper reports: which view does each dataset actually rely on.
    """

    def __init__(self, d, n_views, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.n_views = n_views
        self.last_weights = None

    def forward(self, views):
        # (B, n_views, 1) -> (B, n_views)
        scores = torch.cat([self.score(v) for v in views], dim=1)
        alpha = torch.softmax(scores, dim=1)
        self.last_weights = alpha.detach()
        stacked = torch.stack(views, dim=1)              # (B, n_views, d)
        return (alpha.unsqueeze(-1) * stacked).sum(dim=1)


class CrossAttentionBlock(nn.Module):
    """
    Self-attention over the views, which is cross-attention between them.

    Each view is one token in a sequence of length `n_views`, so attention lets the graph
    embedding be rewritten in terms of what the sequence and descriptor embeddings hold.
    This is the MvMRL mechanism, applied over pretrained view encoders rather than a
    from-scratch SMILES CNN.

    `batch_first=True` because the rest of this codebase carries (B, ...) tensors, and the
    silent transposition bug that the default would invite is not worth the saving.
    """

    def __init__(self, d, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # enable_nested_tensor is a fast path for padded batches. Views are a fixed-length
        # sequence with no padding, so it never applies and only emits a warning.
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)

    def forward(self, views):
        tokens = torch.stack(views, dim=1)               # (B, n_views, d)
        return list(self.blocks(tokens).unbind(dim=1))   # back to a list of (B, d)


class BilinearBlock(nn.Module):
    """
    Low-rank bilinear interaction for every unordered pair of views.

    For views i and j: z_ij = (W_i h_i) * (W_j h_j), elementwise, with W: d -> r. The
    elementwise product of two projections is the rank-r factorisation of a bilinear form,
    so z_ij carries *second-order* terms -- products of a feature from one view with a
    feature from another -- which no concatenation or weighted sum can represent. This is
    the KROVEX mechanism in its cheap form.

    Each pair gets its own projections: what matters in a graph-sequence interaction is
    not the same as in a graph-descriptor one.
    """

    def __init__(self, d, n_views, rank=BILINEAR_RANK):
        super().__init__()
        self.pairs = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]
        self.left = nn.ModuleList([nn.Linear(d, rank) for _ in self.pairs])
        self.right = nn.ModuleList([nn.Linear(d, rank) for _ in self.pairs])
        self.norm = nn.LayerNorm(rank * len(self.pairs))
        self.out_dim = rank * len(self.pairs)

    def forward(self, views):
        out = []
        for k, (i, j) in enumerate(self.pairs):
            out.append(self.left[k](views[i]) * self.right[k](views[j]))
        return self.norm(torch.cat(out, dim=1))


class FusionModule(nn.Module):
    """
    One fusion strategy. `forward` takes a list of `n_views` tensors of shape (B, d) and
    returns (B, out_dim); the caller puts a head on top.

    `view_weights()` returns the per-molecule gate weights for the variants that have a
    gate, and None otherwise.
    """

    def __init__(self, mode, n_views, d, rank=BILINEAR_RANK, n_layers=2, n_heads=4,
                 dropout=0.1):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown fusion mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.n_views = n_views
        self.d = d

        self.gate = None
        self.xattn = None
        self.bilinear = None

        if mode == "concat":
            self.out_dim = n_views * d
        elif mode == "gated":
            self.gate = GateBlock(d, n_views)
            self.out_dim = d
        elif mode == "xattn":
            self.xattn = CrossAttentionBlock(d, n_layers, n_heads, dropout)
            self.out_dim = n_views * d
        elif mode == "bilinear":
            self.bilinear = BilinearBlock(d, n_views, rank)
            self.out_dim = self.bilinear.out_dim
        else:  # proposed
            # Cross-attention aligns the views, the bilinear block adds the explicit
            # second-order term, and the gate produces the interpretable per-molecule
            # attribution. The gated sum is kept alongside the interaction features
            # because the interaction terms alone discard the first-order signal.
            self.xattn = CrossAttentionBlock(d, n_layers, n_heads, dropout)
            self.bilinear = BilinearBlock(d, n_views, rank)
            self.gate = GateBlock(d, n_views)
            self.out_dim = d + self.bilinear.out_dim

    def view_weights(self):
        return None if self.gate is None else self.gate.last_weights

    def forward(self, views):
        if len(views) != self.n_views:
            raise ValueError(f"Expected {self.n_views} views, got {len(views)}")

        if self.mode == "concat":
            return torch.cat(views, dim=1)
        if self.mode == "gated":
            return self.gate(views)
        if self.mode == "xattn":
            return torch.cat(self.xattn(views), dim=1)
        if self.mode == "bilinear":
            return self.bilinear(views)

        refined = self.xattn(views)
        return torch.cat([self.gate(refined), self.bilinear(refined)], dim=1)


def build_fusion(mode="proposed", n_views=3, d=256, **kwargs):
    """Construct one fusion strategy by name."""
    return FusionModule(mode, n_views=n_views, d=d, **kwargs)

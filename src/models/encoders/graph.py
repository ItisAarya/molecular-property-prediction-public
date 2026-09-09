"""
src/models/encoders/graph.py

Edge-aware graph encoder (GINE) for molecular graphs.

WHAT THIS FIXES
---------------
`scripts/make_graphs.py` computes seven bond features per edge -- bond order (single /
double / triple / aromatic), conjugation, ring membership and stereochemistry -- and stores
them as `edge_attr`. The inherited GNN then calls `GINConv(x, edge_index)`, which takes
only node features and the connectivity. Every one of those seven dimensions was computed
and thrown away.

That is not a small omission for molecules. Bond order is most of what distinguishes
benzene from cyclohexane; conjugation drives much of the electronic behaviour these assays
respond to; stereochemistry can flip a drug from active to inert. A model that sees only
"these two atoms are connected" is working from a strictly weaker description of the
molecule than the featuriser already produced.

`GINEConv` is the edge-aware form of `GINConv`: it adds a learned projection of the edge
features to the neighbour's node features before the aggregation, so bond information
enters every message.

THE OTHER CHANGES, AND WHY THEY ARE SEPARABLE
---------------------------------------------
Three further departures from the inherited 2-layer GIN, each independently switchable so
Phase 5 can attribute the gain rather than reporting one lump:

  depth      2 -> 4 layers. Each layer extends the receptive field by one bond, so 2 layers
             see only a 2-bond neighbourhood -- too small for a ring plus its substituents.
             Depth is capped around 4-5 because GNNs oversmooth: with too many rounds every
             node converges to the same vector and the molecule loses its structure.

  residual   Skip connections around each layer. Cheap, and the standard remedy for the
             oversmoothing that depth invites.

  readout    mean-pool -> mean-pool concatenated with sum-pool. Mean alone is size-invariant,
             so it cannot distinguish a small molecule from a large one with the same average
             atom environment; sum carries that magnitude. Concatenating keeps both.

Jumping Knowledge (concatenating every layer's output, not just the last) is available via
`jk=True` -- it lets the readout draw on shallow and deep views at once, which helps when
different tasks need different receptive fields, as in multi-task Tox21.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GINEConv, global_add_pool, global_mean_pool


class GINEEncoder(nn.Module):
    """
    Stack of GINEConv layers producing one embedding per molecule.

    Args:
        in_dim:    atom feature width (34 for this featuriser)
        edge_dim:  bond feature width (7)
        hidden:    channel width per layer
        n_layers:  message-passing rounds; each adds one bond of receptive field
        dropout:   applied to node features before readout
        residual:  skip connection around each layer
        jk:        concatenate every layer's output instead of using only the last
        readout:   "mean", "sum", or "mean+sum"
    """

    def __init__(self, in_dim, edge_dim=7, hidden=256, n_layers=4, dropout=0.3,
                 residual=True, jk=False, readout="mean+sum"):
        super().__init__()
        self.n_layers = n_layers
        self.residual = residual
        self.jk = jk
        self.readout = readout
        self.edge_dim = edge_dim

        # Project atom features to the working width once, so every layer is hidden->hidden
        # and a residual connection is a plain addition with no shape juggling.
        self.input_proj = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden)
            )
            # edge_dim lets GINEConv learn its own projection of the bond features up to
            # the node width, which is the whole point of using GINE over GIN.
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.norms.append(BatchNorm(hidden))

        self.dropout = nn.Dropout(dropout)

        n_pool = 2 if readout == "mean+sum" else 1
        width = hidden * (n_layers if jk else 1)
        self.out_dim = width * n_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr

        # A molecule with no bonds (a lone atom) yields an empty edge set. GINEConv still
        # needs edge_attr to have the right second dimension for its projection, even when
        # there are zero rows to project.
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.zeros(
                (edge_index.size(1), self.edge_dim), device=x.device, dtype=x.dtype
            )

        h = self.input_proj(x)
        per_layer = []

        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            if self.residual:
                h = h + h_in
            per_layer.append(h)

        h = torch.cat(per_layer, dim=-1) if self.jk else h
        h = self.dropout(h)

        if self.readout == "mean":
            return global_mean_pool(h, batch)
        if self.readout == "sum":
            return global_add_pool(h, batch)
        return torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=-1)


class GINEncoder(nn.Module):
    """
    The inherited 2-layer GIN, kept as the Phase 1 ablation reference.

    Preserved deliberately: the Phase 5 table needs to separate "we used bond features"
    from "we used a deeper network", and that requires the original architecture to still
    be runnable through the current training loop.
    """

    def __init__(self, in_dim, hidden=256, dropout=0.3, **_):
        super().__init__()
        from torch_geometric.nn import GINConv

        self.conv1 = GINConv(nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden)))
        self.bn1 = BatchNorm(hidden)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden)))
        self.bn2 = BatchNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        return global_mean_pool(self.dropout(x), batch)


class AttentiveFPEncoder(nn.Module):
    """
    AttentiveFP (Xiong et al. 2020) as an external baseline, via PyTorch Geometric.

    This is the second of the two standard external baselines `02_ENHANCEMENT_PLAN.md` §7
    asks for. It is wrapped as an *encoder* rather than run as a standalone model on
    purpose: it then trains through the same loop, the same splits, the same 256-d
    projection and the same head as every other view in this project. A baseline trained by
    its own author's recipe and compared against ours would confound the architecture with
    the training protocol, which is the exact failure this project exists to document.

    What is genuinely AttentiveFP here is the message passing and the readout: the GRU-based
    node update over attended neighbourhoods, and the `num_timesteps` rounds of
    graph-level attention pooling that replace a mean or sum. `out_channels` is set to the
    common embedding width so the trailing linear layer acts as the projection every other
    encoder gets.

    The published defaults for MoleculeNet are 2-3 layers and 2 timesteps; those are the
    defaults here, and like every other model in this project they are held fixed rather
    than tuned, so the comparison is of architectures under one budget.
    """

    def __init__(self, in_dim, hidden=256, layers=3, timesteps=2, dropout=0.2,
                 edge_dim=7, **_):
        super().__init__()
        from torch_geometric.nn.models import AttentiveFP

        self.edge_dim = edge_dim
        self.net = AttentiveFP(in_channels=in_dim, hidden_channels=hidden,
                               out_channels=hidden, edge_dim=edge_dim,
                               num_layers=layers, num_timesteps=timesteps,
                               dropout=dropout)
        self.out_dim = hidden

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        # Same lone-atom guard as GINEEncoder: no bonds means no edge rows, but the edge
        # projection still needs the right width.
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim),
                                    device=x.device, dtype=x.dtype)
        return self.net(x, edge_index, edge_attr, batch)


def build_graph_encoder(name="gine", in_dim=34, **kwargs):
    """Factory so a config file can name an encoder without importing the class."""
    if name == "gine":
        return GINEEncoder(in_dim=in_dim, **kwargs)
    if name == "gin":
        return GINEncoder(in_dim=in_dim, **kwargs)
    if name == "attentivefp":
        return AttentiveFPEncoder(in_dim=in_dim, **kwargs)
    raise ValueError(f"unknown graph encoder: {name}")

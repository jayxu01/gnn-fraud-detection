"""Model definitions: MLP (non-graph baseline) / GCN / GraphSAGE / GAT.

Shared interface: forward(x, edge_index) -> logits [N, 1] (single logit, positive class = illicit).
MLP ignores edge_index and serves as the non-graph baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv


class MLP(nn.Module):
    """Non-graph baseline: uses node features only, no message passing."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def embed(self, x, edge_index=None):
        """Hidden representation before the output layer (used for t-SNE)."""
        for lin in self.lins:
            x = F.relu(lin(x))
        return x  # [N, hidden]

    def forward(self, x, edge_index=None):
        for lin in self.lins:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x)  # [N, 1]


class GCN(nn.Module):
    """Two-layer GCN."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.out = GCNConv(hidden_dim, 1)
        self.dropout = dropout

    def embed(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x  # [N, hidden]

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)  # [N, 1]


class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE (mean aggregation)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)])
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.out = SAGEConv(hidden_dim, 1)
        self.dropout = dropout

    def embed(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x  # [N, hidden]

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)  # [N, 1]


class GAT(nn.Module):
    """Multi-head GAT with a single-head final layer."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList(
            [GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)]
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        # single head on the last layer, outputs [N, 1]
        self.out = GATConv(hidden_dim * heads, 1, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def embed(self, x, edge_index):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        return x  # [N, hidden*heads]

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)  # [N, 1]


def build_model(name: str, in_dim: int, hidden_dim: int, num_layers: int,
                dropout: float, gat_heads: int = 4) -> nn.Module:
    """Build a model by name."""
    if name == "MLP":
        return MLP(in_dim, hidden_dim, num_layers, dropout)
    if name == "GCN":
        return GCN(in_dim, hidden_dim, num_layers, dropout)
    if name == "GraphSAGE":
        return GraphSAGE(in_dim, hidden_dim, num_layers, dropout)
    if name == "GAT":
        return GAT(in_dim, hidden_dim, num_layers, gat_heads, dropout)
    raise ValueError(f"Unknown model name: {name}")

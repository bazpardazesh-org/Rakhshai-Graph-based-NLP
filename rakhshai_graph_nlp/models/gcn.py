"""PyTorch Geometric-based Graph Convolutional Network classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCNClassifier(nn.Module):
    """Two-layer GCN implemented with PyTorch Geometric."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    @torch.no_grad()
    def predict(self, data: Data) -> torch.Tensor:
        self.eval()
        logits = self(data)
        return torch.softmax(logits, dim=-1).argmax(dim=-1)

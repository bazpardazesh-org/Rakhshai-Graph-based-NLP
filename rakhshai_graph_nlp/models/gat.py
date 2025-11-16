"""Graph Attention utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from ..graphs.graph import Graph


class GATLayer:
    """Legacy NumPy-based attention layer used by :mod:`tasks.summarization`."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rng: np.random.Generator | None = None,
        leaky_relu_negative_slope: float = 0.2,
    ):
        rng = rng or np.random.default_rng()
        self.in_dim = in_dim
        self.out_dim = out_dim
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.a = rng.uniform(-limit, limit, size=(2 * out_dim, 1))
        self.negative_slope = leaky_relu_negative_slope

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.negative_slope * x)

    def forward(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        H = X @ self.W
        adjacency = graph.adjacency
        e = np.full((n, n), -np.inf)
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] != 0 or i == j:
                    concatenated = np.concatenate([H[i], H[j]])
                    e[i, j] = self._leaky_relu((concatenated @ self.a).item())
        alpha = np.zeros_like(e)
        for i in range(n):
            row = e[i]
            finite_mask = row != -np.inf
            if not np.any(finite_mask):
                continue
            max_score = np.max(row[finite_mask])
            exps = np.exp(row[finite_mask] - max_score)
            alpha[i, finite_mask] = exps / np.sum(exps)
        H_out = np.zeros_like(H)
        for i in range(n):
            for j in range(n):
                if alpha[i, j] > 0:
                    H_out[i] += alpha[i, j] * H[j]
        return H_out


class GATClassifier(nn.Module):
    """Graph Attention Network classifier based on PyTorch Geometric."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        heads: int = 4,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    @torch.no_grad()
    def predict(self, data: Data) -> torch.Tensor:
        self.eval()
        logits = self(data)
        return torch.softmax(logits, dim=-1).argmax(dim=-1)

"""GraphSAGE implementations."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from ..graphs.graph import Graph


class GraphSAGELayer:
    """A single GraphSAGE aggregation layer using mean aggregator."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_samples: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_samples = num_samples
        rng = rng or np.random.default_rng()
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W_self = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.W_neigh = rng.uniform(-limit, limit, size=(in_dim, out_dim))

    def forward(
        self, graph: Graph, X: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        adjacency = graph.adjacency
        n = X.shape[0]
        if self.num_samples is None:
            degrees = adjacency.sum(axis=1, keepdims=True)
            degrees = np.where(degrees > 0, degrees, 1)
            mean_neigh = (adjacency @ X) / degrees
            return (X @ self.W_self) + (mean_neigh @ self.W_neigh)

        H = np.zeros((n, self.out_dim))
        for i in range(n):
            neighbours = np.where(adjacency[i] > 0)[0].tolist()
            if self.num_samples is not None and len(neighbours) > self.num_samples:
                neighbours = rng.choice(
                    neighbours, size=self.num_samples, replace=False
                ).tolist()
            if neighbours:
                neigh_feat = X[neighbours]
                mean_neigh = neigh_feat.mean(axis=0)
            else:
                mean_neigh = np.zeros(self.in_dim)
            self_feat = X[i]
            H[i] = (self_feat @ self.W_self) + (mean_neigh @ self.W_neigh)
        return H


class GraphSAGE:
    """A multi-layer GraphSAGE model for embedding extraction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_samples: Optional[int] = None,
        activation: Optional[callable] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.rng = rng or np.random.default_rng()
        dims = [input_dim] + hidden_dims
        self.layers = [
            GraphSAGELayer(dims[i], dims[i + 1], num_samples=num_samples, rng=self.rng)
            for i in range(len(dims) - 1)
        ]
        self.activation = activation or (lambda x: np.maximum(x, 0))

    def forward(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        h = X
        for i, layer in enumerate(self.layers):
            h = layer.forward(graph, h, rng=self.rng)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h


class GraphSAGEClassifier(nn.Module):
    """Two-layer GraphSAGE classifier for node prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
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

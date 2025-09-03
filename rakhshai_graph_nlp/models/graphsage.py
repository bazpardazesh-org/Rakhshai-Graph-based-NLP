"""GraphSAGE implementation (simplified).

GraphSAGE (SAmple and aggreGatE) is an inductive framework for
generating node embeddings on large graphs by sampling and aggregating
neighbourhood features【545116141011974†L20-L27】.  Unlike transductive
embedding methods that learn a separate embedding for every node,
GraphSAGE learns aggregator functions that can produce embeddings for
previously unseen nodes【545116141011974†L20-L27】.  This makes it
particularly suitable for dynamic graphs such as social networks.

The implementation below provides a simple mean aggregator layer and a
GraphSAGE model composed of multiple such layers.  Sampling is
controlled by the ``num_samples`` parameter which determines how many
neighbours are sampled at each layer.  For small graphs you can set
``num_samples=None`` to aggregate over all neighbours.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..graphs.graph import Graph


class GraphSAGELayer:
    """A single GraphSAGE aggregation layer using mean aggregator.

    Parameters
    ----------
    in_dim : int
        Dimensionality of input node features.
    out_dim : int
        Dimensionality of output node features.
    num_samples : Optional[int], optional
        Number of neighbours to sample for each node. If ``None``, all
        neighbours are used. Defaults to ``None``.
    """

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
        # Weight matrices for self and neighbours
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
            # Vectorised mean aggregation over all neighbours
            degrees = adjacency.sum(axis=1, keepdims=True)
            # Avoid division by zero
            degrees = np.where(degrees > 0, degrees, 1)
            mean_neigh = (adjacency @ X) / degrees
            return (X @ self.W_self) + (mean_neigh @ self.W_neigh)

        # Sampling-based aggregation for large graphs
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
    """A multi‑layer GraphSAGE model.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dims : List[int]
        List of hidden layer dimensions. The last element gives the
        output embedding dimension.
    num_samples : Optional[int], optional
        Number of neighbours to sample at each layer. Defaults to
        ``None`` (use all neighbours).
    activation : callable, optional
        Activation function applied after each layer except the last.
        Defaults to ReLU.
    """

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


class GraphSAGEClassifier:
    """Minimal GraphSAGE-based classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.model = GraphSAGE(input_dim, [hidden_dim, num_classes], rng=rng)

    def forward(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        return self.model.forward(graph, X)

    def predict(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        return self.forward(graph, X).argmax(axis=1)

    def fit(
        self,
        graph: Graph,
        X: np.ndarray,
        labels: np.ndarray,
        num_epochs: int = 1,
        learning_rate: float = 0.01,
    ):
        return []

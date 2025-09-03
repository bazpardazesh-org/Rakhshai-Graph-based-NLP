"""Graph Attention Network (GAT) implementation (simplified).

This module provides a minimal implementation of a Graph Attention
Network layer based on the formulation of Veličković et al. (2018).
Each node computes attention scores to its neighbours and aggregates
their features accordingly.  We provide only a single attention head
without multi‑head concatenation.  The implementation is kept simple
and does not include optimised GPU operations.  Its primary purpose
is to serve as a reference for research and experimentation.

For more details on the GAT architecture, see the original paper.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..graphs.graph import Graph


class GATLayer:
    """A single‑head Graph Attention layer.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input features.
    out_dim : int
        Dimensionality of the output features.
    leaky_relu_negative_slope : float, optional
        Negative slope of the LeakyReLU used in the attention mechanism.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rng: Optional[np.random.Generator] = None,
        leaky_relu_negative_slope: float = 0.2,
    ):
        rng = rng or np.random.default_rng()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Weight matrices
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.a = rng.uniform(-limit, limit, size=(2 * out_dim, 1))
        self.negative_slope = leaky_relu_negative_slope

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.negative_slope * x)

    def forward(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        """Apply the GAT layer to the input features.

        Parameters
        ----------
        graph : Graph
            Input graph.
        X : np.ndarray
            Node feature matrix of shape ``(n_nodes, in_dim)``.

        Returns
        -------
        np.ndarray
            Output node features of shape ``(n_nodes, out_dim)``.
        """
        n = X.shape[0]
        # Linear transformation
        H = X @ self.W  # shape (n, out_dim)
        # Prepare attention coefficients
        # For each edge (i, j), compute e_ij = LeakyReLU(a^T [h_i || h_j])
        adjacency = graph.adjacency
        e = np.full((n, n), -np.inf)
        # We set -inf for non‑edges so that softmax yields zero attention
        # Compute raw attention scores only for existing edges (including self‑loops)
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] != 0 or i == j:
                    # Concatenate h_i and h_j
                    concatenated = np.concatenate([H[i], H[j]])
                    score = self._leaky_relu((concatenated @ self.a).item())
                    e[i, j] = score
        # Normalise attention coefficients with softmax along rows
        # For each node i, alpha_ij = softmax_j(e_ij)
        alpha = np.zeros_like(e)
        for i in range(n):
            # softmax over neighbours
            row = e[i]
            # subtract max for numerical stability
            finite_mask = row != -np.inf
            if not np.any(finite_mask):
                continue
            max_score = np.max(row[finite_mask])
            exps = np.exp(row[finite_mask] - max_score)
            sum_exps = np.sum(exps)
            alpha[i, finite_mask] = exps / sum_exps
        # Aggregate
        H_out = np.zeros_like(H)
        for i in range(n):
            for j in range(n):
                if alpha[i, j] > 0:
                    H_out[i] += alpha[i, j] * H[j]
        return H_out


class GATClassifier:
    """Minimal classifier using a single GAT layer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        rng: Optional[np.random.Generator] = None,
    ):
        self.layer = GATLayer(input_dim, hidden_dim, rng=rng)
        rng = rng or np.random.default_rng()
        limit = np.sqrt(6 / (hidden_dim + num_classes))
        self.W = rng.uniform(-limit, limit, size=(hidden_dim, num_classes))

    def forward(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        H = self.layer.forward(graph, X)
        return H @ self.W

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
        # Dummy training loop for API compatibility
        return []

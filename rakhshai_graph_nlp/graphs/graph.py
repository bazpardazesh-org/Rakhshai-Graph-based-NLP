"""Simple graph data structure for Rakhshai Graph-based NLP.

This module defines a lightweight graph class that stores a list of
nodes, an adjacency matrix and optional node metadata. It is
intentionally minimal to avoid dependencies on external libraries like
NetworkX. The adjacency matrix is stored as a NumPy array. For
weighted graphs, the matrix entries represent the edge weights; for
unweighted graphs, the matrix entries are 0 or 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np


@dataclass
class Graph:
    """A simple graph that may be directed or undirected.

    Parameters
    ----------
    nodes : Sequence[Any]
        A sequence of node identifiers. The order of this sequence
        determines the ordering of the adjacency matrix rows and
        columns.
    adjacency : np.ndarray
        A square NumPy array of shape ``(n_nodes, n_nodes)`` representing
        the adjacency matrix of the graph. For undirected graphs the
        matrix should be symmetric; this is not strictly enforced.
    node_types : Optional[Sequence[str]]
        Optional sequence specifying a type for each node. This can be
        used to distinguish between different categories of nodes
        (e.g. words versus documents) when training heterogeneous
        graph neural networks.
    node_features : Optional[np.ndarray]
        Optional matrix of node feature vectors. Each row corresponds
        to a node and columns represent feature dimensions.
    directed : bool, optional
        Whether the graph is directed. Defaults to ``False``.
    """

    nodes: Sequence[Any]
    adjacency: np.ndarray
    node_types: Optional[Sequence[str]] = None
    node_features: Optional[np.ndarray] = None
    directed: bool = False

    def __post_init__(self):
        if not isinstance(self.adjacency, np.ndarray):
            raise TypeError("adjacency must be a NumPy array")
        n = len(self.nodes)
        if self.adjacency.shape != (n, n):
            raise ValueError(
                f"Adjacency shape {self.adjacency.shape} does not match number of nodes {n}"
            )
        if self.node_types is not None and len(self.node_types) != n:
            raise ValueError(
                "node_types length must match number of nodes"
            )
        if self.node_features is not None and self.node_features.shape[0] != n:
            raise ValueError(
                "node_features number of rows must match number of nodes"
            )

    def add_self_loops(self, weight: float = 1.0) -> None:
        """Add self‑loops to all nodes.

        Self‑loops are sometimes required when applying graph
        convolution layers. This method modifies the adjacency matrix
        in place.

        Parameters
        ----------
        weight : float, optional
            The weight of each self‑loop. Defaults to 1.0.
        """
        np.fill_diagonal(self.adjacency, self.adjacency.diagonal() + weight)

    def degree_matrix(self) -> np.ndarray:
        """Return the degree matrix of the graph.

        The degree matrix is a diagonal matrix where each diagonal
        entry is the sum of the corresponding row of the adjacency
        matrix.

        Returns
        -------
        np.ndarray
            The degree matrix.
        """
        degrees = self.adjacency.sum(axis=1)
        return np.diag(degrees)

    def normalized_adjacency(self) -> np.ndarray:
        """Compute a normalised adjacency matrix.

        For undirected graphs this computes the symmetric normalisation
        ``D^{-1/2} A D^{-1/2}`` commonly used by GCNs.  For directed
        graphs a row‑normalisation ``D^{-1} A`` is applied, where ``D``
        contains the out‑degrees of each node.

        Returns
        -------
        np.ndarray
            The normalised adjacency matrix.
        """
        deg = self.adjacency.sum(axis=1)
        if self.directed:
            with np.errstate(divide="ignore"):
                inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
            return np.diag(inv_deg) @ self.adjacency
        # Undirected case
        with np.errstate(divide="ignore"):
            inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        return D_inv_sqrt @ self.adjacency @ D_inv_sqrt

    def to_edge_list(self) -> list[tuple]:
        """Convert the graph into a list of edges.

        Returns
        -------
        list[tuple]
            A list of tuples ``(u, v, w)`` representing edges and
            weights. Both ``u`` and ``v`` are node identifiers drawn
            from ``self.nodes``, and ``w`` is the corresponding
            weight from the adjacency matrix.
        """
        edges = []
        n = len(self.nodes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = self.adjacency[i, j]
                if w != 0:
                    if self.directed or j > i:
                        edges.append((self.nodes[i], self.nodes[j], w))
        return edges

    def copy(self) -> "Graph":
        """Return a deep copy of the graph."""
        return Graph(
            nodes=list(self.nodes),
            adjacency=self.adjacency.copy(),
            node_types=list(self.node_types) if self.node_types is not None else None,
            node_features=self.node_features.copy() if self.node_features is not None else None,
            directed=self.directed,
        )
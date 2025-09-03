"""Social network analysis utilities.

Graph neural networks are well‑suited for analysing social networks
where nodes represent users and messages, and edges encode follower
relationships or interactions.  GraphSAGE is particularly useful
because it can generate embeddings for unseen nodes by sampling and
aggregating neighbourhood information【545116141011974†L20-L27】.

This module implements functions to compute node embeddings using a
GraphSAGE model.  These embeddings can then be used for tasks such as
community detection, influence estimation or link prediction.  The
functions here are intentionally lightweight; for large networks,
consider using dedicated graph processing frameworks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..graphs.graph import Graph
from ..models.graphsage import GraphSAGE


def compute_social_embeddings(
    graph: Graph,
    node_features: np.ndarray,
    hidden_dims: list[int] = [32, 16],
    num_samples: Optional[int] = None,
) -> np.ndarray:
    """Compute node embeddings for a social network using GraphSAGE.

    Parameters
    ----------
    graph : Graph
        The social network graph where nodes might represent users or
        posts.
    node_features : np.ndarray
        Matrix of shape ``(n_nodes, feature_dim)`` containing
        features for each node. These could be user profile features,
        textual embeddings, etc.
    hidden_dims : list[int], optional
        Hidden layer dimensions for the GraphSAGE model. Defaults to
        ``[32, 16]``.
    num_samples : Optional[int], optional
        Number of neighbours to sample during aggregation.  If
        ``None``, all neighbours are used.  Setting this to a small
        number makes the method scalable to large graphs.  Defaults
        to ``None``.

    Returns
    -------
    np.ndarray
        Matrix of node embeddings of shape ``(n_nodes, hidden_dims[-1])``.
    """
    input_dim = node_features.shape[1]
    sage = GraphSAGE(input_dim, hidden_dims, num_samples=num_samples)
    embeddings = sage.forward(graph, node_features)
    return embeddings
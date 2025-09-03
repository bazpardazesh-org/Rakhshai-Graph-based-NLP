"""Text classification tasks using graphs.

This module provides utilities for training graph neural networks on
graphs derived from Persian text.  The most common scenario is to
construct a co‑occurrence or document graph and then perform node or
graph classification.  Our implementation focuses on node
classification using the Graph Convolutional Network (GCN) described
by Kipf & Welling and used in the TextGCN model【136383271440271†L816-L825】.

Example
-------
Suppose you have a corpus of documents and corresponding labels.  You
can build a corpus‑level co‑occurrence graph using
``rakhshai_graph_nlp.graphs.co_occurrence.build_cooccurrence_graph``, derive
simple node features (e.g. one‑hot word vectors or TF‑IDF vectors),
and then train a ``GCNClassifier``::

    >>> from rakhshai_graph_nlp.graphs.co_occurrence import build_cooccurrence_graph
    >>> from rakhshai_graph_nlp.tasks.classification import train_gcn_classifier
    >>> docs = ["خبر سیاسی", "ورزش", "هنر"]
    >>> labels = [0, 1, 2]  # example class indices
    >>> # Tokenise and build graph
    >>> from rakhshai_graph_nlp.features.tokenizer import tokenize
    >>> tokenised = [tokenize(d) for d in docs]
    >>> graph = build_cooccurrence_graph(tokenised)
    >>> # Simple identity features for words
    >>> import numpy as np
    >>> X = np.eye(len(graph.nodes))
    >>> clf, losses = train_gcn_classifier(graph, X, np.array(labels), hidden_dim=8, num_epochs=50)
    >>> preds = clf.predict(graph, X)
    >>> print(preds)

This simplistic example illustrates how to set up the pipeline.  In
practice, you will want to use richer features and possibly a corpus
graph with both word and document nodes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..graphs.graph import Graph
from ..models.gcn import GCNClassifier


def train_gcn_classifier(
    graph: Graph,
    X: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
    hidden_dim: int = 16,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0,
) -> Tuple[GCNClassifier, List[float]]:
    """Train a two‑layer GCN classifier on a graph.

    Parameters
    ----------
    graph : Graph
        The input graph.
    X : np.ndarray
        Node feature matrix of shape ``(n_nodes, input_dim)``.
    labels : np.ndarray
        Ground truth class indices of shape ``(n_nodes,)``.
    mask : np.ndarray, optional
        Boolean array indicating which nodes to include in the training
        loss. If ``None``, all nodes are used. This is useful for
        semi‑supervised learning where only a subset of nodes have
        labels.
    hidden_dim : int, optional
        Dimensionality of the hidden layer. Defaults to ``16``.
    num_epochs : int, optional
        Number of training epochs. Defaults to ``200``.
    learning_rate : float, optional
        Learning rate for gradient descent. Defaults to ``0.01``.
    weight_decay : float, optional
        Weight decay coefficient for L2 regularisation. Defaults to
        ``0.0``.

    Returns
    -------
    Tuple[GCNClassifier, List[float]]
        The trained classifier and a list of loss values per epoch.
    """
    input_dim = X.shape[1]
    num_classes = int(labels.max()) + 1
    clf = GCNClassifier(input_dim, hidden_dim, num_classes)
    losses = clf.fit(
        graph,
        X,
        labels,
        mask=mask,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return clf, losses
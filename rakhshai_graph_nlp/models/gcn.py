r"""Graph Convolutional Network (GCN) implementation.

This module provides a minimal implementation of a Graph Convolutional
Network for node and graph classification using NumPy.  The
implementation follows the formulation introduced in Kipf & Welling
and used in TextGCN【136383271440271†L816-L824】.  For a graph with adjacency
matrix ``A`` and input node features ``X``, a single graph convolution
layer computes ``\hat{A} X W``, where ``\hat{A} = D^{-1/2} (A + I)
D^{-1/2}`` is the symmetrically normalised adjacency matrix with
self‑loops and ``W`` is a trainable weight matrix.

The ``GCNClassifier`` class stacks two such layers with a ReLU
activation in between and a softmax output layer for multi‑class
classification.  Training is performed using simple gradient descent.

Note
----
This implementation is intended for educational use and small graphs.
It is not optimised for large graphs or GPU acceleration.  For
production use, we recommend more efficient libraries.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..graphs.graph import Graph


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


class GCNLayer:
    """A single graph convolution layer.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input features.
    out_dim : int
        Dimensionality of the output features.
    """

    def __init__(self, in_dim: int, out_dim: int, rng: Optional[np.random.Generator] = None):
        self.in_dim = in_dim
        self.out_dim = out_dim
        rng = rng or np.random.default_rng()
        # Weight initialisation (Xavier/Glorot)
        limit = np.sqrt(6 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))

    def forward(self, X: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
        """Apply the graph convolution to input features.

        Parameters
        ----------
        X : np.ndarray
            Node feature matrix of shape ``(n_nodes, in_dim)``.
        A_norm : np.ndarray
            Normalised adjacency matrix of shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        np.ndarray
            The transformed node features of shape ``(n_nodes, out_dim)``.
        """
        return A_norm @ (X @ self.W)


class GCNClassifier:
    """Two‑layer GCN for node classification.

    This classifier applies two graph convolution layers followed by a
    softmax. A ReLU activation is used between the layers. The
    classifier can be trained via the ``fit`` method using gradient
    descent.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        self.layer1 = GCNLayer(input_dim, hidden_dim, rng=self.rng)
        self.layer2 = GCNLayer(hidden_dim, num_classes, rng=self.rng)

    def forward(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        """Compute the logits for each node.

        Parameters
        ----------
        graph : Graph
            The input graph. Its adjacency will be normalised with
            self‑loops before the forward pass.
        X : np.ndarray
            Node features of shape ``(n_nodes, input_dim)``.

        Returns
        -------
        np.ndarray
            Logits for each node and class of shape ``(n_nodes, num_classes)``.
        """
        # Add self loops and normalise using Graph utility so directed graphs
        # are handled correctly.
        A = graph.adjacency.copy()
        np.fill_diagonal(A, A.diagonal() + 1)
        tmp_graph = Graph(
            nodes=graph.nodes,
            adjacency=A,
            node_types=graph.node_types,
            node_features=graph.node_features,
            directed=graph.directed,
        )
        A_norm = tmp_graph.normalized_adjacency()
        h = self.layer1.forward(X, A_norm)
        h = relu(h)
        logits = self.layer2.forward(h, A_norm)
        return logits

    def predict(self, graph: Graph, X: np.ndarray) -> np.ndarray:
        """Predict class labels for each node.

        Returns
        -------
        np.ndarray
            Array of predicted class indices.
        """
        logits = self.forward(graph, X)
        probs = softmax(logits)
        return probs.argmax(axis=1)

    def fit(
        self,
        graph: Graph,
        X: np.ndarray,
        labels: np.ndarray,
        mask: Optional[np.ndarray] = None,
        num_epochs: int = 200,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ) -> list[float]:
        """Train the GCN classifier using gradient descent.

        Parameters
        ----------
        graph : Graph
            Input graph.
        X : np.ndarray
            Node features of shape ``(n_nodes, input_dim)``.
        labels : np.ndarray
            Ground truth labels of shape ``(n_nodes,)`` with integer
            class indices.
        mask : np.ndarray, optional
            Boolean mask array of shape ``(n_nodes,)`` indicating which
            nodes participate in the loss (e.g. training nodes in a
            semi‑supervised setting). If ``None``, all nodes are used.
        num_epochs : int, optional
            Number of gradient descent epochs. Defaults to ``200``.
        learning_rate : float, optional
            Learning rate for gradient descent. Defaults to ``0.01``.
        weight_decay : float, optional
            Weight decay (L2 regularisation) coefficient. Defaults to
            ``0.0``.

        Returns
        -------
        List[float]
            A list of loss values per epoch.
        """
        if mask is None:
            mask = np.ones_like(labels, dtype=bool)
        # Precompute normalised adjacency with self loops
        A = graph.adjacency.copy()
        np.fill_diagonal(A, A.diagonal() + 1)
        tmp_graph = Graph(
            nodes=graph.nodes,
            adjacency=A,
            node_types=graph.node_types,
            node_features=graph.node_features,
            directed=graph.directed,
        )
        A_norm = tmp_graph.normalized_adjacency()
        # Extract weights for optimisation
        W1 = self.layer1.W
        W2 = self.layer2.W
        losses: list[float] = []
        n_classes = W2.shape[1]
        for epoch in range(num_epochs):
            # Forward propagation
            h1 = A_norm @ (X @ W1)
            h1 = relu(h1)
            logits = A_norm @ (h1 @ W2)
            # Compute probabilities for masked nodes only
            logits_masked = logits[mask]
            labels_masked = labels[mask]
            probs = softmax(logits_masked)
            # One‑hot labels
            y_onehot = np.eye(n_classes)[labels_masked]
            # Cross‑entropy loss
            log_probs = np.log(probs + 1e-9)
            loss = -np.mean((y_onehot * log_probs).sum(axis=1))
            # Regularisation
            if weight_decay > 0.0:
                loss += weight_decay / 2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
            losses.append(float(loss))
            # Backpropagation
            # Gradient of loss w.r.t logits for masked nodes
            grad_logits_masked = probs - y_onehot
            # Expand gradient to full nodes: zeros for unmasked
            grad_logits_full = np.zeros_like(logits)
            grad_logits_full[mask] = grad_logits_masked
            # Gradients for W2
            grad_W2 = h1.T @ (A_norm.T @ grad_logits_full)
            # Gradient through W2 to h1
            grad_h1 = (grad_logits_full @ W2.T)
            # Propagate through adjacency
            grad_h1 = A_norm.T @ grad_h1
            # Derivative of ReLU
            grad_h1[h1 <= 0] = 0
            # Gradients for W1
            grad_W1 = X.T @ (A_norm.T @ grad_h1)
            # Apply weight decay
            if weight_decay > 0.0:
                grad_W1 += weight_decay * W1
                grad_W2 += weight_decay * W2
            # Parameter update
            W1 -= learning_rate * grad_W1
            W2 -= learning_rate * grad_W2
        return losses
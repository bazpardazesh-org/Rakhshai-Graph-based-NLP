"""Text classification tasks using PyTorch Geometric models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Optional

import numpy as np
import torch
from torch import nn

from ..features.pyg_data import (
    build_feature_matrix,
    graph_to_data,
    preprocess_persian_corpus,
)
from ..graphs.graph import Graph
from ..models.gat import GATClassifier
from ..models.gcn import GCNClassifier
from ..models.graphsage import GraphSAGEClassifier

MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    "gcn": GCNClassifier,
    "graphsage": GraphSAGEClassifier,
    "gat": GATClassifier,
}


def _select_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _prepare_features(
    graph: Graph,
    X: Optional[np.ndarray | torch.Tensor],
    texts: Optional[Sequence[str]],
    *,
    lemmatize: bool,
    embedding_lookup: Optional[Mapping[str, Sequence[float]]],
    use_gpu: bool,
) -> np.ndarray | torch.Tensor:
    if X is not None:
        return X
    if graph.node_features is not None:
        return graph.node_features
    if texts is not None:
        tokens = preprocess_persian_corpus(texts, lemmatize=lemmatize, use_gpu=use_gpu)
        return build_feature_matrix(tokens, embedding_lookup=embedding_lookup)
    return np.eye(len(graph.nodes), dtype=float)


def train_node_classifier(
    graph: Graph,
    labels: np.ndarray,
    *,
    X: Optional[np.ndarray | torch.Tensor] = None,
    mask: Optional[np.ndarray] = None,
    model_type: Literal["gcn", "graphsage", "gat"] = "gcn",
    hidden_dim: int = 64,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    texts: Optional[Sequence[str]] = None,
    lemmatize: bool = False,
    embedding_lookup: Optional[Mapping[str, Sequence[float]]] = None,
    device: str | torch.device = "cpu",
    gat_heads: int = 4,
) -> tuple[nn.Module, list[float]]:
    """Train a node classifier using PyTorch Geometric modules."""

    if labels.shape[0] != len(graph.nodes):
        raise ValueError("labels must match number of graph nodes")

    device_t = _select_device(device)
    features = _prepare_features(
        graph,
        X,
        texts,
        lemmatize=lemmatize,
        embedding_lookup=embedding_lookup,
        use_gpu=device_t.type == "cuda",
    )
    data = graph_to_data(graph, features=features, labels=labels)
    data = data.to(device_t)

    num_classes = int(labels.max()) + 1
    input_dim = data.num_node_features

    if model_type == "gat":
        model = GATClassifier(input_dim, hidden_dim, num_classes, heads=gat_heads, dropout=dropout)
    else:
        model = MODEL_BUILDERS[model_type](input_dim, hidden_dim, num_classes, dropout=dropout)

    model = model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    if mask is None:
        train_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device_t)
    else:
        if mask.shape[0] != data.num_nodes:
            raise ValueError("mask must match number of nodes")
        train_mask = torch.as_tensor(mask, dtype=torch.bool, device=device_t)

    losses: list[float] = []
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return model, losses


def train_gcn_classifier(
    graph: Graph,
    X: np.ndarray,
    labels: np.ndarray,
    mask: Optional[np.ndarray] = None,
    hidden_dim: int = 16,
    num_epochs: int = 200,
    learning_rate: float = 0.01,
    weight_decay: float = 0.0,
    dropout: float = 0.5,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, list[float]]:
    """Backward compatible wrapper for the classic ``train_gcn_classifier`` API."""

    return train_node_classifier(
        graph,
        labels,
        X=X,
        mask=mask,
        model_type="gcn",
        hidden_dim=hidden_dim,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        device=device,
    )

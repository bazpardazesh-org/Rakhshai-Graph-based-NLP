"""Text classification tasks using PyTorch Geometric models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
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
from ..graphs.text_graph import _compute_pmi, _compute_tfidf
from ..models.gat import GATClassifier
from ..models.gcn import GCNClassifier
from ..models.graphsage import GraphSAGEClassifier
from ..metrics import accuracy, macro_f1

MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    "gcn": GCNClassifier,
    "graphsage": GraphSAGEClassifier,
    "gat": GATClassifier,
}


@dataclass
class TextGraphClassifierConfig:
    """Configuration needed to reproduce a fitted text graph classifier."""

    model_type: Literal["gcn", "graphsage", "gat"] = "gcn"
    hidden_dim: int = 64
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    dropout: float = 0.5
    window_size: int = 20
    min_count: int = 1
    smooth_idf: bool = True
    gat_heads: int = 4
    seed: int = 0


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
        model = GATClassifier(
            input_dim,
            hidden_dim,
            num_classes,
            heads=gat_heads,
            dropout=dropout,
        )
    else:
        model = MODEL_BUILDERS[model_type](
            input_dim,
            hidden_dim,
            num_classes,
            dropout=dropout,
        )

    model = model.to(device_t)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
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


def _tokenise_texts(texts: Sequence[str]) -> list[list[str]]:
    return preprocess_persian_corpus(texts, lemmatize=False, use_gpu=False)


def _vocab_from_tokens(
    token_lists: Sequence[Sequence[str]],
    *,
    min_count: int,
) -> list[str]:
    counts: dict[str, int] = {}
    for tokens in token_lists:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    vocab = [token for token, count in counts.items() if count >= min_count]
    if not vocab:
        raise ValueError("No words meet the frequency threshold")
    return vocab


def _build_fixed_vocab_text_graph(
    token_lists: Sequence[Sequence[str]],
    vocab: Sequence[str],
    *,
    window_size: int,
    smooth_idf: bool,
) -> Graph:
    """Build a TextGCN-style graph while keeping the fitted vocabulary stable."""

    if not token_lists:
        raise ValueError("token_lists must not be empty")
    V = len(vocab)
    D = len(token_lists)
    pmi_matrix = _compute_pmi(token_lists, vocab, window_size=window_size)
    tfidf = _compute_tfidf(token_lists, vocab, smooth_idf=smooth_idf)

    adjacency = np.zeros((V + D, V + D), dtype=float)
    adjacency[:V, :V] = pmi_matrix
    adjacency[:V, V:] = tfidf
    adjacency[V:, :V] = tfidf.T
    nodes = [*vocab, *[f"doc_{i}" for i in range(D)]]
    node_types = ["word"] * V + ["doc"] * D
    return Graph(nodes=nodes, adjacency=adjacency, node_types=node_types)


def _fixed_vocab_features(
    token_lists: Sequence[Sequence[str]],
    vocab: Sequence[str],
) -> np.ndarray:
    """Create stable word/doc features in the fitted vocabulary space."""

    vocab_index = {token: idx for idx, token in enumerate(vocab)}
    features = np.zeros((len(vocab) + len(token_lists), len(vocab)), dtype=float)
    features[: len(vocab), :] = np.eye(len(vocab), dtype=float)
    for doc_idx, tokens in enumerate(token_lists):
        row = len(vocab) + doc_idx
        for token in tokens:
            col = vocab_index.get(token)
            if col is not None:
                features[row, col] += 1.0
        norm = np.linalg.norm(features[row])
        if norm > 0:
            features[row] /= norm
    return features


class TextGraphClassifier:
    """End-to-end TextGCN-style classifier for Persian text.

    The class stores the fitted vocabulary, label mapping, graph settings and
    training texts so a saved model can be loaded and used for inference on new
    documents with the same feature space.
    """

    def __init__(
        self,
        model: Literal["gcn", "graphsage", "gat"] = "gcn",
        *,
        hidden_dim: int = 64,
        num_epochs: int = 200,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        dropout: float = 0.5,
        window_size: int = 20,
        min_count: int = 1,
        smooth_idf: bool = True,
        gat_heads: int = 4,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.config = TextGraphClassifierConfig(
            model_type=model,
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            window_size=window_size,
            min_count=min_count,
            smooth_idf=smooth_idf,
            gat_heads=gat_heads,
            seed=seed,
        )
        self.device = _select_device(device)
        self.model_: nn.Module | None = None
        self.vocab_: list[str] | None = None
        self.label_to_id_: dict[str, int] | None = None
        self.id_to_label_: dict[int, str] | None = None
        self.training_texts_: list[str] | None = None
        self.losses_: list[float] = []

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str | int],
    ) -> "TextGraphClassifier":
        """Fit the classifier on labelled documents."""

        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("texts must not be empty")

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.training_texts_ = list(texts)
        token_lists = _tokenise_texts(texts)
        self.vocab_ = _vocab_from_tokens(token_lists, min_count=self.config.min_count)
        graph = _build_fixed_vocab_text_graph(
            token_lists,
            self.vocab_,
            window_size=self.config.window_size,
            smooth_idf=self.config.smooth_idf,
        )
        features = _fixed_vocab_features(token_lists, self.vocab_)

        labels_as_str = [str(label) for label in labels]
        self.label_to_id_ = {
            label: idx for idx, label in enumerate(sorted(set(labels_as_str)))
        }
        self.id_to_label_ = {idx: label for label, idx in self.label_to_id_.items()}
        encoded_doc_labels = np.array(
            [self.label_to_id_[label] for label in labels_as_str], dtype=int
        )

        node_labels = np.zeros(len(graph.nodes), dtype=int)
        doc_start = len(self.vocab_)
        node_labels[doc_start:] = encoded_doc_labels
        train_mask = np.zeros(len(graph.nodes), dtype=bool)
        train_mask[doc_start:] = True

        self.model_, self.losses_ = train_node_classifier(
            graph,
            node_labels,
            X=features,
            mask=train_mask,
            model_type=self.config.model_type,
            hidden_dim=self.config.hidden_dim,
            num_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            dropout=self.config.dropout,
            device=self.device,
            gat_heads=self.config.gat_heads,
        )
        return self

    def _require_fitted(self) -> None:
        if (
            self.model_ is None
            or self.vocab_ is None
            or self.label_to_id_ is None
            or self.id_to_label_ is None
            or self.training_texts_ is None
        ):
            raise ValueError("TextGraphClassifier is not fitted")

    def _predict_ids(self, texts: Sequence[str]) -> np.ndarray:
        self._require_fitted()
        assert self.model_ is not None
        assert self.vocab_ is not None
        assert self.training_texts_ is not None

        all_texts = [*self.training_texts_, *list(texts)]
        token_lists = _tokenise_texts(all_texts)
        graph = _build_fixed_vocab_text_graph(
            token_lists,
            self.vocab_,
            window_size=self.config.window_size,
            smooth_idf=self.config.smooth_idf,
        )
        features = _fixed_vocab_features(token_lists, self.vocab_)
        data = graph_to_data(graph, features=features).to(self.device)
        preds = self.model_.predict(data).cpu().numpy()
        return preds[len(self.vocab_) + len(self.training_texts_) :]

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict labels for new documents."""

        self._require_fitted()
        assert self.id_to_label_ is not None
        pred_ids = self._predict_ids(texts)
        return [self.id_to_label_[int(pred_id)] for pred_id in pred_ids]

    def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[str | int],
    ) -> dict[str, float | int]:
        """Evaluate predictions with accuracy and macro-F1."""

        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self._require_fitted()
        assert self.label_to_id_ is not None
        pred_labels = self.predict(texts)
        y_true = np.array([self.label_to_id_.get(str(label), -1) for label in labels])
        y_pred = np.array([self.label_to_id_[label] for label in pred_labels])
        known = y_true >= 0
        if not np.all(known):
            raise ValueError("labels contain values that were not seen during fit")
        return {
            "count": len(labels),
            "accuracy": accuracy(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
        }

    def save(self, path: str | Path) -> None:
        """Save model weights and pipeline metadata to a directory."""

        self._require_fitted()
        assert self.model_ is not None
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "config": asdict(self.config),
            "vocab": self.vocab_,
            "label_to_id": self.label_to_id_,
            "training_texts": self.training_texts_,
            "losses": self.losses_,
        }
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        torch.save(self.model_.state_dict(), path / "model.pt")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "TextGraphClassifier":
        """Load a saved classifier."""

        path = Path(path)
        with (path / "metadata.json").open(encoding="utf-8") as f:
            metadata = json.load(f)
        config = TextGraphClassifierConfig(**metadata["config"])
        classifier = cls(
            model=config.model_type,
            hidden_dim=config.hidden_dim,
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
            window_size=config.window_size,
            min_count=config.min_count,
            smooth_idf=config.smooth_idf,
            gat_heads=config.gat_heads,
            seed=config.seed,
            device=device,
        )
        classifier.vocab_ = list(metadata["vocab"])
        classifier.label_to_id_ = {
            str(label): int(idx) for label, idx in metadata["label_to_id"].items()
        }
        classifier.id_to_label_ = {
            idx: label for label, idx in classifier.label_to_id_.items()
        }
        classifier.training_texts_ = list(metadata["training_texts"])
        classifier.losses_ = [float(loss) for loss in metadata.get("losses", [])]

        input_dim = len(classifier.vocab_)
        num_classes = len(classifier.label_to_id_)
        if config.model_type == "gat":
            model = GATClassifier(
                input_dim,
                config.hidden_dim,
                num_classes,
                heads=config.gat_heads,
                dropout=config.dropout,
            )
        else:
            model = MODEL_BUILDERS[config.model_type](
                input_dim,
                config.hidden_dim,
                num_classes,
                dropout=config.dropout,
            )
        model.load_state_dict(
            torch.load(path / "model.pt", map_location=classifier.device)
        )
        classifier.model_ = model.to(classifier.device)
        classifier.model_.eval()
        return classifier

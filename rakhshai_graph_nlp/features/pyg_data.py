"""Utilities for bridging NumPy graphs with PyTorch Geometric data structures.

PyTorch Geometric expects sparse edge lists stored inside a
:class:`torch_geometric.data.Data` object instead of the dense adjacency
matrices used throughout the legacy parts of the project.  This module
provides helper functions to convert existing
:class:`~rakhshai_graph_nlp.graphs.graph.Graph` instances into ``Data``
objects and to construct feature matrices from Persian text.  The helper
functions apply the normalisation routines defined in
:mod:`rakhshai_graph_nlp.features.preprocessing` and optionally use the
Stanza lemmatiser when available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from ..graphs.graph import Graph
from .preprocessing import preprocess
from .tokenizer import tokenize

try:  # pragma: no cover - optional dependency
    import stanza
except ImportError:  # pragma: no cover - Stanza is optional
    stanza = None  # type: ignore[assignment]


def preprocess_persian_corpus(
    texts: Sequence[str],
    *,
    lemmatize: bool = False,
    use_gpu: bool = False,
) -> list[list[str]]:
    """Normalise, tokenise and optionally lemmatise Persian text.

    Parameters
    ----------
    texts:
        Raw text snippets. Each entry typically corresponds to a node in a
        graph (for instance a document node in a TextGCN style graph).
    lemmatize:
        If ``True`` and Stanza is installed, use Stanza's Persian pipeline to
        obtain lemmas. Falls back to token strings if Stanza is unavailable.
    use_gpu:
        Forwarded to Stanza so that lemmatisation runs on GPU when CUDA is
        accessible.
    """

    cleaned = [preprocess(text) for text in texts]
    if lemmatize and stanza is not None:
        if not hasattr(preprocess_persian_corpus, "_nlp"):
            preprocess_persian_corpus._nlp = stanza.Pipeline(  # type: ignore[attr-defined]
                lang="fa", processors="tokenize,lemma", use_gpu=use_gpu
            )
        nlp = preprocess_persian_corpus._nlp  # type: ignore[attr-defined]
        processed: list[list[str]] = []
        for text in cleaned:
            doc = nlp(text)
            lemmas: list[str] = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    lemmas.append(word.lemma or word.text)
            processed.append(lemmas)
        return processed
    # Default to rule-based tokeniser
    return [tokenize(text) for text in cleaned]


def build_feature_matrix(
    tokenised_nodes: Sequence[Sequence[str]],
    *,
    embedding_lookup: Optional[Mapping[str, Sequence[float]]] = None,
) -> torch.Tensor:
    """Create a feature matrix for nodes based on token sequences.

    When ``embedding_lookup`` is provided, tokens are mapped to pre-trained
    vectors and averaged to produce a dense representation per node.  If no
    embeddings are supplied, a simple bag-of-words representation is returned.
    """

    if not tokenised_nodes:
        raise ValueError("No tokens supplied for feature construction")

    if embedding_lookup:
        first_vec = next(iter(embedding_lookup.values()))
        dim = len(first_vec)
        rows: list[torch.Tensor] = []
        for tokens in tokenised_nodes:
            vectors = [
                torch.tensor(list(embedding_lookup[token]), dtype=torch.float32)
                for token in tokens
                if token in embedding_lookup
            ]
            if vectors:
                stacked = torch.stack(vectors, dim=0)
                rows.append(stacked.mean(dim=0))
            else:
                rows.append(torch.zeros(dim, dtype=torch.float32))
        return torch.stack(rows, dim=0)

    # Bag-of-words fallback
    vocab: dict[str, int] = {}
    for tokens in tokenised_nodes:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
    features = torch.zeros((len(tokenised_nodes), len(vocab)), dtype=torch.float32)
    for row_idx, tokens in enumerate(tokenised_nodes):
        for token in tokens:
            col = vocab[token]
            features[row_idx, col] += 1.0
    return F.normalize(features, p=2, dim=1)


def graph_to_data(
    graph: Graph,
    *,
    features: Optional[np.ndarray | torch.Tensor] = None,
    labels: Optional[np.ndarray | torch.Tensor] = None,
) -> Data:
    """Convert a :class:`Graph` into a :class:`torch_geometric.data.Data` object."""

    if features is None and graph.node_features is not None:
        features = graph.node_features

    if features is None:
        # Fall back to identity features when nothing else is available.
        features = np.eye(len(graph.nodes), dtype=float)

    if isinstance(features, np.ndarray):
        x = torch.from_numpy(features).float()
    else:
        x = features.float()

    adjacency = graph.adjacency
    src, dst = np.nonzero(adjacency)
    if src.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        edge_weight = torch.tensor(adjacency[src, dst], dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
    data.edge_weight = edge_weight

    if labels is not None:
        if isinstance(labels, np.ndarray):
            data.y = torch.from_numpy(labels).long()
        else:
            data.y = labels.long()

    return data

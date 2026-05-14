"""Graph-based extractive summarisation for Persian text."""

from __future__ import annotations

import re

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

from ..graphs.graph import Graph
from ..models.gat import GATLayer


def split_sentences(text: str) -> list[str]:
    """Split Persian text into sentences."""

    delimiters = r"[\.!؟۔]"
    parts = re.split(delimiters, text)
    return [part.strip() for part in parts if part.strip()]


def _build_sentence_graph(sentences: list[str]) -> tuple[Graph, list[str]]:
    """Build a sentence graph weighted by TF-IDF cosine similarity."""

    if TfidfVectorizer is None or cosine_similarity is None:
        raise ImportError("scikit-learn is required for summarisation")
    if len(sentences) == 0:
        raise ValueError("No sentences provided")
    vectorizer = TfidfVectorizer(stop_words=None)
    features = vectorizer.fit_transform(sentences).astype(float).toarray()
    similarity = cosine_similarity(features)
    np.fill_diagonal(similarity, 0.0)
    graph = Graph(nodes=list(range(len(sentences))), adjacency=similarity)
    return graph, sentences


def _pagerank(
    adjacency: np.ndarray,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Compute PageRank scores for a weighted graph."""

    n = adjacency.shape[0]
    row_sums = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency)
    for i in range(n):
        if row_sums[i] > 0:
            transition[i] = adjacency[i] / row_sums[i]
        else:
            transition[i] = np.ones(n) / n

    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.linalg.norm(new_scores - scores, 1) < tol:
            scores = new_scores
            break
        scores = new_scores
    return scores


def textrank_summarise(text: str, top_k: int = 3) -> str:
    """Generate an extractive summary using TextRank."""

    sentences = split_sentences(text)
    graph, sent_list = _build_sentence_graph(sentences)
    scores = _pagerank(graph.adjacency)
    top_indices = np.argsort(scores)[::-1][:top_k]
    selected = sorted(top_indices)
    return " \n".join(sent_list[i] for i in selected)


def _sentence_tfidf_features(sentences: list[str]) -> np.ndarray:
    if TfidfVectorizer is None:
        raise ImportError("scikit-learn is required for GAT summarisation")
    vectorizer = TfidfVectorizer(stop_words=None)
    return vectorizer.fit_transform(sentences).astype(float).toarray()


class GATSummarizer:
    """Unsupervised GAT-style sentence ranker."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        random_state: int | None = 0,
    ):
        rng = np.random.default_rng(random_state)
        self.gat = GATLayer(input_dim, hidden_dim, rng=rng)
        limit = np.sqrt(6 / (hidden_dim + output_dim))
        self.W = rng.uniform(-limit, limit, size=(hidden_dim, output_dim))

    def summarise(self, graph: Graph, X: np.ndarray, top_k: int = 3) -> np.ndarray:
        """Select top sentence/node indices using graph attention."""

        H = self.gat.forward(graph, X)
        H_proj = H @ self.W
        norms = np.linalg.norm(H_proj, axis=1)
        return np.argsort(norms)[::-1][:top_k]


def gat_summarise(
    text: str,
    top_k: int = 3,
    *,
    hidden_dim: int = 32,
    output_dim: int = 16,
    random_state: int | None = 0,
) -> str:
    """Generate an extractive summary using the GAT sentence ranker."""

    sentences = split_sentences(text)
    graph, sent_list = _build_sentence_graph(sentences)
    features = _sentence_tfidf_features(sent_list)
    summarizer = GATSummarizer(
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        random_state=random_state,
    )
    top_indices = summarizer.summarise(graph, features, top_k=top_k)
    selected = sorted(top_indices)
    return " \n".join(sent_list[i] for i in selected)

"""Graph‑based text summarisation.

This module provides tools for extractive summarisation of Persian
documents using graph‑based techniques. The classic TextRank
algorithm constructs a graph of sentences and ranks them using
PageRank【140071671119304†L74-L104】.  Sentences with the highest
centrality are selected to form the summary.  We provide a
lightweight implementation of TextRank that relies only on NumPy and
scikit‑learn.

Graph neural networks can also be used for summarisation by
incorporating long‑range dependencies and discourse structure.  Recent
research has explored the use of Graph Attention Networks (GATs) and
Rhetorical Structure Theory (RST) graphs to improve summarisation
performance【721059693311575†L88-L103】.  Our implementation includes a
`GATSummarizer` stub to illustrate how such models might be
integrated in the future.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None  # type: ignore
    cosine_similarity = None  # type: ignore

from ..models.gat import GATLayer
from ..graphs.graph import Graph


def split_sentences(text: str) -> list[str]:
    """Split Persian text into sentences.

    This simple splitter looks for common sentence terminators such as
    Persian full stop ("."), question mark ("؟"), exclamation mark
    ("!") and the Arabic full stop ("٫").  It returns a list of
    sentences without the trailing punctuation.

    Parameters
    ----------
    text : str
        The input text.

    Returns
    -------
    List[str]
        A list of sentence strings.
    """
    import re
    # Define sentence delimiters
    delimiters = r"[\.!؟۔]"
    # Split and strip
    parts = re.split(delimiters, text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _build_sentence_graph(sentences: list[str]) -> tuple[Graph, list[str]]:
    """Build a sentence similarity graph for TextRank.

    Each sentence becomes a node and edges are weighted by the cosine
    similarity of TF‑IDF vectors.  Sentences are assumed to be in
    Persian; no additional preprocessing is performed.

    Parameters
    ----------
    sentences : List[str]
        List of sentence strings.

    Returns
    -------
    Graph
        A graph with sentences as nodes and weighted edges representing
        sentence similarity.
    List[str]
        The list of sentences corresponding to the graph nodes.
    """
    if TfidfVectorizer is None or cosine_similarity is None:
        raise ImportError(
            "scikit‑learn is required for TextRank summarisation"
        )
    if len(sentences) == 0:
        raise ValueError("No sentences provided")
    # Compute TF‑IDF vectors
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(sentences).astype(float).toarray()
    # Compute cosine similarity matrix
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0.0)
    # Build Graph
    graph = Graph(nodes=list(range(len(sentences))), adjacency=sim)
    return graph, sentences


def _pagerank(adjacency: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """Compute PageRank scores for a weighted graph.

    Parameters
    ----------
    adjacency : np.ndarray
        Weighted adjacency matrix (n x n). Assumed non‑negative.
    damping : float, optional
        Damping factor (teleport probability). Defaults to ``0.85``.
    max_iter : int, optional
        Maximum number of iterations. Defaults to ``100``.
    tol : float, optional
        Convergence tolerance. Iteration stops when the L1 norm of the
        score difference between successive iterations falls below this
        threshold. Defaults to ``1e-6``.

    Returns
    -------
    np.ndarray
        Array of PageRank scores summing to 1.
    """
    n = adjacency.shape[0]
    # Normalise rows to obtain transition probabilities
    row_sums = adjacency.sum(axis=1)
    # Replace zero rows with uniform distribution
    M = np.zeros_like(adjacency)
    for i in range(n):
        if row_sums[i] > 0:
            M[i] = adjacency[i] / row_sums[i]
        else:
            M[i] = np.ones(n) / n
    # Initialise PageRank scores uniformly
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * M.T @ scores
        if np.linalg.norm(new_scores - scores, 1) < tol:
            scores = new_scores
            break
        scores = new_scores
    return scores


def textrank_summarise(text: str, top_k: int = 3) -> str:
    """Generate an extractive summary using the TextRank algorithm.

    Parameters
    ----------
    text : str
        Input document in Persian.
    top_k : int, optional
        Number of sentences to include in the summary. Defaults to ``3``.

    Returns
    -------
    str
        The concatenated summary consisting of the top sentences in their
        original order.
    """
    sentences = split_sentences(text)
    graph, sent_list = _build_sentence_graph(sentences)
    scores = _pagerank(graph.adjacency)
    # Select top_k sentences based on scores
    top_indices = np.argsort(scores)[::-1][:top_k]
    # Preserve original order
    selected = sorted(top_indices)
    summary_sentences = [sent_list[i] for i in selected]
    return " \n".join(summary_sentences)


class GATSummarizer:
    """Stub for a GAT‑based summariser.

    Recent research has explored using graph neural networks – such as
    Graph Attention Networks (GATs) – to incorporate discourse and
    semantic information into summarisation models【721059693311575†L88-L103】.  This
    class illustrates how such a model could be structured.  It is not
    fully implemented; instead, it provides a scaffold for future work.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # Initialise a single GAT layer for demonstration
        self.gat = GATLayer(input_dim, hidden_dim)
        # Linear projection to output_dim
        rng = np.random.default_rng()
        limit = np.sqrt(6 / (hidden_dim + output_dim))
        self.W = rng.uniform(-limit, limit, size=(hidden_dim, output_dim))

    def summarise(self, graph: Graph, X: np.ndarray, top_k: int = 3) -> np.ndarray:
        """Generate a summary representation using GAT.

        Parameters
        ----------
        graph : Graph
            A graph whose nodes correspond to sentences or other text
            units.
        X : np.ndarray
            Input features for nodes (e.g. sentence embeddings).
        top_k : int, optional
            Number of top nodes to select based on attention scores.

        Returns
        -------
        np.ndarray
            The indices of the selected nodes. This stub simply
            computes new node embeddings with GAT and selects nodes
            with highest L2 norms.
        """
        H = self.gat.forward(graph, X)
        H_proj = H @ self.W
        norms = np.linalg.norm(H_proj, axis=1)
        top_indices = np.argsort(norms)[::-1][:top_k]
        return top_indices
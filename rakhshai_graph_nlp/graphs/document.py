"""Document similarity graph construction.

This module defines a function that builds a graph over documents by
computing their pairwise similarity. The similarity can be measured
using TF‑IDF vectors or precomputed embeddings. In the context of
graph‑based text classification, document graphs are used to propagate
information across documents via shared words or topics【136383271440271†L794-L812】.

The default implementation uses scikit‑learn's :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
to transform raw documents into vectors and then computes the cosine
similarity between every pair of documents. This yields a dense
similarity matrix that is interpreted as a weighted adjacency matrix.

Example:

    >>> from rakhshai_graph_nlp.graphs.document import build_document_graph
    >>> docs = ["این متن اول است", "این متن دوم است"]
    >>> graph = build_document_graph(docs)
    >>> graph.adjacency
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None  # type: ignore[assignment]
    cosine_similarity = None  # type: ignore[assignment]

from .graph import Graph


def build_document_graph(
    documents: list[str],
    min_similarity: float = 0.0,
    vectorizer: object | None = None,
    precomputed_embeddings: np.ndarray | None = None,
) -> Graph:
    """Build a document similarity graph.

    Parameters
    ----------
    documents : Sequence[str]
        A collection of raw documents (strings). If
        ``precomputed_embeddings`` is provided, this argument may be
        ignored.
    min_similarity : float, optional
        Threshold below which similarities are set to zero. Useful to
        sparsify the graph by dropping weak similarities. Defaults to
        ``0.0`` (keep all similarities).
    vectorizer : object, optional
        A vectoriser instance following the scikit‑learn API, with a
        ``fit_transform`` method. If ``None``, a default
        :class:`~sklearn.feature_extraction.text.TfidfVectorizer` is
        used. This argument is ignored if ``precomputed_embeddings`` is
        provided.
    precomputed_embeddings : np.ndarray, optional
        An array of shape ``(n_documents, dim)`` containing vector
        representations of each document. If provided, TF‑IDF is not
        computed and the pairwise cosine similarity is calculated on
        these embeddings instead.

    Returns
    -------
    Graph
        A graph whose nodes correspond to the input documents and whose
        weighted edges represent the cosine similarity between the
        documents.
    """
    if precomputed_embeddings is not None:
        X = precomputed_embeddings
    else:
        if TfidfVectorizer is None or cosine_similarity is None:
            raise ImportError(
                "scikit‑learn is required for TF‑IDF graph construction"
            )
        if vectorizer is None:
            vectorizer = TfidfVectorizer(stop_words=None)
        X = vectorizer.fit_transform(documents).astype(float).toarray()
    sim = cosine_similarity(X)
    # Zero out self similarities
    np.fill_diagonal(sim, 0.0)
    # Threshold similarities
    if min_similarity > 0.0:
        sim[sim < min_similarity] = 0.0
    nodes = [f"doc_{i}" for i in range(len(X))]
    return Graph(nodes=nodes, adjacency=sim)
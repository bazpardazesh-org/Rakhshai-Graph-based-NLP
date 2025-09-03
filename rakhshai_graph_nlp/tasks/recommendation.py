"""Document recommendation using graph similarity.

This module implements a simple content‑based recommender for Persian
documents.  Given a query document and a corpus of existing
documents, it constructs a document graph (using TF‑IDF similarity) and
returns the most similar documents.  This approach can be extended to
use graph neural networks or more sophisticated embeddings for
recommendation.  The focus here is on simplicity and ease of use.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..graphs.document import build_document_graph


def recommend_similar(
    query: str,
    documents: Sequence[str],
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """Recommend the most similar documents to a query.

    Parameters
    ----------
    query : str
        The query document.
    documents : Sequence[str]
        A list of candidate documents to search over.
    top_k : int, optional
        Number of recommendations to return. Defaults to ``5``.

    Returns
    -------
    List[Tuple[int, float]]
        A list of tuples ``(index, score)`` indicating the index of the
        recommended document in the input list and its similarity score.
    """
    # Combine query with documents
    all_docs = [query] + list(documents)
    graph = build_document_graph(all_docs)
    # Similarity between query node (index 0) and others is encoded
    sim_row = graph.adjacency[0]
    # Exclude the query itself
    scores = sim_row[1:]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]
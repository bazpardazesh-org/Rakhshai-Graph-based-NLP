"""Co‑occurrence graph construction.

This module contains a function for building a simple word co‑occurrence
graph from a list of tokenised documents. Each unique word in the
collection becomes a node; an undirected edge connects two words if
they appear within a sliding window of fixed size. The weight of the
edge is the number of co‑occurrences across the corpus. If there is
only one document, the resulting graph is a document‑level graph; if
multiple documents are provided, a single corpus‑level graph is
constructed. Similar strategies are used in the TextGCN model, where
word–word edges are weighted by pointwise mutual information (PMI) and
word–document edges are weighted by TF‑IDF【136383271440271†L794-L812】.

Example:

    >>> from rakhshai_graph_nlp.features.tokenizer import tokenize
    >>> from rakhshai_graph_nlp.graphs.co_occurrence import build_cooccurrence_graph
    >>> docs = ["این یک تست است", "تست بعدی"]
    >>> tokenised = [tokenize(d) for d in docs]
    >>> graph = build_cooccurrence_graph(tokenised, window_size=2)
    >>> print(len(graph.nodes), graph.adjacency.sum())
"""

from __future__ import annotations

import numpy as np

from .graph import Graph


def build_cooccurrence_graph(
    token_lists: list[list[str]],
    window_size: int = 2,
    min_count: int = 1,
) -> Graph:
    """Build a word co‑occurrence graph.

    Parameters
    ----------
    token_lists : Sequence[Sequence[str]]
        A sequence of tokenised documents. Each element of
        ``token_lists`` should itself be a sequence of tokens (strings).
    window_size : int, optional
        The size of the sliding window. Two words are considered a
        co‑occurrence if they appear within ``window_size`` positions
        of one another. Defaults to ``2``.
    min_count : int, optional
        Minimum frequency threshold for including a word in the graph.
        Tokens that appear fewer than ``min_count`` times across all
        documents are filtered out. Defaults to ``1`` (include all
        tokens).

    Returns
    -------
    Graph
        A graph with nodes corresponding to words and weighted edges
        representing co‑occurrence counts.
    """
    # Count term frequencies across the corpus
    freq: dict[str, int] = {}
    for tokens in token_lists:
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
    # Build vocabulary of words meeting the frequency threshold
    vocab = {word for word, count in freq.items() if count >= min_count}
    if not vocab:
        raise ValueError("No tokens meet the minimum count threshold")
    # Map words to indices
    word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    n = len(word2idx)
    adjacency = np.zeros((n, n), dtype=float)
    # Sliding window co‑occurrence counting
    for tokens in token_lists:
        length = len(tokens)
        for i, center in enumerate(tokens):
            if center not in word2idx:
                continue
            center_idx = word2idx[center]
            # Consider tokens within the window to the left and right
            left = max(i - window_size, 0)
            right = min(i + window_size + 1, length)
            for j in range(left, right):
                if j == i:
                    continue
                neighbour = tokens[j]
                if neighbour not in word2idx:
                    continue
                neigh_idx = word2idx[neighbour]
                adjacency[center_idx, neigh_idx] += 1
                adjacency[neigh_idx, center_idx] += 1  # Ensure symmetry
    nodes = list(word2idx.keys())
    return Graph(nodes=nodes, adjacency=adjacency)
"""Build a heterogeneous text graph for document classification.

This module implements construction of a graph inspired by the TextGCN
model.  It returns a graph whose nodes include both the words in the
corpus and the documents themselves.  Edges between word nodes are
weighted by pointwise mutual information (PMI) computed on a sliding
window over the corpus, and edges between word and document nodes are
weighted by the term‑frequency inverse document frequency (TF‑IDF) of
the word in that document.  Document–document edges are not
explicitly created (all zeros).

This graph can be used with graph neural networks to perform text
classification where the documents are nodes in the graph and their
labels can be predicted by propagating information through the word
graph.  The construction here follows the high‑level description of
TextGCN: a corpus‑level graph with word–word edges weighted by PMI
and word–document edges weighted by TF‑IDF【341556962532327†L779-L785】.  PMI
captures global co‑occurrence statistics and TF‑IDF emphasises
important words in each document.  Words appearing fewer than
``min_count`` times are filtered out.

Example
-------
>>> from rakhshai_graph_nlp.features.tokenizer import tokenize
>>> from rakhshai_graph_nlp.graphs.text_graph import build_text_graph
>>> docs = ["خبر سیاسی", "ورزش", "هنر"]
>>> tokenised = [tokenize(d) for d in docs]
>>> graph = build_text_graph(tokenised)
>>> # The last len(docs) nodes correspond to documents
>>> doc_nodes = graph.nodes[-len(docs):]
>>> print(doc_nodes)

Note
----
This implementation is designed for small corpora in research or
education settings.  It uses dense NumPy arrays and may not scale
well to very large vocabularies or document collections.  For large
datasets, more efficient sparse representations and streaming PMI
computation should be used.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np

from .graph import Graph

def _compute_pmi(
    token_lists: Sequence[Sequence[str]],
    vocab: Sequence[str],
    window_size: int = 20,
) -> np.ndarray:
    """Compute a PMI matrix for the given vocabulary.

    PMI is computed based on co‑occurrence counts within a sliding
    window over the corpus.  Windows are taken separately for each
    document and concatenated to estimate co‑occurrence probabilities.

    Parameters
    ----------
    token_lists : Sequence[Sequence[str]]
        List of tokenised documents.
    vocab : Sequence[str]
        Ordered list of vocabulary terms.
    window_size : int, optional
        Size of the context window.  Defaults to ``20``.

    Returns
    -------
    np.ndarray
        A symmetric matrix of PMI values for each pair of vocabulary
        words.
    """
    vocab_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    # Count word frequencies and co‑occurrences
    freq = np.zeros(V, dtype=float)
    co_counts = np.zeros((V, V), dtype=float)
    total_windows = 0
    for tokens in token_lists:
        n = len(tokens)
        # Count word frequency
        for t in tokens:
            idx = vocab_index.get(t)
            if idx is not None:
                freq[idx] += 1
        # Sliding windows
        for i in range(n):
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            center = tokens[i]
            c_idx = vocab_index.get(center)
            if c_idx is None:
                continue
            # context excluding centre
            for j in range(left, right):
                if j == i:
                    continue
                neighbour = tokens[j]
                n_idx = vocab_index.get(neighbour)
                if n_idx is None:
                    continue
                co_counts[c_idx, n_idx] += 1
                # symmetric count
        total_windows += n
    # Compute probabilities
    p_i = freq / freq.sum() if freq.sum() > 0 else np.zeros_like(freq)
    p_ij = co_counts / co_counts.sum() if co_counts.sum() > 0 else np.zeros_like(co_counts)
    # Compute PMI
    pmi = np.zeros((V, V), dtype=float)
    for i in range(V):
        for j in range(V):
            if i == j or p_ij[i, j] == 0 or p_i[i] == 0 or p_i[j] == 0:
                continue
            val = math.log(p_ij[i, j] / (p_i[i] * p_i[j]))
            if val > 0:
                pmi[i, j] = val
    # Symmetrise
    pmi = (pmi + pmi.T) / 2.0
    return pmi


def _compute_tfidf(
    token_lists: Sequence[Sequence[str]],
    vocab: Sequence[str],
    smooth_idf: bool = True,
) -> np.ndarray:
    """Compute TF‑IDF weights for each (word, document) pair.

    Parameters
    ----------
    token_lists : Sequence[Sequence[str]]
        List of tokenised documents.
    vocab : Sequence[str]
        Vocabulary list matching the order of the rows in the output.
    smooth_idf : bool, optional
        Whether to add 1 to document frequencies to avoid division by
        zero.  Defaults to ``True``.

    Returns
    -------
    np.ndarray
        An array of shape ``(len(vocab), n_docs)`` containing TF‑IDF
        scores.  Entry ``(i, d)`` is the TF‑IDF of word ``vocab[i]`` in
        document ``token_lists[d]``.
    """
    vocab_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    n_docs = len(token_lists)
    tf = np.zeros((V, n_docs), dtype=float)
    df = np.zeros(V, dtype=float)
    # Compute term frequency and document frequency
    for d, tokens in enumerate(token_lists):
        counts = Counter(t for t in tokens if t in vocab_index)
        for t, c in counts.items():
            i = vocab_index[t]
            tf[i, d] = c
            df[i] += 1
    if smooth_idf:
        idf = np.log((n_docs + 1) / (df + 1)) + 1
    else:
        # avoid divide by zero
        idf = np.log(n_docs / (df + (df == 0))) + 1
    tfidf = tf * idf[:, None]
    return tfidf


def build_text_graph(
    token_lists: Sequence[Sequence[str]],
    window_size: int = 20,
    min_count: int = 1,
    smooth_idf: bool = True,
) -> Graph:
    """Construct a heterogeneous text graph with word and document nodes.

    Parameters
    ----------
    token_lists : Sequence[Sequence[str]]
        Tokenised documents.  Each element is a list of tokens.
    window_size : int, optional
        Sliding window size for PMI computation.  Defaults to ``20``.
    min_count : int, optional
        Minimum frequency threshold for words.  Words appearing fewer
        than ``min_count`` times across the corpus are discarded.
        Defaults to ``1``.
    smooth_idf : bool, optional
        Whether to smooth the IDF calculation by adding 1 to document
        frequencies.  Defaults to ``True``.

    Returns
    -------
    Graph
        A graph with ``len(vocab) + len(token_lists)`` nodes.  The
        first ``len(vocab)`` nodes are words, the last ``len(token_lists)``
        nodes are documents.  The adjacency matrix encodes PMI between
        words and TF‑IDF between words and documents.
    """
    if not token_lists:
        raise ValueError("token_lists must not be empty")
    # Flatten and count frequencies
    freq: dict[str, int] = defaultdict(int)
    for tokens in token_lists:
        for t in tokens:
            freq[t] += 1
    # Build vocabulary
    vocab = [w for w, c in freq.items() if c >= min_count]
    if not vocab:
        raise ValueError("No words meet the frequency threshold")
    V = len(vocab)
    D = len(token_lists)
    # Compute PMI for word–word edges
    pmi_matrix = _compute_pmi(token_lists, vocab, window_size=window_size)
    # Compute TF‑IDF for word–document edges
    tfidf = _compute_tfidf(token_lists, vocab, smooth_idf=smooth_idf)
    # Build adjacency matrix
    N = V + D
    adjacency = np.zeros((N, N), dtype=float)
    # Word–word edges
    adjacency[:V, :V] = pmi_matrix
    # Word–document edges
    adjacency[:V, V:] = tfidf
    adjacency[V:, :V] = tfidf.T
    # Document–document edges remain zero
    # Create node labels
    nodes: list[str] = []
    node_types: list[str] = []
    # Word nodes
    nodes.extend(vocab)
    node_types.extend(["word"] * V)
    # Document nodes
    for i in range(D):
        nodes.append(f"doc_{i}")
        node_types.append("doc")
    return Graph(nodes=nodes, adjacency=adjacency, node_types=node_types)

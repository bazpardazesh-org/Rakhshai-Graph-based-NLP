"""Semantic graph construction.

This module provides a minimal implementation for building semantic
graphs using externally provided lexical relations.  In the absence of a
comprehensive Persian WordNet, users can supply a mapping of words to
their semantic neighbours (for example synonyms).  Edges are added
between words appearing in the provided relation mapping.  If no
relations are given, an empty graph is returned.

Example:

    >>> from rakhshai_graph_nlp.graphs.semantic import build_semantic_graph
    >>> relations = {"گربه": ["سگ"]}
    >>> graph = build_semantic_graph(["گربه", "سگ"], relations=relations)
    >>> graph.adjacency.sum()
    2.0
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Optional

import numpy as np

from .graph import Graph


def build_semantic_graph(
    words: Sequence[str],
    relations: Optional[Mapping[str, Iterable[str]]] = None,
) -> Graph:
    """Construct a semantic graph from a list of words.

    Parameters
    ----------
    words : Sequence[str]
        A sequence of words for which to build a semantic graph. Each
        word becomes a node in the graph.
    relations : Mapping[str, Iterable[str]], optional
        Mapping of a word to a collection of semantically related words
        (e.g. synonyms). Only relations where both words appear in the
        ``words`` sequence are added to the graph. If ``None`` (the
        default), an empty graph with no edges is returned.

    Returns
    -------
    Graph
        A graph with the given words as nodes. Edges represent semantic
        relations supplied via ``relations``.
    """
    nodes = list(words)
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=float)
    if relations:
        index = {w: i for i, w in enumerate(nodes)}
        for src, targets in relations.items():
            if src not in index:
                continue
            src_idx = index[src]
            for tgt in targets:
                if tgt not in index:
                    continue
                tgt_idx = index[tgt]
                adjacency[src_idx, tgt_idx] += 1
                adjacency[tgt_idx, src_idx] += 1
    return Graph(nodes=nodes, adjacency=adjacency)
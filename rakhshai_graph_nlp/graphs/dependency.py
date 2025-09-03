"""Dependency graph construction.

This module exposes a function for building a dependency graph from
tokenised sentences using the Stanza library. Each token becomes a
node, and directed edges connect tokens according to their syntactic
dependency relations. Node attributes include the part‑of‑speech tag and
lemma where available.  Stanza is a multilingual neural NLP toolkit
supporting tokenisation, POS tagging and dependency parsing for many
languages, including Persian【534291576090157†L11-L33】.

Stanza must be installed separately. If it is not available, this
module will raise an ``ImportError``. To install Stanza and download
Persian models, run::

    pip install stanza
    python -m stanza.download fa

After installation, you can build a dependency graph as follows::

    >>> from rakhshai_graph_nlp.features.tokenizer import tokenize
    >>> from rakhshai_graph_nlp.graphs.dependency import build_dependency_graph
    >>> text = "او می‌نویسد و من می‌خوانم."
    >>> graph = build_dependency_graph([text])
    >>> print(graph.nodes)

The returned :class:`~rakhshai_graph_nlp.graphs.graph.Graph` is directed
so the adjacency matrix need not be symmetric.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

try:
    import stanza
except ImportError:
    stanza = None  # type: ignore[assignment]

from .graph import Graph


def build_dependency_graph(sentences: Sequence[str]) -> Graph:
    """Build a dependency graph using Stanza.

    Parameters
    ----------
    sentences : Sequence[str]
        A sequence of sentences in Persian. Each sentence will be
        processed independently by Stanza's pipeline.

    Returns
    -------
    Graph
        A graph where nodes are tokens and edges connect tokens that
        participate in a dependency relation. If Stanza is not
        installed, an exception is raised.
    """
    if stanza is None:
        raise ImportError(
            "Stanza is required for dependency graph construction. "
            "Please install stanza and download the Persian model."
        )
    # Load Persian pipeline if not already initialised. Use caching to
    # avoid reinitialising across calls.
    if not hasattr(build_dependency_graph, "_nlp"):
        build_dependency_graph._nlp = stanza.Pipeline(
            lang="fa", processors="tokenize,pos,lemma,depparse", use_gpu=False
        )  # type: ignore[attr-defined]
    nlp = build_dependency_graph._nlp  # type: ignore[attr-defined]
    nodes: list[str] = []
    edges: list[tuple[int, int]] = []
    # Process each sentence and accumulate token indices and edges
    for sent in sentences:
        doc = nlp(sent)
        for sent_idx, sentence in enumerate(doc.sentences):
            token_offset = len(nodes)
            # Add tokens to node list
            for word in sentence.words:
                nodes.append(word.text)
            # Add dependency edges (parent and child)
            for word in sentence.words:
                if word.head == 0:
                    continue  # skip root
                # Stanza uses 1‑based indexing for heads within a sentence
                parent_idx = token_offset + word.head - 1
                child_idx = token_offset + word.id - 1
                edges.append((parent_idx, child_idx))
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=float)
    for u, v in edges:
        adjacency[u, v] += 1
    return Graph(nodes=nodes, adjacency=adjacency, directed=True)
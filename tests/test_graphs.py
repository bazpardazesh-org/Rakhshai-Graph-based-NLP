import numpy as np
import pytest

from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.graphs.co_occurrence import build_cooccurrence_graph
from rakhshai_graph_nlp.graphs.dependency import build_dependency_graph
from rakhshai_graph_nlp.graphs.document import build_document_graph
from rakhshai_graph_nlp.graphs.semantic import build_semantic_graph
from rakhshai_graph_nlp.graphs.text_graph import build_text_graph


def _simple_graph():
    nodes = ["a", "b", "c"]
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    return Graph(nodes=nodes, adjacency=adjacency)


def test_graph_basic_operations():
    g = _simple_graph()
    g.add_self_loops()
    assert np.all(np.diag(g.adjacency) == 1)
    deg = g.degree_matrix()
    assert np.allclose(deg, np.diag([2, 3, 2]))
    norm = g.normalized_adjacency()
    assert norm.shape == g.adjacency.shape
    edges = g.to_edge_list()
    assert ("a", "b", 1.0) in edges and ("b", "c", 1.0) in edges
    g2 = g.copy()
    g2.adjacency[0, 1] = 5
    assert g.adjacency[0, 1] == 1


def test_directed_graph_edge_list():
    nodes = ["a", "b"]
    adjacency = np.array([[0, 1], [0, 0]], dtype=float)
    g = Graph(nodes=nodes, adjacency=adjacency, directed=True)
    edges = g.to_edge_list()
    assert ("a", "b", 1.0) in edges and ("b", "a", 1.0) not in edges
    norm = g.normalized_adjacency()
    assert np.allclose(norm, adjacency)


def test_build_cooccurrence_graph():
    tokens = [["a", "b", "a"], ["b", "c"]]
    g = build_cooccurrence_graph(tokens, window_size=1)
    assert set(g.nodes) == {"a", "b", "c"}
    assert g.adjacency.shape == (3, 3)
    assert np.array_equal(g.adjacency, g.adjacency.T)


def test_cooccurrence_min_count():
    with pytest.raises(ValueError):
        build_cooccurrence_graph([["a"]], min_count=2)


def test_dependency_requires_stanza(monkeypatch):
    monkeypatch.setattr("rakhshai_graph_nlp.graphs.dependency.stanza", None)
    with pytest.raises(ImportError):
        build_dependency_graph(["متن"])


def test_build_document_graph():
    docs = ["این یک متن است", "متن دیگر"]
    g = build_document_graph(docs)
    assert len(g.nodes) == 2
    assert g.adjacency.shape == (2, 2)
    assert np.allclose(g.adjacency, g.adjacency.T)


def test_build_semantic_graph():
    words = ["گربه", "سگ"]
    g = build_semantic_graph(words)
    assert g.adjacency.sum() == 0
    relations = {"گربه": ["سگ"]}
    g_rel = build_semantic_graph(words, relations=relations)
    assert g_rel.adjacency[0, 1] == 1 and g_rel.adjacency[1, 0] == 1


def test_build_text_graph():
    token_lists = [["a", "b"], ["b", "c"]]
    g = build_text_graph(token_lists, window_size=1)
    v = 3
    d = 2
    assert len(g.nodes) == v + d
    assert g.adjacency.shape == (v + d, v + d)
    assert g.node_types.count("word") == v
    assert g.node_types.count("doc") == d

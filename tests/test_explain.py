import numpy as np

from rakhshai_graph_nlp.explain.explainer import node_importance
from rakhshai_graph_nlp.graphs.graph import Graph


def test_node_importance():
    g = Graph(nodes=[0, 1], adjacency=np.zeros((2, 2)))
    imp = node_importance(None, g, 0)
    assert imp.shape == (2,) and imp[0] == 1.0

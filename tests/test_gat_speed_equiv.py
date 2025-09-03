import numpy as np

from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.models.gat import GATClassifier


def test_gat_runs():
    g = Graph(nodes=[0, 1], adjacency=np.ones((2, 2)))
    X = np.eye(2)
    model = GATClassifier(input_dim=2, hidden_dim=2, num_classes=2)
    model.fit(g, X, np.array([0, 1]), num_epochs=1)
    preds = model.predict(g, X)
    assert preds.shape == (2,)

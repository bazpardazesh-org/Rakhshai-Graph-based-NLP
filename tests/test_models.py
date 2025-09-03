import numpy as np

from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.models.gat import GATLayer
from rakhshai_graph_nlp.models.gcn import GCNClassifier, GCNLayer
from rakhshai_graph_nlp.models.graphsage import GraphSAGE

def test_gcnlayer_forward_shape():
    layer = GCNLayer(2, 3)
    X = np.ones((2, 2))
    A = np.eye(2)
    out = layer.forward(X, A)
    assert out.shape == (2, 3)

def test_gcnclassifier_fit_and_predict():
    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    X = np.eye(2)
    labels = np.array([0, 1])
    clf = GCNClassifier(input_dim=2, hidden_dim=2, num_classes=2)
    losses = clf.fit(g, X, labels, num_epochs=5, learning_rate=0.05)
    assert len(losses) == 5
    preds = clf.predict(g, X)
    assert preds.shape == (2,)

def test_gatlayer_forward_shape():
    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    X = np.random.rand(2, 4)
    layer = GATLayer(4, 3)
    out = layer.forward(g, X)
    assert out.shape == (2, 3)

def test_graphsage_forward_shape():
    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    X = np.random.rand(2, 4)
    sage = GraphSAGE(input_dim=4, hidden_dims=[3, 2])
    out = sage.forward(g, X)
    assert out.shape == (2, 2)

import numpy as np

from rakhshai_graph_nlp.features.pyg_data import graph_to_data
from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.tasks.classification import train_node_classifier

def test_gat_runs():
    g = Graph(nodes=[0, 1], adjacency=np.ones((2, 2)))
    X = np.eye(2)
    labels = np.array([0, 1])
    model, _ = train_node_classifier(g, labels, X=X, model_type="gat", num_epochs=1)
    data = graph_to_data(g, features=X, labels=labels)
    preds = model.predict(data)
    assert preds.shape == (2,)

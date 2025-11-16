import numpy as np
import torch
from torch_geometric.data import Data

from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.models.gat import GATClassifier, GATLayer
from rakhshai_graph_nlp.models.gcn import GCNClassifier
from rakhshai_graph_nlp.models.graphsage import GraphSAGEClassifier


def _toy_data() -> Data:
    x = torch.eye(3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 1], [1, 0, 2, 1, 0, 2]])
    y = torch.tensor([0, 1, 0])
    return Data(x=x, edge_index=edge_index, y=y)


def test_gcnclassifier_forward_shape():
    data = _toy_data()
    model = GCNClassifier(input_dim=3, hidden_dim=4, num_classes=2)
    out = model(data)
    assert out.shape == (3, 2)


def test_graphsage_forward_shape():
    data = _toy_data()
    model = GraphSAGEClassifier(input_dim=3, hidden_dim=4, num_classes=2)
    out = model(data)
    assert out.shape == (3, 2)


def test_gatclassifier_forward_shape():
    data = _toy_data()
    model = GATClassifier(input_dim=3, hidden_dim=2, num_classes=2, heads=2)
    out = model(data)
    assert out.shape == (3, 2)


def test_legacy_gatlayer():
    graph = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    X = np.random.rand(2, 4)
    layer = GATLayer(4, 3)
    out = layer.forward(graph, X)
    assert out.shape == (2, 3)

"""Neural models for graph representation learning.

This subpackage now wraps PyTorch Geometric primitives such as
``GCNConv``, ``SAGEConv`` and ``GATConv`` so that experiments can run on
either CPUs or GPUs with minimal changes to the public API.
"""

from .gat import GATClassifier, GATLayer
from .gcn import GCNClassifier
from .graphsage import GraphSAGE, GraphSAGEClassifier, GraphSAGELayer

__all__ = [
    "GCNClassifier",
    "GATClassifier",
    "GATLayer",
    "GraphSAGE",
    "GraphSAGEClassifier",
    "GraphSAGELayer",
]

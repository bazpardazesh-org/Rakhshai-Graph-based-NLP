"""Top level package for the Rakhshai Graph-based NLP library."""

from importlib import import_module

from .features.tokenize import split_sentences, tokenize
from .graphs import sparse as graphs_sparse
from .metrics import accuracy, confusion_matrix, macro_f1

# Re-export models package
from . import models

__all__ = [
    "accuracy",
    "confusion_matrix",
    "macro_f1",
    "tokenize",
    "split_sentences",
    "graphs",
    "models",
]


def __getattr__(name: str):
    if name == "graphs":
        return import_module("rakhshai_graph_nlp.graphs")
    raise AttributeError(name)


# expose sparse helpers
graphs = import_module("rakhshai_graph_nlp.graphs")
graphs.sparse = graphs_sparse

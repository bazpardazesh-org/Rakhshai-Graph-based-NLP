"""Top-level package for the stable Rakhshai Graph-based NLP API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from . import api as _api

__version__ = "2.2.0"
API_STATUS = _api.API_STATUS
API_VERSION = _api.API_VERSION
STABLE_API_VERSION = _api.STABLE_API_VERSION
__api_version__ = _api.__api_version__
stable_api = _api.stable_api

_SUBPACKAGES = {
    "article_llm": "rakhshai_graph_nlp.article_llm",
    "api": "rakhshai_graph_nlp.api",
    "features": "rakhshai_graph_nlp.features",
    "graphs": "rakhshai_graph_nlp.graphs",
    "lm": "rakhshai_graph_nlp.lm",
    "llm": "rakhshai_graph_nlp.llm",
    "models": "rakhshai_graph_nlp.models",
    "tasks": "rakhshai_graph_nlp.tasks",
}

__all__ = [
    "__version__",
    "API_STATUS",
    "API_VERSION",
    "STABLE_API_VERSION",
    "__api_version__",
    "stable_api",
    *_SUBPACKAGES,
    *_api.stable_api(),
]


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        module = import_module(_SUBPACKAGES[name])
        if name == "graphs":
            module.sparse = import_module("rakhshai_graph_nlp.graphs.sparse")
        globals()[name] = module
        return module
    if name in _api.stable_api():
        value = getattr(_api, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

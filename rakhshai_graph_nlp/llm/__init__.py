"""High-level LLM workflows built on Rakhshai engines."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SUBPACKAGES = {
    "article": "rakhshai_graph_nlp.llm.article",
}

__all__ = [*_SUBPACKAGES]


def __getattr__(name: str) -> Any:
    if name not in _SUBPACKAGES:
        raise AttributeError(name)
    module = import_module(_SUBPACKAGES[name])
    globals()[name] = module
    return module

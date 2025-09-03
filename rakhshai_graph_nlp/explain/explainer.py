"""Explainability stubs."""

from __future__ import annotations

import numpy as np


def node_importance(model, graph, node_id: int) -> np.ndarray:
    """Return dummy importance scores for ``node_id``."""
    n = len(graph.nodes)  # type: ignore[arg-type]
    imp = np.zeros(n)
    imp[node_id] = 1.0
    return imp

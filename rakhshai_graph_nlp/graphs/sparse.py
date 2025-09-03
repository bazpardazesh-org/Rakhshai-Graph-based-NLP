"""Utility helpers for working with sparse adjacency matrices."""

from __future__ import annotations

import numpy as np

try:
    import scipy.sparse as sparse
except Exception:  # pragma: no cover - optional dependency
    sparse = None  # type: ignore[assignment]

__all__ = ["to_undirected_coo", "add_self_loops", "row_normalize_csr"]


def to_undirected_coo(A: sparse.spmatrix) -> sparse.coo_matrix:
    """Make a COO matrix symmetric."""
    if sparse is None:
        raise ImportError("scipy is required for sparse graph utilities")
    if not sparse.isspmatrix_coo(A):
        A = A.tocoo()
    upper = sparse.triu(A)
    result = upper + upper.T - sparse.diags(A.diagonal())
    return result.tocoo()


def add_self_loops(A: sparse.spmatrix, v: float = 1.0) -> sparse.spmatrix:
    """Add self-loop value ``v`` to the diagonal of ``A``."""
    if sparse is None:
        raise ImportError("scipy is required for sparse graph utilities")
    if not sparse.issparse(A):
        raise TypeError("A must be a scipy sparse matrix")
    return (A + sparse.identity(A.shape[0]) * v).asformat(A.getformat())


def row_normalize_csr(A: sparse.spmatrix) -> sparse.csr_matrix:
    """Row-normalise a CSR matrix."""
    if sparse is None:
        raise ImportError("scipy is required for sparse graph utilities")
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()
    row_sum = np.array(A.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1
    inv = 1.0 / row_sum
    D = sparse.diags(inv)
    return D.dot(A)

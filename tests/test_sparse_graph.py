import numpy as np
import pytest
from rakhshai_graph_nlp.graphs.sparse import (
    add_self_loops,
    row_normalize_csr,
    to_undirected_coo,
)

sparse = pytest.importorskip("scipy.sparse")


def test_sparse_helpers():
    A = sparse.coo_matrix(([1], ([0], [1])), shape=(2, 2))
    B = to_undirected_coo(A)
    arr = B.toarray()
    assert arr[0, 1] == 1 and arr[1, 0] == 1
    C = add_self_loops(B, v=0.5)
    assert C.diagonal().sum() == 1.0
    D = row_normalize_csr(C)
    rowsum = np.array(D.sum(axis=1)).flatten()
    assert np.allclose(rowsum, 1.0)

import pytest

from rakhshai_graph_nlp.features.text_graph import cooccurrence_matrix

sparse = pytest.importorskip("scipy.sparse")


def test_cooccurrence_symmetry():
    corpus = [["a", "b", "a"]]
    A, vocab = cooccurrence_matrix(corpus, window_size=1)
    assert vocab == ["a", "b"]
    assert A.shape == (2, 2)
    arr = A.toarray()
    assert arr[0, 1] == 2 and arr[1, 0] == 2
    assert sparse.isspmatrix_coo(A)

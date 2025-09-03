"""Utilities for building sparse text graphs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

try:
    import scipy.sparse as sparse
except Exception:  # pragma: no cover - optional dependency
    sparse = None  # type: ignore[assignment]

__all__ = ["cooccurrence_matrix"]


def cooccurrence_matrix(
    corpus: Iterable[list[str]], window_size: int = 2
) -> tuple[sparse.coo_matrix, list[str]]:
    """Build a sliding-window co-occurrence matrix.

    Parameters
    ----------
    corpus:
        Iterable of token lists.
    window_size:
        Size of sliding context window.

    Returns
    -------
    (coo_matrix, vocab)
        Sparse symmetric co-occurrence matrix and vocabulary list.
    """

    vocab = {}
    counts = defaultdict(float)
    for tokens in corpus:
        for i, w in enumerate(tokens):
            if w not in vocab:
                vocab[w] = len(vocab)
            wi = vocab[w]
            j_end = min(len(tokens), i + window_size + 1)
            for j in range(i + 1, j_end):
                v = tokens[j]
                if v not in vocab:
                    vocab[v] = len(vocab)
                wj = vocab[v]
                counts[(wi, wj)] += 1
    if sparse is None:
        raise ImportError("scipy is required for sparse co-occurrence matrices")

    if not counts:
        return sparse.coo_matrix((0, 0)), []
    rows, cols, data = zip(*[(i, j, v) for (i, j), v in counts.items()])
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)))
    mat = mat + mat.T
    return mat.tocoo(), [w for w, _ in sorted(vocab.items(), key=lambda x: x[1])]

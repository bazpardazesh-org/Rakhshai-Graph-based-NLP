"""Utility functions for evaluating model predictions.

This module provides basic classification metrics with optional
scikit-learn acceleration if the library is installed. All functions
operate on numpy arrays or array-like sequences and return Python
floats or numpy arrays. The implementations avoid external
dependencies when possible so that the package can function in minimal
installations.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import (
        confusion_matrix as sk_confusion_matrix,
        f1_score,
    )

    _HAVE_SKLEARN = True
except Exception:  # pragma: no cover - sklearn not available
    _HAVE_SKLEARN = False


__all__ = ["accuracy", "macro_f1", "confusion_matrix"]


def _to_numpy(arr: Sequence[int] | np.ndarray) -> np.ndarray:
    """Convert a sequence or array into a 1-D numpy array of integers."""

    return np.asarray(list(arr), dtype=int)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute the fraction of correctly predicted labels.

    Parameters
    ----------
    y_true: Sequence[int]
        Ground truth labels.
    y_pred: Sequence[int]
        Predicted labels.

    Returns
    -------
    float
        The accuracy score ``(y_true == y_pred).mean()``.
    """

    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(yt == yp))


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """Compute the confusion matrix.

    If scikit-learn is installed this function defers to
    :func:`sklearn.metrics.confusion_matrix`. Otherwise a minimal
    numpy-based implementation is used.
    """

    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if _HAVE_SKLEARN:  # pragma: no branch - trivial guard
        return sk_confusion_matrix(yt, yp)

    labels = np.unique(np.concatenate([yt, yp]))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(yt, yp):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Compute the unweighted mean F1 score across labels.

    Parameters
    ----------
    y_true: Sequence[int]
        Ground truth labels.
    y_pred: Sequence[int]
        Predicted labels.

    Returns
    -------
    float
        Macro-averaged F1 score.
    """

    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if _HAVE_SKLEARN:  # pragma: no branch - trivial guard
        return float(f1_score(yt, yp, average="macro"))

    cm = confusion_matrix(yt, yp)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    return float(np.mean(f1))

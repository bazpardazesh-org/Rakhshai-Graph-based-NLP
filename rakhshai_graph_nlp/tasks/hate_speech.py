"""Hate speech and fake news detection (stub).

Detecting hate speech or misinformation in social media requires
datasets and models that capture the nuances of Persian language and
cultural context.  This module provides a placeholder implementation
that uses a simple keyword list to flag potentially hateful messages.
For serious applications, one should collect a labelled dataset and
train a graph neural network that combines user, message and topic
graphs【136383271440271†L720-L726】.
"""

from __future__ import annotations

from collections.abc import Iterable


def contains_hate_speech(text: str, hate_terms: Iterable[str]) -> bool:
    """Check whether a text contains any hateful terms.

    Parameters
    ----------
    text : str
        Input text (e.g. a tweet or post) in Persian.
    hate_terms : Iterable[str]
        A collection of terms considered hateful.  If any of these
        terms appears as a substring of ``text``, the function returns
        ``True``.

    Returns
    -------
    bool
        ``True`` if the text contains a hateful term, ``False`` otherwise.
    """
    for term in hate_terms:
        if term in text:
            return True
    return False
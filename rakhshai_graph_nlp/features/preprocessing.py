"""Text normalisation utilities.

Persian text contains several orthographic variants of letters and
punctuation marks. In order to obtain consistent graphs and improve
tokenisation quality, it is often useful to normalise the text
before further processing.  This module implements a few simple
normalisation functions, such as converting Arabic forms of "ye" and
"kaf" to their Persian equivalents and removing diacritics.
"""

from __future__ import annotations

import re


ARABIC_YE = "ي"
PERSIAN_YE = "ی"
ARABIC_KE = "ك"
PERSIAN_KE = "ک"

DIACRITICS_PATTERN = re.compile(
    r"[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655]"
)


def normalise_characters(text: str) -> str:
    """Normalise common Arabic characters to Persian equivalents.

    Parameters
    ----------
    text : str
        Input text in Persian.

    Returns
    -------
    str
        Normalised text.
    """
    text = text.replace(ARABIC_YE, PERSIAN_YE)
    text = text.replace(ARABIC_KE, PERSIAN_KE)
    # Normalise half spaces: replace various zero‑width non‑joiners with a single space
    text = text.replace("\u200c", " ")
    return text


def remove_diacritics(text: str) -> str:
    """Remove Arabic diacritic marks from the text.

    Arabic script uses diacritic marks for short vowels and other
    phonetic cues. These marks can be removed to simplify matching
    and tokenisation.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text with diacritics removed.
    """
    return DIACRITICS_PATTERN.sub("", text)


def normalise_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text with consecutive whitespace replaced by a single space.
    """
    return " ".join(text.split())


def preprocess(text: str) -> str:
    """Apply a sequence of normalisation steps to Persian text.

    This function applies character normalisation, diacritic removal and
    whitespace normalisation. Additional preprocessing steps can be
    added as needed.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Cleaned and normalised text.
    """
    text = normalise_characters(text)
    text = remove_diacritics(text)
    text = normalise_whitespace(text)
    return text
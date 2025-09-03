"""Persian tokenisation utilities.

This module provides a simple interface for tokenising Persian text.
If the `naghz` library is installed, it will be used to perform
high‑quality tokenisation. Otherwise, a fallback implementation
splits on whitespace and punctuation.  You can install `naghz` from
GitHub:

    pip install git+https://github.com/bazpardazesh-org/naghz.git

Example:

    >>> from rakhshai_graph_nlp.features.tokenizer import tokenize
    >>> tokens = tokenize("سلام دنیا!")
    >>> print(tokens)
    ['سلام', 'دنیا']
"""

from __future__ import annotations

import re
from typing import List

try:
    from naghz import Normalizer, Tokenizer  # type: ignore
    _naghz_available = True
except ImportError:
    _naghz_available = False

def tokenize(text: str) -> List[str]:
    """Tokenise Persian text into a list of tokens.

    If the Naghz library is installed, this function will normalise
    and tokenise the text using Naghz's models. Otherwise, it falls
    back to a simple regex based tokeniser that splits on
    whitespace and punctuation.

    Parameters
    ----------
    text : str
        Input text in Persian.

    Returns
    -------
    List[str]
        List of token strings.
    """
    if _naghz_available:
        # Initialise normaliser and tokeniser once
        if not hasattr(tokenize, "_norm"):
            tokenize._norm = Normalizer()
            tokenize._tok = Tokenizer()
        norm = tokenize._norm
        tok = tokenize._tok
        normalized = norm.normalize(text)
        tokens = tok.tokenize(normalized)
        # Naghz returns tokens as dictionaries with 'text' field
        return [t["text"] if isinstance(t, dict) else str(t) for t in tokens]
    # Fallback: simple regex splitting
    # Replace half spaces with normal spaces
    text = text.replace("\u200c", " ")
    # Remove diacritics and punctuation by spacing them out
    tokens = re.findall(r"[\w\d]+", text, flags=re.UNICODE)
    return tokens
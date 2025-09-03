"""Unicode-aware tokenisation and sentence splitting."""

from __future__ import annotations

from typing import List

try:  # pragma: no cover - optional dependency
    import regex as _re
except Exception:  # pragma: no cover
    import re as _re  # type: ignore

try:  # pragma: no cover - optional dependency
    import stanza
except Exception:  # pragma: no cover
    stanza = None

_WORD_RE = _re.compile(r"\p{L}[\p{L}\p{N}_]*|\p{N}+", _re.UNICODE)
_SENT_RE = _re.compile(r"[^.!?؟؛…]+", _re.UNICODE)


def tokenize(text: str, use_stanza: bool = False) -> List[str]:
    """Return a list of word tokens for ``text``."""
    if use_stanza and stanza is not None:  # pragma: no cover - heavy
        nlp = stanza.Pipeline(lang="multilingual", processors="tokenize")
        doc = nlp(text)
        return [t.text for s in doc.sentences for t in s.tokens]
    return _WORD_RE.findall(text)


def split_sentences(text: str, use_stanza: bool = False) -> List[str]:
    """Split ``text`` into sentences."""
    if use_stanza and stanza is not None:  # pragma: no cover
        nlp = stanza.Pipeline(lang="multilingual", processors="tokenize")
        doc = nlp(text)
        return [s.text for s in doc.sentences]
    spans = _SENT_RE.findall(text)
    return [s.strip() for s in spans if s.strip()]

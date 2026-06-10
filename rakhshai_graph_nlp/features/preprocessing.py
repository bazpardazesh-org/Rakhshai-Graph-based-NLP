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
import unicodedata
from dataclasses import asdict, dataclass


ARABIC_YE = "ي"
PERSIAN_YE = "ی"
ARABIC_KE = "ك"
PERSIAN_KE = "ک"
HALF_SPACE = "\u200c"

DIACRITICS_PATTERN = re.compile(
    r"[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655\u0670]"
)
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200d\ufeff]")

ARABIC_TO_PERSIAN_CHARS = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ۀ": "ه",
        "ة": "ه",
        "ؤ": "و",
    }
)

ARABIC_PERSIAN_DIGITS = str.maketrans(
    {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)


@dataclass
class PersianNormalizerConfig:
    """Configuration for Persian text normalisation.

    ``half_space`` controls ZWNJ handling:

    - ``preserve`` keeps Persian half-spaces and normalises surrounding spaces.
    - ``split`` replaces half-spaces with normal whitespace.
    - ``remove`` removes half-spaces without inserting whitespace.
    """

    half_space: str = "preserve"
    normalize_digits: bool = True
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    unicode_normalization: str = "NFC"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "PersianNormalizerConfig":
        if not data:
            return cls()
        return cls(
            half_space=str(data.get("half_space", "preserve")),
            normalize_digits=bool(data.get("normalize_digits", True)),
            remove_diacritics=bool(data.get("remove_diacritics", True)),
            remove_tatweel=bool(data.get("remove_tatweel", True)),
            unicode_normalization=str(data.get("unicode_normalization", "NFC")),
        )


class PersianNormalizer:
    """Reusable Persian normalizer shared by LM and feature pipelines."""

    def __init__(self, config: PersianNormalizerConfig | None = None):
        self.config = config or PersianNormalizerConfig()
        if self.config.half_space not in {"preserve", "split", "remove"}:
            raise ValueError("half_space must be one of: preserve, split, remove")

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.config.unicode_normalization, text)
        text = text.translate(ARABIC_TO_PERSIAN_CHARS)
        if self.config.normalize_digits:
            text = text.translate(ARABIC_PERSIAN_DIGITS)
        if self.config.remove_tatweel:
            text = text.replace("\u0640", "")
        if self.config.remove_diacritics:
            text = DIACRITICS_PATTERN.sub("", text)

        text = ZERO_WIDTH_PATTERN.sub("", text)
        if self.config.half_space == "preserve":
            text = re.sub(r"\s*\u200c\s*", HALF_SPACE, text)
        elif self.config.half_space == "split":
            text = text.replace(HALF_SPACE, " ")
        else:
            text = text.replace(HALF_SPACE, "")
        return normalise_whitespace(text)


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


def normalize_persian_text(
    text: str,
    *,
    config: PersianNormalizerConfig | None = None,
    half_space: str | None = None,
) -> str:
    """Normalize Persian text with the shared configurable normalizer."""

    cfg = config or PersianNormalizerConfig()
    if half_space is not None:
        cfg = PersianNormalizerConfig(
            half_space=half_space,
            normalize_digits=cfg.normalize_digits,
            remove_diacritics=cfg.remove_diacritics,
            remove_tatweel=cfg.remove_tatweel,
            unicode_normalization=cfg.unicode_normalization,
        )
    return PersianNormalizer(cfg).normalize(text)

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

# Always-applied orthographic unification (Arabic letter forms → Persian).
# Folding of hamza-bearing letters (ئ, ؤ) and the ezafe marker (ۀ / ه + U+0654)
# is handled separately so it can be disabled — see ``normalize_hamza`` and
# ``ezafe_mode`` in :class:`PersianNormalizerConfig`.
ARABIC_TO_PERSIAN_CHARS = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
    }
)

# Hamza-bearing letters folded onto their plain Persian counterparts. Applied
# only when ``normalize_hamza`` is enabled, since this merges orthographically
# distinct spellings (e.g. مسائل → مسایل, مؤثر → موثر).
HAMZA_TO_PERSIAN_CHARS = str.maketrans(
    {
        "ئ": "ی",
        "ؤ": "و",
    }
)

HEH = "ه"  # ه
PERSIAN_YE_CHAR = "ی"  # ی
HAMZA_ABOVE = "ٔ"  # combining hamza above (ezafe written as ه + ٔ)
HEH_WITH_YEH_ABOVE = "ۀ"  # ۀ (precomposed ezafe heh)

# Persian numeric separators → ASCII equivalents so decimals and digit grouping
# survive tokenisation instead of splitting numbers into pieces.
NUMERIC_SEPARATORS = str.maketrans(
    {
        "٫": ".",  # ARABIC DECIMAL SEPARATOR ٫
        "٬": ",",  # ARABIC THOUSANDS SEPARATOR ٬
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

    ``ezafe_mode`` controls how the ezafe (ۀ / ه + U+0654) is handled:

    - ``marker`` (default) rewrites it as ``ه‌ی`` so the grammatical ezafe
      survives as an explicit, learnable token.
    - ``collapse`` folds it onto a plain ``ه`` (loses the construction).
    """

    half_space: str = "preserve"
    normalize_digits: bool = True
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    unicode_normalization: str = "NFC"
    normalize_hamza: bool = True
    ezafe_mode: str = "marker"

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
            normalize_hamza=bool(data.get("normalize_hamza", True)),
            # Configs serialised before this field existed were produced under
            # the old collapse behaviour; fall back to it so already-saved
            # tokenisers/models reproduce their original normalisation.
            ezafe_mode=str(data.get("ezafe_mode", "collapse")),
        )


class PersianNormalizer:
    """Reusable Persian normalizer shared by LM and feature pipelines."""

    def __init__(self, config: PersianNormalizerConfig | None = None):
        self.config = config or PersianNormalizerConfig()
        if self.config.half_space not in {"preserve", "split", "remove"}:
            raise ValueError("half_space must be one of: preserve, split, remove")
        if self.config.ezafe_mode not in {"collapse", "marker"}:
            raise ValueError("ezafe_mode must be one of: collapse, marker")

    def _normalize_ezafe(self, text: str) -> str:
        """Normalise the ezafe written as \u06c0 or as \u0647 + combining hamza (U+0654).

        This must run *before* diacritic removal, otherwise the combining
        hamza is stripped and the ezafe information is lost silently. In
        ``marker`` mode the ezafe is rewritten as ``\u0647\u200c\u06cc`` so the grammatical
        construction survives as an explicit token; ``collapse`` keeps the
        historical behaviour of folding it onto a plain ``\u0647``.
        """

        marker = self.config.ezafe_mode == "marker"
        replacement = f"{HEH}\u200c{PERSIAN_YE_CHAR}" if marker else HEH
        text = text.replace(f"{HEH}{HAMZA_ABOVE}", replacement)
        text = text.replace(HEH_WITH_YEH_ABOVE, replacement)
        return text

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.config.unicode_normalization, text)
        text = text.translate(ARABIC_TO_PERSIAN_CHARS)
        if self.config.normalize_hamza:
            text = text.translate(HAMZA_TO_PERSIAN_CHARS)
        text = self._normalize_ezafe(text)
        if self.config.normalize_digits:
            text = text.translate(ARABIC_PERSIAN_DIGITS)
            text = text.translate(NUMERIC_SEPARATORS)
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
            normalize_hamza=cfg.normalize_hamza,
            ezafe_mode=cfg.ezafe_mode,
        )
    return PersianNormalizer(cfg).normalize(text)

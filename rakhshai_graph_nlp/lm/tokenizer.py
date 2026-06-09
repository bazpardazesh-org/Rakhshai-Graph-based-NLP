"""Persian word-level tokenizer for causal language modelling."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..features.preprocessing import DIACRITICS_PATTERN


TOKEN_PATTERN = re.compile(
    r"[\u0600-\u06FF]+(?:\u200c[\u0600-\u06FF]+)*|[A-Za-z]+|\d+(?:[./]\d+)*",
    flags=re.UNICODE,
)


@dataclass
class PersianTokenizer:
    """A compact Persian tokenizer that maps cleaned text to token ids.

    The tokenizer normalises Arabic ``ي``/``ك`` to Persian ``ی``/``ک``, removes
    diacritics, keeps or splits half-spaces by configuration, and stores a
    JSON vocabulary suitable for full Graph-LM save/load.
    """

    token_to_id: dict[str, int] = field(default_factory=dict)
    keep_half_space: bool = True
    min_freq: int = 1
    max_vocab_size: int | None = None
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    def __post_init__(self) -> None:
        if not self.token_to_id:
            self.token_to_id = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3,
            }
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def normalize(self, text: str) -> str:
        """Clean Persian text while preserving language-model useful tokens."""

        text = text.replace("ي", "ی").replace("ك", "ک")
        text = text.replace("\u0640", "")
        text = DIACRITICS_PATTERN.sub("", text)
        if self.keep_half_space:
            text = re.sub(r"[\u200b\u200d\ufeff]", "", text)
            text = re.sub(r"\s*\u200c\s*", "\u200c", text)
        else:
            text = re.sub(r"[\u200b\u200c\u200d\ufeff]", " ", text)
        return " ".join(text.split())

    def tokenize(self, text: str) -> list[str]:
        return TOKEN_PATTERN.findall(self.normalize(text))

    def fit(self, texts: Iterable[str]) -> "PersianTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize(text))

        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        vocab_items = [
            token
            for token, count in counts.most_common()
            if count >= self.min_freq and token not in special_tokens
        ]
        if self.max_vocab_size is not None:
            vocab_items = vocab_items[: max(0, self.max_vocab_size - len(special_tokens))]

        self.token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
        for token in vocab_items:
            self.token_to_id[token] = len(self.token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        return self

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        special = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special_tokens and token in special:
                continue
            tokens.append(token)
        return " ".join(tokens).replace(" \u200c ", "\u200c")

    def to_dict(self) -> dict[str, object]:
        return {
            "token_to_id": self.token_to_id,
            "keep_half_space": self.keep_half_space,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "special_tokens": {
                "pad_token": self.pad_token,
                "unk_token": self.unk_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PersianTokenizer":
        special = data.get("special_tokens", {})
        if not isinstance(special, dict):
            special = {}
        token_to_id = {
            str(token): int(idx)
            for token, idx in dict(data["token_to_id"]).items()  # type: ignore[arg-type]
        }
        return cls(
            token_to_id=token_to_id,
            keep_half_space=bool(data.get("keep_half_space", True)),
            min_freq=int(data.get("min_freq", 1)),
            max_vocab_size=data.get("max_vocab_size"),  # type: ignore[arg-type]
            pad_token=str(special.get("pad_token", "<pad>")),
            unk_token=str(special.get("unk_token", "<unk>")),
            bos_token=str(special.get("bos_token", "<bos>")),
            eos_token=str(special.get("eos_token", "<eos>")),
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PersianTokenizer":
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

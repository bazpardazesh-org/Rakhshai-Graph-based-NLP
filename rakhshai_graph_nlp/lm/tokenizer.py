"""Persian word/subword tokenizer for causal language modelling."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..features.preprocessing import PersianNormalizerConfig, normalize_persian_text


TOKEN_PATTERN = re.compile(
    r"[\u0600-\u06FF]+(?:\u200c[\u0600-\u06FF]+)*|[A-Za-z]+|\d+(?:[./]\d+)*",
    flags=re.UNICODE,
)

COMPOUND_LIGHT_VERBS = (
    "کرد",
    "کردم",
    "کردی",
    "کردند",
    "می\u200cکند",
    "نمی\u200cکند",
    "شد",
    "شدم",
    "شدند",
    "گرفت",
    "گرفتم",
    "گرفتند",
    "داد",
    "دادم",
    "دادند",
    "داشت",
    "داشتم",
    "داشتند",
    "آورد",
    "آوردند",
)

PREFIXES = ("نمی\u200c", "می\u200c", "بی\u200c", "هم\u200c")
SUFFIXES = (
    "هایمان",
    "هایتان",
    "هایشان",
    "هایی",
    "های",
    "ها",
    "ترین",
    "تر",
    "مان",
    "تان",
    "شان",
    "ام",
    "ات",
    "اش",
)
BPE_END = "</w>"


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
    tokenizer_type: str = "word"
    subword_chunk_size: int = 3
    normalizer_config: PersianNormalizerConfig = field(default_factory=PersianNormalizerConfig)
    morph_splitting: bool = False
    compound_verb_mode: str = "none"
    bpe_num_merges: int = 200
    bpe_merges: list[tuple[str, str]] = field(default_factory=list)
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    tokenizer_version: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.normalizer_config, PersianNormalizerConfig):
            self.normalizer_config = PersianNormalizerConfig.from_dict(
                self.normalizer_config  # type: ignore[arg-type]
            )
        if self.compound_verb_mode not in {"none", "join"}:
            raise ValueError("compound_verb_mode must be one of: none, join")
        if self.tokenizer_type == "subword":
            self.tokenizer_type = "char_chunk"
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

        half_space = "preserve" if self.keep_half_space else "split"
        text = normalize_persian_text(
            text,
            config=self.normalizer_config,
            half_space=half_space,
        )
        if self.compound_verb_mode == "join":
            text = self._join_compound_verbs(text)
        return text

    def tokenize(self, text: str) -> list[str]:
        words = self._word_tokens(text)
        if self.tokenizer_type == "word":
            return words
        if self.tokenizer_type == "char_chunk":
            tokens: list[str] = []
            for word in words:
                tokens.extend(self._word_to_subwords(word))
            return tokens
        if self.tokenizer_type == "bpe":
            tokens = []
            for word in words:
                tokens.extend(self._word_to_bpe_pieces(word))
            return tokens
        if self.tokenizer_type != "unigram":
            raise ValueError("tokenizer_type must be one of: word, char_chunk, bpe, unigram")
        # Lightweight fallback: BPE with fewer assumptions until a separate unigram
        # trainer is needed.
        tokens: list[str] = []
        for word in words:
            tokens.extend(self._word_to_bpe_pieces(word))
        return tokens

    def _word_tokens(self, text: str) -> list[str]:
        words = TOKEN_PATTERN.findall(self.normalize(text))
        if not self.morph_splitting:
            return words
        tokens: list[str] = []
        for word in words:
            tokens.extend(self._split_morphology(word))
        return [token for token in tokens if token]

    def _join_compound_verbs(self, text: str) -> str:
        escaped = "|".join(re.escape(verb) for verb in COMPOUND_LIGHT_VERBS)
        pattern = re.compile(rf"([\u0600-\u06FF]+)\s+({escaped})(?=\s|$)")
        return pattern.sub(lambda match: f"{match.group(1)}\u200c{match.group(2)}", text)

    def _split_morphology(self, word: str) -> list[str]:
        pieces: list[str] = []
        stem = word
        for prefix in PREFIXES:
            if stem.startswith(prefix) and len(stem) > len(prefix) + 1:
                pieces.append(prefix.rstrip("\u200c"))
                stem = stem[len(prefix) :]
                break
        suffix_piece = ""
        for suffix in SUFFIXES:
            if stem.endswith(suffix) and len(stem) > len(suffix) + 1:
                suffix_piece = suffix
                stem = stem[: -len(suffix)].rstrip("\u200c")
                break
        if stem:
            pieces.append(stem)
        if suffix_piece:
            pieces.append("##" + suffix_piece)
        return pieces or [word]

    def _word_to_subwords(self, word: str) -> list[str]:
        if len(word) <= self.subword_chunk_size + 1:
            return [word]
        first = word[: self.subword_chunk_size]
        chunks = [
            "##" + word[i : i + self.subword_chunk_size]
            for i in range(self.subword_chunk_size, len(word), self.subword_chunk_size)
        ]
        return [first, *chunks]

    def _word_to_bpe_pieces(self, word: str) -> list[str]:
        if not word:
            return []
        pieces = list(word) + [BPE_END]
        for left, right in self.bpe_merges:
            pieces = self._apply_bpe_merge(pieces, left, right)
        if pieces and pieces[-1] == BPE_END:
            pieces = pieces[:-1]
        elif pieces and pieces[-1].endswith(BPE_END):
            pieces[-1] = pieces[-1][: -len(BPE_END)]
        pieces = [piece for piece in pieces if piece]
        if not pieces:
            return [word]
        return [pieces[0], *["##" + piece for piece in pieces[1:]]]

    @staticmethod
    def _apply_bpe_merge(pieces: list[str], left: str, right: str) -> list[str]:
        merged: list[str] = []
        i = 0
        while i < len(pieces):
            if i < len(pieces) - 1 and pieces[i] == left and pieces[i + 1] == right:
                merged.append(left + right)
                i += 2
            else:
                merged.append(pieces[i])
                i += 1
        return merged

    def _train_bpe(self, words: Iterable[str]) -> None:
        vocab = Counter(tuple(word) + (BPE_END,) for word in words if word)
        merges: list[tuple[str, str]] = []
        for _ in range(max(0, self.bpe_num_merges)):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for pieces, count in vocab.items():
                for i in range(len(pieces) - 1):
                    pair_counts[(pieces[i], pieces[i + 1])] += count
            if not pair_counts:
                break
            pair, freq = pair_counts.most_common(1)[0]
            if freq < 2:
                break
            merges.append(pair)
            new_vocab: Counter[tuple[str, ...]] = Counter()
            for pieces, count in vocab.items():
                merged = tuple(self._apply_bpe_merge(list(pieces), pair[0], pair[1]))
                new_vocab[merged] += count
            vocab = new_vocab
        self.bpe_merges = merges

    def fit(self, texts: Iterable[str]) -> "PersianTokenizer":
        texts = list(texts)
        if self.tokenizer_type in {"bpe", "unigram"}:
            self._train_bpe(word for text in texts for word in self._word_tokens(text))

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
        if self.tokenizer_type in {"char_chunk", "bpe", "unigram"}:
            words: list[str] = []
            for token in tokens:
                if token.startswith("##") and words:
                    words[-1] += token[2:]
                else:
                    words.append(token[2:] if token.startswith("##") else token)
            tokens = words
        return " ".join(tokens).replace(" \u200c ", "\u200c")

    def to_dict(self) -> dict[str, object]:
        return {
            "tokenizer_version": self.tokenizer_version,
            "token_to_id": self.token_to_id,
            "keep_half_space": self.keep_half_space,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "tokenizer_type": self.tokenizer_type,
            "subword_chunk_size": self.subword_chunk_size,
            "normalizer_config": self.normalizer_config.to_dict(),
            "morph_splitting": self.morph_splitting,
            "compound_verb_mode": self.compound_verb_mode,
            "bpe_num_merges": self.bpe_num_merges,
            "bpe_merges": [list(pair) for pair in self.bpe_merges],
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
        tokenizer_type = data.get("tokenizer_type") or "word"
        if tokenizer_type == "subword":
            tokenizer_type = "char_chunk"
        raw_merges = data.get("bpe_merges", [])
        bpe_merges = [
            (str(pair[0]), str(pair[1]))
            for pair in raw_merges  # type: ignore[union-attr]
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
        return cls(
            token_to_id=token_to_id,
            keep_half_space=bool(data.get("keep_half_space", True)),
            min_freq=int(data.get("min_freq", 1)),
            max_vocab_size=data.get("max_vocab_size"),  # type: ignore[arg-type]
            tokenizer_type=str(tokenizer_type),
            subword_chunk_size=int(data.get("subword_chunk_size", 3)),
            normalizer_config=PersianNormalizerConfig.from_dict(
                data.get("normalizer_config")  # type: ignore[arg-type]
            ),
            morph_splitting=bool(data.get("morph_splitting", False)),
            compound_verb_mode=str(data.get("compound_verb_mode", "none")),
            bpe_num_merges=int(data.get("bpe_num_merges", 200)),
            bpe_merges=bpe_merges,
            pad_token=str(special.get("pad_token", "<pad>")),
            unk_token=str(special.get("unk_token", "<unk>")),
            bos_token=str(special.get("bos_token", "<bos>")),
            eos_token=str(special.get("eos_token", "<eos>")),
            tokenizer_version=int(data.get("tokenizer_version", 1)),
        )

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "PersianTokenizer":
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

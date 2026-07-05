"""Token shard datasets for scalable native LM pretraining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import BYTE_TOKENS, PersianTokenizer


@dataclass
class TokenShardConfig:
    """Configuration for writing local token shards."""

    output_dir: str
    input_paths: Sequence[str] | None = None
    corpus_dir: str | None = None
    tokenizer_path: str | None = None
    tokenizer_type: str = "unigram"
    tokenizer_half_space: str = "preserve"
    tokenizer_morph_splitting: bool = False
    tokenizer_compound_verb_mode: str = "none"
    tokenizer_bpe_merges: int = 200
    tokenizer_unigram_num_pieces: int = 8000
    tokenizer_max_vocab_size: int | None = None
    byte_fallback: bool = False
    block_size: int = 128
    stride: int | None = None
    tokens_per_shard: int = 2_000_000
    seed: int = 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _discover_split_paths(config: TokenShardConfig) -> dict[str, list[Path]]:
    if config.corpus_dir:
        root = Path(config.corpus_dir)
        splits = {
            split: [root / f"{split}.txt"]
            for split in ["train", "validation", "test"]
            if (root / f"{split}.txt").exists()
        }
        if not splits and (root / "corpus.txt").exists():
            splits = {"train": [root / "corpus.txt"]}
        if splits:
            return splits
    if not config.input_paths:
        raise ValueError("TokenShardConfig requires input_paths or corpus_dir")
    return {"train": [Path(path) for path in config.input_paths]}


def _load_split_texts(paths: Sequence[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        texts.extend(_read_lines(path))
    return texts


def tokenizer_audit(tokenizer: PersianTokenizer, texts: Iterable[str]) -> dict[str, object]:
    texts = list(texts)
    token_count = 0
    char_count = 0
    unk_count = 0
    byte_count = 0
    half_space_count = 0
    continuation_count = 0
    split_counts: dict[str, int] = {}
    byte_ids = {tokenizer.token_to_id[token] for token in BYTE_TOKENS if token in tokenizer.token_to_id}
    for text in texts:
        char_count += len(text)
        half_space_count += text.count("\u200c")
        tokens = tokenizer.tokenize(text)
        split_counts["wordpieces"] = split_counts.get("wordpieces", 0) + len(tokens)
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_count += len(ids)
        unk_count += sum(1 for token_id in ids if token_id == tokenizer.unk_id)
        byte_count += sum(1 for token_id in ids if token_id in byte_ids)
        continuation_count += sum(1 for token in tokens if token.startswith("##"))
    return {
        "tokenizer_type": tokenizer.tokenizer_type,
        "vocab_size": tokenizer.vocab_size,
        "byte_fallback": tokenizer.byte_fallback,
        "texts": len(texts),
        "characters": char_count,
        "tokens": token_count,
        "unknown_tokens": unk_count,
        "unknown_rate": unk_count / max(1, token_count),
        "byte_fallback_tokens": byte_count,
        "byte_fallback_rate": byte_count / max(1, token_count),
        "half_space_count": half_space_count,
        "half_space_rate": half_space_count / max(1, char_count),
        "continuation_piece_count": continuation_count,
        "continuation_piece_rate": continuation_count / max(1, token_count),
        "tokens_per_character": token_count / max(1, char_count),
        "split_counts": split_counts,
    }


def _build_tokenizer(config: TokenShardConfig, train_texts: Sequence[str]) -> PersianTokenizer:
    if config.tokenizer_path:
        return PersianTokenizer.load(config.tokenizer_path)
    return PersianTokenizer(
        tokenizer_type=config.tokenizer_type,
        keep_half_space=config.tokenizer_half_space == "preserve",
        morph_splitting=config.tokenizer_morph_splitting,
        compound_verb_mode=config.tokenizer_compound_verb_mode,
        bpe_num_merges=config.tokenizer_bpe_merges,
        unigram_num_pieces=config.tokenizer_unigram_num_pieces,
        max_vocab_size=config.tokenizer_max_vocab_size,
        byte_fallback=config.byte_fallback,
    ).fit(train_texts)


def _texts_to_token_ids(texts: Sequence[str], tokenizer: PersianTokenizer) -> np.ndarray:
    ids: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        if len(encoded) > 1:
            ids.extend(encoded)
    if len(ids) < 2:
        raise ValueError("token shard input must contain at least two tokens")
    return np.asarray(ids, dtype=np.int64)


def _write_split_shards(
    split: str,
    ids: np.ndarray,
    *,
    output_dir: Path,
    tokens_per_shard: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, start in enumerate(range(0, len(ids), max(1, tokens_per_shard))):
        shard_ids = ids[start : start + max(1, tokens_per_shard)]
        path = output_dir / f"{split}-{index:05d}.tokens.bin"
        shard_ids.astype(np.int64).tofile(path)
        rows.append(
            {
                "split": split,
                "path": path.name,
                "dtype": "int64",
                "num_tokens": int(shard_ids.size),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def write_token_shards(config: TokenShardConfig) -> dict[str, object]:
    """Write token shards and a manifest from local corpus files."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = _discover_split_paths(config)
    split_texts = {
        split: texts
        for split, paths in split_paths.items()
        if (texts := _load_split_texts(paths))
    }
    if not split_texts:
        raise ValueError("token shard input contains no non-empty splits")
    tokenizer = _build_tokenizer(config, split_texts.get("train") or next(iter(split_texts.values())))
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    shard_rows: list[dict[str, object]] = []
    audit = {split: tokenizer_audit(tokenizer, texts) for split, texts in split_texts.items()}
    for split, texts in split_texts.items():
        ids = _texts_to_token_ids(texts, tokenizer)
        shard_rows.extend(
            _write_split_shards(
                split,
                ids,
                output_dir=output_dir,
                tokens_per_shard=config.tokens_per_shard,
            )
        )
    manifest = {
        "config": asdict(config),
        "tokenizer": "tokenizer.json",
        "block_size": config.block_size,
        "stride": config.stride or config.block_size,
        "shards": shard_rows,
        "audit": audit,
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
    }
    (output_dir / "shard_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


class TokenShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy fixed-window dataset over memory-mapped token shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "train",
        block_size: int | None = None,
        stride: int | None = None,
        pad_token_id: int = 0,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.block_size = int(block_size or self.manifest.get("block_size", 128))
        self.stride = int(stride or self.manifest.get("stride", self.block_size))
        self.pad_token_id = int(pad_token_id)
        self.shards: list[np.memmap] = []
        self.index: list[tuple[int, int]] = []
        for shard_index, row in enumerate(self.manifest.get("shards", [])):
            if row.get("split") != split:
                continue
            path = self.root / str(row["path"])
            data = np.memmap(path, dtype=np.int64, mode="r")
            self.shards.append(data)
            if data.size < 2:
                continue
            for start in range(0, max(1, data.size - 1), self.stride):
                self.index.append((len(self.shards) - 1, start))
        if not self.index:
            raise ValueError(f"no token shard windows found for split {split!r}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_index, start = self.index[index]
        data = self.shards[shard_index]
        chunk = np.asarray(data[start : start + self.block_size + 1], dtype=np.int64)
        input_ids = chunk[:-1].tolist()
        labels = chunk[1:].tolist()
        pad_len = self.block_size - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

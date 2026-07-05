"""Dataset, training and generation helpers for native Persian article LMs."""

from __future__ import annotations

import csv
from collections import Counter
import json
import math
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from ...lm.graph_builder import build_graph_lm_graph
from ...lm.graph_memory import GraphMemoryArtifact, GraphMemoryConfig
from ...lm.model import GenerationConfig, GraphCausalLM, GraphLMConfig
from ...lm.tokenizer import PersianTokenizer
from ...lm.trainer import LMTrainingConfig, train_graph_lm


DEFAULT_ARTICLE_GRAPH_RELATIONS = (
    "cooccurrence",
    "pmi",
    "dependency",
    "stem",
    "subword",
    "word_document",
    "topic_document",
)


@dataclass
class ArticleCorpusConfig:
    """Configuration for converting raw Persian articles to Graph-LM corpus files."""

    input_path: str
    output_dir: str
    input_format: str = "auto"
    training_format: str = "article_fields"
    title_field: str = "title"
    body_field: str = "body"
    summary_field: str = "summary"
    keywords_field: str = "keywords"
    metadata_field: str = "metadata"
    prompt_audience: str = "عمومی"
    prompt_tone: str = "دانشنامه‌ای"
    prompt_sections: int = 3
    min_body_chars: int = 80
    validation_ratio: float = 0.1
    seed: int = 0


@dataclass
class ArticleTrainingConfig:
    """Article-oriented training profile wrapping the existing Graph-LM trainer."""

    corpus_path: str
    output_dir: str = "runs/article-llm"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    validation_ratio: float = 0.1
    block_size: int = 128
    stride: int | None = None
    min_freq: int = 1
    max_vocab_size: int | None = None
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    graph_encoder: str = "gat"
    graph_hidden_dim: int = 128
    graph_heads: int = 4
    fusion: str = "context_gated"
    fusion_layers: str = "input"
    fusion_levels: str = "token"
    graph_window_size: int = 4
    graph_min_count: int = 1
    graph_weighting: str = "distance"
    graph_min_edge_weight: float = 0.0
    graph_top_k: int | None = None
    graph_directed: bool = False
    graph_scope: str = "document"
    context_node_type: str = "none"
    graph_relations: Sequence[str] | None = field(
        default_factory=lambda: list(DEFAULT_ARTICLE_GRAPH_RELATIONS)
    )
    relation_weights: dict[str, float] | None = None
    semantic_similarity_threshold: float = 0.6
    semantic_top_k: int | None = 4
    semantic_method: str = "distributional"
    linguistic_backend: str = "auto"
    topic_top_k: int = 8
    graph_relation_mode: str = "embedding"
    graph_pooling: str = "none"
    graph_node_importance: bool = False
    graph_node_type_embedding: bool = True
    graph_fusion_scale: float = 1.0
    graph_fusion_dropout: float = 0.0
    task_losses: str = (
        "next_token,masked_token,edge,neighbor,node_relation,graph_text,"
        "sentence_graph"
    )
    next_token_weight: float = 1.0
    masked_token_weight: float = 0.25
    edge_prediction_weight: float = 0.1
    neighbor_prediction_weight: float = 0.1
    node_relation_weight: float = 0.1
    graph_text_alignment_weight: float = 0.1
    sentence_graph_alignment_weight: float = 0.1
    mask_probability: float = 0.15
    negative_samples: int = 1
    text_augmentation: bool = True
    augmentation_ratio: float = 0.5
    token_dropout: float = 0.05
    punctuation_dropout: float = 0.5
    node_dropout: float = 0.05
    edge_dropout: float = 0.1
    subgraph_sampling_ratio: float = 0.9
    contrastive_weight: float = 0.05
    curriculum_learning: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    checkpoint_metric: str = "next_token"
    max_grad_norm: float = 1.0
    dynamic_graph: bool = False
    graph_build_batch_size: int | None = None
    graph_cache_dir: str | None = None
    reuse_graph_cache: bool = True
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    amp: bool = False
    resume_from: str | None = None
    tokenizer_type: str = "unigram"
    tokenizer_half_space: str = "preserve"
    tokenizer_morph_splitting: bool = False
    tokenizer_compound_verb_mode: str = "none"
    tokenizer_bpe_merges: int = 200
    tokenizer_unigram_num_pieces: int = 8000
    device: str = "cpu"
    seed: int = 0


@dataclass
class ArticleGenerationConfig:
    """Configuration for loading a trained checkpoint and generating an article."""

    model_dir: str
    topic: str
    audience: str = "عمومی"
    tone: str = "تحلیلی"
    sections: int = 3
    min_new_tokens: int = 80
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_k: int = 50
    repetition_penalty: float = 1.15
    graph_memory: bool = True
    graph_memory_top_k: int = 32
    graph_memory_depth: int = 1
    graph_memory_max_edges: int = 256
    graph_memory_min_score: float = 0.0
    graph_memory_relation_weights: dict[str, float] | None = None
    device: str = "cpu"


@dataclass
class ArticleAuditConfig:
    """Native corpus and tokenizer audit for Persian article training data."""

    input_path: str
    output_dir: str
    input_format: str = "auto"
    training_format: str = "article_fields"
    title_field: str = "title"
    body_field: str = "body"
    summary_field: str = "summary"
    keywords_field: str = "keywords"
    metadata_field: str = "metadata"
    min_body_chars: int = 80
    validation_ratio: float = 0.1
    seed: int = 0
    tokenizer_types: Sequence[str] = field(
        default_factory=lambda: ["word", "bpe", "unigram"]
    )
    tokenizer_half_space: str = "preserve"
    tokenizer_morph_splitting: bool = False
    tokenizer_compound_verb_mode: str = "none"
    tokenizer_bpe_merges: int = 200
    tokenizer_unigram_num_pieces: int = 8000
    tokenizer_max_vocab_size: int | None = None
    tokenizer_probe_epochs: int = 0
    tokenizer_probe_block_size: int = 64
    tokenizer_probe_d_model: int = 64
    tokenizer_probe_n_heads: int = 4
    tokenizer_probe_n_layers: int = 1
    tokenizer_probe_batch_size: int = 4
    near_duplicate_ngram_size: int = 13
    near_duplicate_threshold: float = 0.82
    near_duplicate_max_pairs: int = 20000
    max_examples: int | None = None
    device: str = "cpu"


@dataclass
class ArticleAblationConfig:
    """Train native graph ablation variants from the same article corpus."""

    training_config: ArticleTrainingConfig
    output_dir: str = "runs/article-ablation"
    graph_encoders: Sequence[str] = field(
        default_factory=lambda: ["none", "gat", "rgcn"]
    )
    graph_scopes: Sequence[str] = field(default_factory=lambda: ["document", "sentence"])
    relation_groups: dict[str, Sequence[str]] | None = None
    probe_topic: str | None = None
    probe_sections: int = 2
    probe_max_new_tokens: int = 24


@dataclass
class PersianArticle:
    """Structured Persian article generated by a trained Article Graph-LM."""

    title: str
    introduction: str
    sections: list[dict[str, str]]
    conclusion: str
    full_markdown: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _detect_format(path: Path, requested: str) -> str:
    fmt = requested.lower()
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    return "txt"


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u200c", "\u200c")
    return re.sub(r"\s+", " ", text).strip()


def _coerce_keywords(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    return [
        part
        for part in (_clean_text(item) for item in re.split(r"[,،;؛|]", str(value)))
        if part
    ]


def _coerce_metadata(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def _normalise_training_format(value: str) -> str:
    fmt = value.lower().replace("-", "_")
    if fmt not in {"article_fields", "prompt_completion", "wikipedia_prompt"}:
        raise ValueError(
            "training_format must be one of: article_fields, "
            "prompt_completion, wikipedia_prompt"
        )
    return fmt


def _body_from_row(row: dict[str, Any], config: ArticleCorpusConfig) -> object:
    body = row.get(config.body_field, "")
    if (
        body == ""
        and _normalise_training_format(config.training_format) == "wikipedia_prompt"
    ):
        return row.get("text", "")
    return body


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.؟!])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _chunk_items(items: Sequence[str], count: int) -> list[list[str]]:
    if count <= 0:
        return []
    if not items:
        return [[] for _ in range(count)]
    chunk_size = max(1, math.ceil(len(items) / count))
    chunks = [
        list(items[i : i + chunk_size])
        for i in range(0, len(items), chunk_size)
    ]
    while len(chunks) < count:
        chunks.append([])
    return chunks[:count]


def _prompt_header(
    *,
    topic: str,
    audience: str,
    tone: str,
    sections: int,
) -> list[str]:
    """Conditioning header shared verbatim between training text and prompts.

    Training corpora and the generation-time prompt must present the topic,
    audience, tone and structure fields with identical surface forms, otherwise
    the model is conditioned on tokens it never saw during training.
    """

    section_count = max(1, int(sections))
    return [
        f"موضوع مقاله: {topic}",
        f"مخاطب: {audience}",
        f"لحن: {tone}",
        f"ساختار: عنوان، مقدمه، {section_count} بخش، جمع‌بندی",
        "مقاله:",
    ]


def _record_audience_tone(
    record: dict[str, Any],
    config: ArticleCorpusConfig,
) -> tuple[str, str]:
    metadata = _coerce_metadata(record.get("metadata"))
    audience = (
        _clean_text(metadata.get("audience"))
        or _clean_text(config.prompt_audience)
        or "عمومی"
    )
    tone = (
        _clean_text(metadata.get("tone"))
        or _clean_text(config.prompt_tone)
        or "دانشنامه‌ای"
    )
    return audience, tone


def _format_prompt_completion(record: dict[str, Any], config: ArticleCorpusConfig) -> str:
    title = _clean_text(record.get("title")) or "مقاله فارسی"
    body = _clean_text(record.get("body"))
    section_count = max(1, int(config.prompt_sections))
    sentences = _split_sentences(body)
    introduction = sentences[0] if sentences else body
    conclusion = sentences[-1] if len(sentences) > 1 else ""
    section_sentences = sentences[1:-1] if len(sentences) > 2 else sentences[1:]
    chunks = _chunk_items(section_sentences, section_count)
    audience, tone = _record_audience_tone(record, config)

    parts = [
        *_prompt_header(
            topic=title,
            audience=audience,
            tone=tone,
            sections=section_count,
        ),
        f"# {title}",
    ]
    if introduction:
        parts.append(f"مقدمه: {introduction}")
    for index, chunk in enumerate(chunks, start=1):
        section_body = " ".join(chunk).strip()
        parts.append(f"## بخش {index}: {title}")
        if section_body:
            parts.append(section_body)
    if conclusion:
        parts.append("## جمع‌بندی")
        parts.append(conclusion)
    return " | ".join(parts)


def _format_article_for_lm(
    record: dict[str, Any],
    config: ArticleCorpusConfig,
) -> str:
    fmt = _normalise_training_format(config.training_format)
    if fmt in {"prompt_completion", "wikipedia_prompt"}:
        return _format_prompt_completion(record, config)

    title = _clean_text(record.get("title"))
    summary = _clean_text(record.get("summary"))
    keywords = _coerce_keywords(record.get("keywords"))
    body = _clean_text(record.get("body"))
    audience, tone = _record_audience_tone(record, config)
    parts: list[str] = _prompt_header(
        topic=title or "مقاله فارسی",
        audience=audience,
        tone=tone,
        sections=config.prompt_sections,
    )
    if title:
        parts.append(f"عنوان: {title}")
    if summary:
        parts.append(f"خلاصه: {summary}")
    if keywords:
        parts.append("کلیدواژه‌ها: " + "، ".join(keywords))
    parts.append(f"متن مقاله: {body}")
    return " | ".join(parts)


def _read_txt_articles(path: Path) -> Iterable[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    raw_lines = raw.splitlines()
    heading_pattern = re.compile(r"^#(?!#)\s*(.+)$")
    if any(heading_pattern.match(line.strip()) for line in raw_lines):
        title = ""
        body_lines: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            heading = heading_pattern.match(stripped)
            if heading:
                if title or body_lines:
                    yield {
                        "title": title,
                        "body": " ".join(body_lines),
                        "metadata": {},
                    }
                title = heading.group(1).strip()
                body_lines = []
            else:
                body_lines.append(stripped)
        if title or body_lines:
            yield {"title": title, "body": " ".join(body_lines), "metadata": {}}
        return

    blocks = [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]
    if len(blocks) <= 1:
        blocks = [line.strip() for line in raw.splitlines() if line.strip()]
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        title = ""
        body_lines = lines
        if lines and lines[0].startswith("#"):
            title = lines[0].lstrip("#").strip()
            body_lines = lines[1:]
        yield {"title": title, "body": " ".join(body_lines), "metadata": {}}


def _read_jsonl_articles(
    path: Path,
    config: ArticleCorpusConfig,
) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            yield {
                "title": row.get(config.title_field, ""),
                "body": _body_from_row(row, config),
                "summary": row.get(config.summary_field, ""),
                "keywords": row.get(config.keywords_field, []),
                "metadata": {
                    **_coerce_metadata(row.get(config.metadata_field, {})),
                    "source_line": line_number,
                },
            }


def _read_tabular_articles(
    path: Path,
    config: ArticleCorpusConfig,
    *,
    delimiter: str,
) -> Iterable[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("article dataset must include a header row")
        if config.body_field not in reader.fieldnames:
            raise ValueError(
                f"article dataset is missing body field {config.body_field!r}"
            )
        for row_number, row in enumerate(reader, start=2):
            yield {
                "title": row.get(config.title_field, ""),
                "body": _body_from_row(row, config),
                "summary": row.get(config.summary_field, ""),
                "keywords": row.get(config.keywords_field, ""),
                "metadata": {
                    **_coerce_metadata(row.get(config.metadata_field, {})),
                    "source_row": row_number,
                },
            }


def _load_article_records(
    config: ArticleCorpusConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_path = Path(config.input_path)
    fmt = _detect_format(input_path, config.input_format)
    if fmt == "txt":
        raw_records = _read_txt_articles(input_path)
    elif fmt == "jsonl":
        raw_records = _read_jsonl_articles(input_path, config)
    elif fmt in {"csv", "tsv"}:
        raw_records = _read_tabular_articles(
            input_path,
            config,
            delimiter="\t" if fmt == "tsv" else ",",
        )
    else:
        raise ValueError("input_format must be one of: auto, txt, jsonl, csv, tsv")

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records, start=1):
        cleaned = {
            "title": _clean_text(record.get("title")),
            "body": _clean_text(record.get("body")),
            "summary": _clean_text(record.get("summary")),
            "keywords": _coerce_keywords(record.get("keywords")),
            "metadata": _coerce_metadata(record.get("metadata")),
        }
        if not cleaned["body"]:
            rejected.append({"index": index, "reason": "missing_body"})
            continue
        if len(cleaned["body"]) < config.min_body_chars:
            rejected.append(
                {
                    "index": index,
                    "reason": "body_too_short",
                    "body_chars": len(cleaned["body"]),
                }
            )
            continue
        accepted.append(cleaned)
    return accepted, rejected


def _split_records(
    records: Sequence[dict[str, Any]],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    indices = list(range(len(records)))
    if len(indices) <= 1 or validation_ratio == 0:
        return indices, []
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_count = max(
        1,
        min(len(indices) - 1, int(round(len(indices) * validation_ratio))),
    )
    validation = sorted(indices[:val_count])
    train = sorted(indices[val_count:])
    return train, validation


def _length_stats(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _char_profile(texts: Sequence[str]) -> dict[str, float | int]:
    joined = "".join(texts)
    counts = {
        "total": len(joined),
        "persian_arabic_letters": 0,
        "latin_letters": 0,
        "digits": 0,
        "zwnj": 0,
        "whitespace": 0,
        "other": 0,
    }
    for ch in joined:
        code = ord(ch)
        if ch == "\u200c":
            counts["zwnj"] += 1
        elif 0x0600 <= code <= 0x06FF:
            counts["persian_arabic_letters"] += 1
        elif ("a" <= ch.lower() <= "z"):
            counts["latin_letters"] += 1
        elif ch.isdigit():
            counts["digits"] += 1
        elif ch.isspace():
            counts["whitespace"] += 1
        else:
            counts["other"] += 1
    total = max(1, int(counts["total"]))
    return {
        **counts,
        "persian_arabic_ratio": counts["persian_arabic_letters"] / total,
        "latin_ratio": counts["latin_letters"] / total,
        "zwnj_per_1000_chars": counts["zwnj"] * 1000.0 / total,
    }


def _normalised_duplicate_key(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\u0600-\u06FF\u200c]+", "", text)
    return text


def _char_ngrams(text: str, n: int) -> set[str]:
    compact = _normalised_duplicate_key(text)
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _near_duplicate_pairs(
    texts: Sequence[str],
    *,
    ngram_size: int,
    threshold: float,
    max_pairs: int,
) -> dict[str, Any]:
    signatures = [_char_ngrams(text, ngram_size) for text in texts]
    pairs: list[dict[str, float | int]] = []
    checked = 0
    truncated = False
    for left in range(len(signatures)):
        if checked >= max_pairs:
            truncated = True
            break
        left_sig = signatures[left]
        if not left_sig:
            continue
        for right in range(left + 1, len(signatures)):
            if checked >= max_pairs:
                truncated = True
                break
            right_sig = signatures[right]
            checked += 1
            if not right_sig:
                continue
            union = len(left_sig | right_sig)
            if union == 0:
                continue
            score = len(left_sig & right_sig) / union
            if score >= threshold:
                pairs.append(
                    {
                        "left_index": left,
                        "right_index": right,
                        "jaccard": score,
                    }
                )
    return {
        "threshold": threshold,
        "ngram_size": ngram_size,
        "checked_pairs": checked,
        "max_pairs": max_pairs,
        "truncated": truncated,
        "count": len(pairs),
        "pairs": pairs[:100],
    }


def _metadata_profile(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for record in records:
        metadata = _coerce_metadata(record.get("metadata"))
        keys.update(metadata.keys())
        source = metadata.get("source") or metadata.get("url") or metadata.get("domain")
        if source:
            sources.update([str(source)])
    return {
        "metadata_key_counts": dict(keys.most_common()),
        "source_counts": dict(sources.most_common(25)),
        "records_with_metadata": sum(1 for record in records if record.get("metadata")),
    }


def _tokenizer_config(config: ArticleAuditConfig, tokenizer_type: str) -> PersianTokenizer:
    if tokenizer_type == "subword":
        tokenizer_type = "char_chunk"
    if tokenizer_type not in {"word", "char_chunk", "bpe", "unigram"}:
        raise ValueError(
            "tokenizer_types entries must be one of: word, char_chunk, bpe, unigram"
        )
    return PersianTokenizer(
        tokenizer_type=tokenizer_type,
        min_freq=1,
        max_vocab_size=config.tokenizer_max_vocab_size,
        keep_half_space=config.tokenizer_half_space == "preserve",
        morph_splitting=config.tokenizer_morph_splitting,
        compound_verb_mode=config.tokenizer_compound_verb_mode,
        bpe_num_merges=config.tokenizer_bpe_merges,
        unigram_num_pieces=config.tokenizer_unigram_num_pieces,
    )


def _tokenizer_eval_stats(
    tokenizer: PersianTokenizer,
    texts: Sequence[str],
) -> dict[str, float | int]:
    token_counts: list[int] = []
    unk_tokens = 0
    total_tokens = 0
    continuation_tokens = 0
    zwnj_tokens = 0
    for text in texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.append(len(tokens))
        unk_tokens += sum(1 for token_id in ids if token_id == tokenizer.unk_id)
        total_tokens += len(tokens)
        continuation_tokens += sum(1 for token in tokens if token.startswith("##"))
        zwnj_tokens += sum(1 for token in tokens if "\u200c" in token)
    char_count = sum(len(text) for text in texts)
    safe_tokens = max(1, total_tokens)
    return {
        "examples": len(texts),
        "vocab_size": tokenizer.vocab_size,
        "tokens": total_tokens,
        "tokens_min": min(token_counts) if token_counts else 0,
        "tokens_max": max(token_counts) if token_counts else 0,
        "tokens_mean": sum(token_counts) / len(token_counts) if token_counts else 0.0,
        "tokens_per_char": total_tokens / max(1, char_count),
        "unk_rate": unk_tokens / safe_tokens,
        "continuation_piece_rate": continuation_tokens / safe_tokens,
        "zwnj_token_rate": zwnj_tokens / safe_tokens,
    }


def _run_tokenizer_probe(
    *,
    tokenizer_type: str,
    train_texts: Sequence[str],
    validation_texts: Sequence[str],
    config: ArticleAuditConfig,
    output_dir: Path,
) -> dict[str, Any]:
    if config.tokenizer_probe_epochs <= 0:
        return {"enabled": False}
    probe_dir = output_dir / "tokenizer-probes" / tokenizer_type
    metrics = train_graph_lm(
        list(train_texts),
        training_config=LMTrainingConfig(
            output_dir=str(probe_dir),
            epochs=config.tokenizer_probe_epochs,
            batch_size=config.tokenizer_probe_batch_size,
            validation_ratio=0.0,
            block_size=config.tokenizer_probe_block_size,
            graph_relations=[],
            task_losses="next_token",
            masked_token_weight=0.0,
            edge_prediction_weight=0.0,
            neighbor_prediction_weight=0.0,
            node_relation_weight=0.0,
            graph_text_alignment_weight=0.0,
            sentence_graph_alignment_weight=0.0,
            text_augmentation=False,
            contrastive_weight=0.0,
            edge_dropout=0.0,
            node_dropout=0.0,
            subgraph_sampling_ratio=1.0,
            curriculum_learning=False,
            early_stopping_patience=max(1, config.tokenizer_probe_epochs),
            tokenizer_type=tokenizer_type,
            tokenizer_half_space=config.tokenizer_half_space,
            tokenizer_morph_splitting=config.tokenizer_morph_splitting,
            tokenizer_compound_verb_mode=config.tokenizer_compound_verb_mode,
            tokenizer_bpe_merges=config.tokenizer_bpe_merges,
            tokenizer_unigram_num_pieces=config.tokenizer_unigram_num_pieces,
            device=config.device,
            seed=config.seed,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=config.tokenizer_probe_block_size,
            d_model=config.tokenizer_probe_d_model,
            n_heads=config.tokenizer_probe_n_heads,
            n_layers=config.tokenizer_probe_n_layers,
            dim_feedforward=config.tokenizer_probe_d_model * 4,
            graph_encoder="none",
        ),
        graph_encoder="none",
        fusion="gated",
        validation_texts=list(validation_texts) or None,
    )
    return {
        "enabled": True,
        "checkpoint_dir": metrics.get("checkpoint_dir"),
        # Without held-out texts these metrics come from the training loss.
        "validation_available": bool(validation_texts),
        "best_next_token_loss": metrics.get("best_next_token_loss"),
        "best_perplexity": metrics.get("best_perplexity"),
        "best_epoch": metrics.get("best_epoch"),
    }


def _benchmark_tokenizers(
    records: Sequence[dict[str, Any]],
    *,
    config: ArticleAuditConfig,
    output_dir: Path,
) -> dict[str, Any]:
    train_indices, validation_indices = _split_records(
        records,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    corpus_config = _audit_corpus_config(config)
    all_texts = [_format_article_for_lm(record, corpus_config) for record in records]
    train_texts = [all_texts[index] for index in train_indices]
    validation_texts = [all_texts[index] for index in validation_indices]
    benchmarks: dict[str, Any] = {}
    for tokenizer_type in config.tokenizer_types:
        tokenizer = _tokenizer_config(config, tokenizer_type).fit(train_texts)
        benchmarks[tokenizer.tokenizer_type] = {
            "train": _tokenizer_eval_stats(tokenizer, train_texts),
            "validation": (
                _tokenizer_eval_stats(tokenizer, validation_texts)
                if validation_texts
                else None
            ),
            "native_training_probe": _run_tokenizer_probe(
                tokenizer_type=tokenizer.tokenizer_type,
                train_texts=train_texts,
                validation_texts=validation_texts,
                config=config,
                output_dir=output_dir,
            ),
        }
    return {
        "train_examples": len(train_texts),
        "validation_examples": len(validation_texts),
        "probe_epochs": config.tokenizer_probe_epochs,
        "tokenizers": benchmarks,
    }


def _audit_corpus_config(config: ArticleAuditConfig) -> ArticleCorpusConfig:
    return ArticleCorpusConfig(
        input_path=config.input_path,
        output_dir=config.output_dir,
        input_format=config.input_format,
        training_format=config.training_format,
        title_field=config.title_field,
        body_field=config.body_field,
        summary_field=config.summary_field,
        keywords_field=config.keywords_field,
        metadata_field=config.metadata_field,
        min_body_chars=config.min_body_chars,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )


def audit_article_corpus(config: ArticleAuditConfig) -> dict[str, Any]:
    """Audit a Persian article corpus without using external models or embeddings."""

    corpus_config = _audit_corpus_config(config)
    accepted, rejected = _load_article_records(corpus_config)
    if config.max_examples is not None:
        accepted = accepted[: max(0, config.max_examples)]
    if not accepted:
        raise ValueError("article dataset did not contain any usable records")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bodies = [record["body"] for record in accepted]
    titles = [record["title"] for record in accepted]
    duplicate_keys = Counter(_normalised_duplicate_key(body) for body in bodies)
    exact_duplicates = sum(count - 1 for count in duplicate_keys.values() if count > 1)

    report: dict[str, Any] = {
        "kind": "rakhshai_article_audit",
        "native_constraints": {
            "uses_external_pretrained_lm": False,
            "uses_distillation": False,
            "uses_pretrained_embeddings": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
        "input_path": str(Path(config.input_path)),
        "input_format": _detect_format(Path(config.input_path), config.input_format),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "body_chars": _length_stats([len(body) for body in bodies]),
        "title_chars": _length_stats([len(title) for title in titles if title]),
        "missing_title_count": sum(1 for title in titles if not title),
        "summary_present_count": sum(1 for record in accepted if record.get("summary")),
        "keywords_present_count": sum(1 for record in accepted if record.get("keywords")),
        "char_profile": _char_profile([*titles, *bodies]),
        "persian_surface": {
            "zwnj_count": sum(body.count("\u200c") for body in bodies),
            "space_after_mi_prefix_count": sum(
                len(re.findall(r"(?:^|\s)(?:می|نمی)\s+[\u0600-\u06FF]+", body))
                for body in bodies
            ),
            "ascii_question_mark_count": sum(body.count("?") for body in bodies),
            "persian_question_mark_count": sum(body.count("؟") for body in bodies),
        },
        "duplicates": {
            "exact_duplicate_bodies": exact_duplicates,
            "near_duplicates": _near_duplicate_pairs(
                bodies,
                ngram_size=config.near_duplicate_ngram_size,
                threshold=config.near_duplicate_threshold,
                max_pairs=config.near_duplicate_max_pairs,
            ),
        },
        "metadata": _metadata_profile(accepted),
        "tokenizer_benchmark": _benchmark_tokenizers(
            accepted,
            config=config,
            output_dir=output_dir,
        ),
        "rejection_reasons": dict(
            Counter(str(row.get("reason", "unknown")) for row in rejected)
        ),
        "config": asdict(config),
    }
    report_path = output_dir / "article_audit.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path)
    return report


def prepare_article_corpus(config: ArticleCorpusConfig) -> dict[str, Any]:
    """Convert raw article data into corpus files and a manifest."""

    config = ArticleCorpusConfig(
        **{
            **asdict(config),
            "training_format": _normalise_training_format(config.training_format),
        }
    )
    accepted, rejected = _load_article_records(config)
    if not accepted:
        raise ValueError("article dataset did not contain any usable records")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_lines = [_format_article_for_lm(record, config) for record in accepted]
    train_indices, validation_indices = _split_records(
        accepted,
        validation_ratio=config.validation_ratio,
        seed=config.seed,
    )
    train_lines = [corpus_lines[index] for index in train_indices]
    validation_lines = [corpus_lines[index] for index in validation_indices]

    corpus_path = output_dir / "corpus.txt"
    train_path = output_dir / "train.txt"
    validation_path = output_dir / "validation.txt"
    rejected_path = output_dir / "rejected_records.jsonl"
    manifest_path = output_dir / "manifest.json"
    prepared_path = output_dir / "prepared_articles.jsonl"

    corpus_path.write_text("\n".join(corpus_lines) + "\n", encoding="utf-8")
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    validation_path.write_text(
        ("\n".join(validation_lines) + "\n") if validation_lines else "",
        encoding="utf-8",
    )
    with rejected_path.open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with prepared_path.open("w", encoding="utf-8") as f:
        for record, corpus_line in zip(accepted, corpus_lines):
            payload = {**record, "corpus_text": corpus_line}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    body_lengths = [len(record["body"]) for record in accepted]
    corpus_lengths = [len(line) for line in corpus_lines]
    manifest: dict[str, Any] = {
        "input_path": str(Path(config.input_path)),
        "input_format": _detect_format(Path(config.input_path), config.input_format),
        "output_dir": str(output_dir),
        "corpus_path": str(corpus_path),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "prepared_articles_path": str(prepared_path),
        "rejected_records_path": str(rejected_path),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "splits": {
            "train": len(train_lines),
            "validation": len(validation_lines),
        },
        "body_chars": _length_stats(body_lengths),
        "corpus_chars": _length_stats(corpus_lengths),
        "config": asdict(config),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _load_corpus(path: str | Path, *, allow_empty: bool = False) -> list[str]:
    corpus_path = Path(path)
    with corpus_path.open(encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    if not texts and not allow_empty:
        raise ValueError("article corpus is empty")
    return texts


def _resolve_training_corpora(
    corpus_path: str | Path,
) -> tuple[list[str], list[str], list[str] | None, dict[str, Any]]:
    source_path = Path(corpus_path)
    full_corpus = _load_corpus(source_path)
    train_path = source_path.with_name("train.txt")
    validation_path = source_path.with_name("validation.txt")

    validation_fallback = None
    if (
        source_path.name == "corpus.txt"
        and train_path.exists()
        and validation_path.exists()
    ):
        train_corpus = _load_corpus(train_path)
        validation_corpus = _load_corpus(validation_path, allow_empty=True)
        if validation_corpus:
            split = {
                "uses_prepared_splits": True,
                "source_corpus_path": str(source_path),
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "corpus_examples": len(full_corpus),
                "train_examples": len(train_corpus),
                "validation_examples": len(validation_corpus),
                "validation_fallback": None,
            }
            return full_corpus, train_corpus, validation_corpus, split
        # An empty prepared validation.txt would silently disable validation
        # and let early stopping monitor the training loss; fall back to the
        # trainer's validation_ratio split over the full corpus instead.
        validation_fallback = "empty_prepared_validation"

    split = {
        "uses_prepared_splits": False,
        "source_corpus_path": str(source_path),
        "train_path": None,
        "validation_path": None,
        "corpus_examples": len(full_corpus),
        "train_examples": None,
        "validation_examples": None,
        "validation_fallback": validation_fallback,
    }
    return full_corpus, full_corpus, None, split


def _training_config(config: ArticleTrainingConfig) -> LMTrainingConfig:
    return LMTrainingConfig(
        output_dir=config.output_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        lr_scheduler=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        validation_ratio=config.validation_ratio,
        block_size=config.block_size,
        stride=config.stride,
        min_freq=config.min_freq,
        max_vocab_size=config.max_vocab_size,
        graph_window_size=config.graph_window_size,
        graph_min_count=config.graph_min_count,
        graph_weighting=config.graph_weighting,
        graph_min_edge_weight=config.graph_min_edge_weight,
        graph_top_k=config.graph_top_k,
        graph_directed=config.graph_directed,
        graph_scope=config.graph_scope,
        context_node_type=config.context_node_type,
        graph_relations=config.graph_relations,
        relation_weights=config.relation_weights,
        semantic_similarity_threshold=config.semantic_similarity_threshold,
        semantic_top_k=config.semantic_top_k,
        semantic_method=config.semantic_method,
        linguistic_backend=config.linguistic_backend,
        topic_top_k=config.topic_top_k,
        graph_relation_mode=config.graph_relation_mode,
        graph_pooling=config.graph_pooling,
        graph_node_importance=config.graph_node_importance,
        graph_node_type_embedding=config.graph_node_type_embedding,
        fusion_levels=config.fusion_levels,
        graph_fusion_scale=config.graph_fusion_scale,
        graph_fusion_dropout=config.graph_fusion_dropout,
        task_losses=config.task_losses,
        next_token_weight=config.next_token_weight,
        masked_token_weight=config.masked_token_weight,
        edge_prediction_weight=config.edge_prediction_weight,
        neighbor_prediction_weight=config.neighbor_prediction_weight,
        node_relation_weight=config.node_relation_weight,
        graph_text_alignment_weight=config.graph_text_alignment_weight,
        sentence_graph_alignment_weight=config.sentence_graph_alignment_weight,
        mask_probability=config.mask_probability,
        negative_samples=config.negative_samples,
        text_augmentation=config.text_augmentation,
        augmentation_ratio=config.augmentation_ratio,
        token_dropout=config.token_dropout,
        punctuation_dropout=config.punctuation_dropout,
        node_dropout=config.node_dropout,
        edge_dropout=config.edge_dropout,
        subgraph_sampling_ratio=config.subgraph_sampling_ratio,
        contrastive_weight=config.contrastive_weight,
        curriculum_learning=config.curriculum_learning,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        checkpoint_metric=config.checkpoint_metric,
        max_grad_norm=config.max_grad_norm,
        dynamic_graph=config.dynamic_graph,
        graph_build_batch_size=config.graph_build_batch_size,
        graph_cache_dir=config.graph_cache_dir,
        reuse_graph_cache=config.reuse_graph_cache,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        amp=config.amp,
        resume_from=config.resume_from,
        tokenizer_type=config.tokenizer_type,
        tokenizer_half_space=config.tokenizer_half_space,
        tokenizer_morph_splitting=config.tokenizer_morph_splitting,
        tokenizer_compound_verb_mode=config.tokenizer_compound_verb_mode,
        tokenizer_bpe_merges=config.tokenizer_bpe_merges,
        tokenizer_unigram_num_pieces=config.tokenizer_unigram_num_pieces,
        device=config.device,
        seed=config.seed,
    )


def _model_config(config: ArticleTrainingConfig) -> GraphLMConfig:
    return GraphLMConfig(
        vocab_size=1,
        max_seq_len=config.block_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        graph_encoder=config.graph_encoder,
        graph_hidden_dim=config.graph_hidden_dim,
        graph_heads=config.graph_heads,
        graph_relation_mode=config.graph_relation_mode,
        graph_pooling=config.graph_pooling,
        graph_node_importance=config.graph_node_importance,
        graph_node_type_embedding=config.graph_node_type_embedding,
        fusion=config.fusion,
        fusion_layers=config.fusion_layers,
        fusion_levels=config.fusion_levels,
        graph_fusion_scale=config.graph_fusion_scale,
        graph_fusion_dropout=config.graph_fusion_dropout,
    )


def _zero_gate_report(config: ArticleTrainingConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    fusion = config.fusion.lower()
    supports_zero_gate = fusion in {"gated", "context_gated", "contextual"}
    fusion_stats = metrics.get("fusion_stats")
    alpha_metrics: dict[str, float] = {}
    if isinstance(fusion_stats, dict):
        for key, value in fusion_stats.items():
            if key.endswith("_alpha_tanh"):
                alpha_metrics[key] = float(value)
    return {
        "fusion": config.fusion,
        "supports_zero_init_gate": supports_zero_gate,
        "initial_graph_update_is_zero": supports_zero_gate,
        "alpha_metrics": alpha_metrics,
        "graph_update_active_after_training": any(
            abs(value) > 1e-4 for value in alpha_metrics.values()
        ),
        "note": (
            "gated/context_gated fusion starts with tanh(alpha)=0; "
            "graph updates become active only if training moves alpha away from zero."
            if supports_zero_gate
            else "additive fusion does not provide the zero-init graph gate guarantee."
        ),
    }


def train_article_llm(config: ArticleTrainingConfig) -> dict[str, Any]:
    """Train a native Persian article-writing Graph-LM checkpoint."""

    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    resolved_device = (
        "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if resolved_device != config.device:
        config = ArticleTrainingConfig(**{**asdict(config), "device": resolved_device})

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _, training_corpus, validation_corpus, split = _resolve_training_corpora(
        config.corpus_path
    )
    metrics = train_graph_lm(
        training_corpus,
        training_config=_training_config(config),
        model_config=_model_config(config),
        graph_encoder=config.graph_encoder,
        fusion=config.fusion,
        validation_texts=validation_corpus,
    )
    source_corpus_path = Path(config.corpus_path).resolve()
    output_corpus_path = (output_dir / "corpus.txt").resolve()
    if source_corpus_path != output_corpus_path:
        shutil.copyfile(config.corpus_path, output_dir / "corpus.txt")
    metrics["article_data_split"] = split
    metrics["zero_gate_report"] = _zero_gate_report(config, metrics)
    article_config = {
        "kind": "rakhshai_article_llm",
        "version": 1,
        "native_training": {
            "uses_external_pretrained_lm": False,
            "uses_distillation": False,
            "uses_pretrained_embeddings": False,
            "uses_llm_synthetic_data": False,
        },
        "training_config": asdict(config),
        "data_split": split,
        "zero_gate_report": metrics["zero_gate_report"],
        "checkpoint_dir": str(output_dir),
        "metrics": {
            "best_perplexity": metrics.get("best_perplexity"),
            "best_next_token_loss": metrics.get("best_next_token_loss"),
            "best_epoch": metrics.get("best_epoch"),
        },
    }
    with (output_dir / "article_llm_config.json").open("w", encoding="utf-8") as f:
        json.dump(article_config, f, ensure_ascii=False, indent=2)
    metrics["article_llm_config_path"] = str(output_dir / "article_llm_config.json")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def _default_relation_groups() -> dict[str, Sequence[str]]:
    return {
        "cooccurrence": ["cooccurrence"],
        "pmi": ["pmi"],
        "dependency": ["dependency"],
        "topic_document": ["topic_document"],
    }


def _variant_training_config(
    base: ArticleTrainingConfig,
    *,
    output_dir: Path,
    graph_encoder: str,
    graph_scope: str,
    graph_relations: Sequence[str] | None,
) -> ArticleTrainingConfig:
    values = asdict(base)
    values.update(
        {
            "output_dir": str(output_dir),
            "graph_encoder": graph_encoder,
            "graph_scope": graph_scope,
            "graph_relations": list(graph_relations or []),
        }
    )
    if graph_encoder == "none":
        values.update(
            {
                "graph_relations": [],
                "graph_node_importance": False,
                "graph_pooling": "none",
                "graph_fusion_dropout": 0.0,
                "contrastive_weight": 0.0,
                "edge_prediction_weight": 0.0,
                "neighbor_prediction_weight": 0.0,
                "node_relation_weight": 0.0,
                "graph_text_alignment_weight": 0.0,
                "sentence_graph_alignment_weight": 0.0,
                "edge_dropout": 0.0,
                "node_dropout": 0.0,
                "subgraph_sampling_ratio": 1.0,
            }
        )
    return ArticleTrainingConfig(**values)


def _article_probe_stats(article: PersianArticle) -> dict[str, Any]:
    return {
        "title": article.title,
        "markdown_chars": len(article.full_markdown),
        "sections": len(article.sections),
        "empty_sections": sum(
            1 for section in article.sections if not section.get("body", "").strip()
        ),
    }


def _ablation_generation_probes(
    model_dir: Path,
    *,
    config: ArticleAblationConfig,
    graph_encoder: str,
) -> dict[str, Any]:
    if not config.probe_topic:
        return {"enabled": False}
    probes: dict[str, Any] = {}
    for enabled in (False, True):
        if enabled and graph_encoder == "none":
            continue
        article = generate_persian_article(
            ArticleGenerationConfig(
                model_dir=str(model_dir),
                topic=config.probe_topic,
                sections=config.probe_sections,
                min_new_tokens=0,
                max_new_tokens=config.probe_max_new_tokens,
                graph_memory=enabled,
                device=config.training_config.device,
            )
        )
        probes["graph_memory_on" if enabled else "graph_memory_off"] = (
            _article_probe_stats(article)
        )
    return {"enabled": True, "probes": probes}


def _summarise_ablation_variant(
    *,
    name: str,
    variant_config: ArticleTrainingConfig,
    metrics: dict[str, Any],
    generation_probes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "checkpoint_dir": variant_config.output_dir,
        "graph_encoder": variant_config.graph_encoder,
        "graph_scope": variant_config.graph_scope,
        "graph_relations": list(variant_config.graph_relations or []),
        "best_perplexity": metrics.get("best_perplexity"),
        "best_next_token_loss": metrics.get("best_next_token_loss"),
        "best_epoch": metrics.get("best_epoch"),
        "zero_gate_report": metrics.get("zero_gate_report"),
        "fusion_stats": metrics.get("fusion_stats", {}),
        "generation_probes": generation_probes,
    }


def _ablation_variants(
    config: ArticleAblationConfig,
) -> list[tuple[str, str, str, Sequence[str] | None]]:
    graph_encoders = list(dict.fromkeys(config.graph_encoders))
    graph_scopes = list(dict.fromkeys(config.graph_scopes)) or ["document"]
    relation_groups = config.relation_groups or _default_relation_groups()
    base_relations = config.training_config.graph_relations
    variants: list[tuple[str, str, str, Sequence[str] | None]] = []

    if "none" in graph_encoders:
        variants.append(("no_graph", "none", "document", []))

    graph_enabled = [encoder for encoder in graph_encoders if encoder != "none"]
    for encoder in graph_enabled:
        for scope in graph_scopes:
            variants.append((f"{encoder}_{scope}_all_relations", encoder, scope, base_relations))

    if graph_enabled:
        encoder = graph_enabled[0]
        scope = graph_scopes[0]
        for group_name, relations in relation_groups.items():
            variants.append((f"{encoder}_{scope}_{group_name}", encoder, scope, relations))
    return variants


def run_article_ablation(config: ArticleAblationConfig) -> dict[str, Any]:
    """Run native graph ablations around the article training workflow."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, graph_encoder, graph_scope, relations in _ablation_variants(config):
        variant_dir = output_dir / name
        variant_config = _variant_training_config(
            config.training_config,
            output_dir=variant_dir,
            graph_encoder=graph_encoder,
            graph_scope=graph_scope,
            graph_relations=relations,
        )
        metrics = train_article_llm(variant_config)
        generation_probes = _ablation_generation_probes(
            variant_dir,
            config=config,
            graph_encoder=graph_encoder,
        )
        rows.append(
            _summarise_ablation_variant(
                name=name,
                variant_config=variant_config,
                metrics=metrics,
                generation_probes=generation_probes,
            )
        )

    report: dict[str, Any] = {
        "kind": "rakhshai_article_ablation",
        "native_constraints": {
            "uses_external_pretrained_lm": False,
            "uses_distillation": False,
            "uses_pretrained_embeddings": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
        "output_dir": str(output_dir),
        "variants": rows,
        "config": {
            **asdict(config),
            "training_config": asdict(config.training_config),
        },
    }
    report_path = output_dir / "article_ablation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path)
    return report


def _article_prompt(config: ArticleGenerationConfig) -> str:
    header = _prompt_header(
        topic=config.topic,
        audience=config.audience,
        tone=config.tone,
        sections=config.sections,
    )
    # Prepared corpora join segments with " | ", so the prompt must too.
    return " | ".join(header) + " | "


def _strip_prompt(generated: str, prompt: str) -> str:
    text = generated.strip()
    stripped_prompt = prompt.strip()
    if text.startswith(stripped_prompt):
        return text[len(stripped_prompt) :].strip().lstrip("|").strip()
    # "مقاله:" must be its own segment; a plain substring search would match
    # inside "موضوع مقاله:" or "متن مقاله:".
    match = re.search(r"(?:\||\n)\s*مقاله:\s*\|?\s*", text)
    if match:
        return text[match.end() :].strip()
    return text


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [
        part.strip() for part in re.split(r"\n+|(?<=[.؟!])\s+", text)
    ]
    return [paragraph for paragraph in paragraphs if paragraph]


def _chunk_paragraphs(paragraphs: Sequence[str], count: int) -> list[list[str]]:
    if count <= 0:
        return []
    if not paragraphs:
        return [[] for _ in range(count)]
    chunk_size = max(1, math.ceil(len(paragraphs) / count))
    chunks = [
        list(paragraphs[i : i + chunk_size])
        for i in range(0, len(paragraphs), chunk_size)
    ]
    while len(chunks) < count:
        chunks.append([])
    return chunks[:count]


def _article_field_key(label: str) -> str | None:
    normalised = label.replace("\u200c", " ").replace("ي", "ی").strip()
    if normalised == "عنوان":
        return "title"
    if normalised in {"خلاصه", "چکیده"}:
        return "introduction"
    if normalised == "مقدمه":
        return "introduction"
    if normalised in {"متن مقاله", "متن"}:
        return "body"
    if normalised in {"جمع بندی", "نتیجه", "نتیجه گیری"}:
        return "conclusion"
    if normalised.startswith("کلیدواژه"):
        return "keywords"
    return None


_ARTICLE_FIELD_RE = re.compile(
    r"^(عنوان|خلاصه|چکیده|مقدمه|متن مقاله|متن|"
    r"جمع‌بندی|جمع بندی|نتیجه|نتیجه‌گیری|نتیجه گیری|"
    r"کلیدواژه‌ها|کلیدواژه ها)\s*[:：]\s*(.*)$"
)


def _strip_article_field_label(text: str) -> str:
    match = _ARTICLE_FIELD_RE.match(text.strip())
    if match and _article_field_key(match.group(1)) != "keywords":
        return match.group(2).strip()
    return text.strip()


def _extract_article_fields(clean: str) -> tuple[dict[str, str], str]:
    fields: dict[str, list[str]] = {}
    remainder: list[str] = []
    current_key: str | None = None
    multiline_keys = {"body", "introduction", "conclusion"}
    for line in (part.strip() for part in clean.splitlines()):
        if not line:
            continue
        match = _ARTICLE_FIELD_RE.match(line)
        if match:
            key = _article_field_key(match.group(1))
            value = match.group(2).strip()
            if key is None or key == "keywords":
                current_key = None
                continue
            if value:
                fields.setdefault(key, []).append(value)
            current_key = key if key in multiline_keys else None
            continue
        if current_key in multiline_keys:
            fields.setdefault(current_key, []).append(line)
        else:
            remainder.append(line)
    return {
        key: "\n".join(value for value in values if value).strip()
        for key, values in fields.items()
        if any(values)
    }, "\n".join(remainder).strip()


def _parse_generated_article(
    raw_text: str,
    config: ArticleGenerationConfig,
) -> PersianArticle:
    # Prepared corpora use " | " as the segment separator; map it back to
    # newlines so markdown heading detection works on generated text.
    clean = re.sub(r"\s*\|\s*", "\n", raw_text).strip()
    labelled_fields, remainder = _extract_article_fields(clean)
    default_title = f"مقاله درباره {config.topic}"
    title = _strip_article_field_label(labelled_fields.get("title", "")) or default_title
    introduction = _strip_article_field_label(
        labelled_fields.get("introduction", "")
    )
    conclusion = _strip_article_field_label(labelled_fields.get("conclusion", ""))
    explicit_introduction = bool(introduction)
    explicit_conclusion = bool(conclusion)
    sections: list[dict[str, str]] = []
    parse_source = "\n".join(
        part for part in (labelled_fields.get("body", ""), remainder) if part
    ).strip()
    if not parse_source:
        parse_source = clean

    heading_match = re.search(r"^#\s+(.+)$", parse_source, flags=re.MULTILINE)
    if heading_match:
        if "title" not in labelled_fields:
            title = _strip_article_field_label(heading_match.group(1))
        parse_source = parse_source.replace(heading_match.group(0), "", 1).strip()
    elif "title" not in labelled_fields:
        title_line = next(
            (line.strip() for line in parse_source.splitlines() if line.strip()),
            "",
        )
        if (
            title_line
            and len(title_line) <= 90
            and not title_line.endswith((".", "؟", "!"))
        ):
            title = _strip_article_field_label(title_line.lstrip("#"))
            parse_source = parse_source.replace(title_line, "", 1).strip()

    markdown_sections = re.split(r"^##\s+", parse_source, flags=re.MULTILINE)
    intro_source = markdown_sections[0].strip() if markdown_sections else parse_source
    if len(markdown_sections) > 1:
        introduction = introduction or _strip_article_field_label(intro_source)
        for block in markdown_sections[1:]:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            heading = lines[0]
            body = _strip_article_field_label("\n".join(lines[1:]))
            if "جمع" in heading or "نتیجه" in heading:
                conclusion = conclusion or body
            else:
                sections.append({"heading": heading, "body": body})

    if not sections:
        paragraphs = [
            _strip_article_field_label(paragraph)
            for paragraph in _split_paragraphs(parse_source)
        ]
        if paragraphs:
            introduction = introduction or paragraphs[0]
        body_start = 0 if explicit_introduction else 1 if len(paragraphs) > 1 else 0
        body_paragraphs = paragraphs[body_start:]
        if not explicit_conclusion and len(paragraphs) > 1:
            conclusion = paragraphs[-1]
            if body_paragraphs and body_paragraphs[-1] == conclusion:
                body_paragraphs = body_paragraphs[:-1]
        for index, chunk in enumerate(
            _chunk_paragraphs(body_paragraphs, config.sections),
            start=1,
        ):
            sections.append(
                {
                    "heading": f"بخش {index}: {config.topic}",
                    "body": " ".join(chunk).strip(),
                }
            )

    while len(sections) < max(1, config.sections):
        sections.append(
            {
                "heading": f"بخش {len(sections) + 1}: {config.topic}",
                "body": "",
            }
        )
    sections = sections[: max(1, config.sections)]
    conclusion = _strip_article_field_label(conclusion)
    introduction = _strip_article_field_label(introduction)

    full_markdown = _render_markdown(title, introduction, sections, conclusion)
    return PersianArticle(
        title=title,
        introduction=introduction,
        sections=sections,
        conclusion=conclusion,
        full_markdown=full_markdown,
        metadata={
            "topic": config.topic,
            "audience": config.audience,
            "tone": config.tone,
            "model_dir": config.model_dir,
            "raw_generation": raw_text,
        },
    )


def _render_markdown(
    title: str,
    introduction: str,
    sections: Sequence[dict[str, str]],
    conclusion: str,
) -> str:
    parts = [f"# {title}"]
    if introduction:
        parts.append(introduction)
    for section in sections:
        heading = section.get("heading", "بخش")
        body = section.get("body", "")
        parts.append(f"## {heading}")
        if body:
            parts.append(body)
    if conclusion:
        parts.append("## جمع‌بندی")
        parts.append(conclusion)
    return "\n\n".join(parts).strip() + "\n"


def _load_generation_artifacts(config: ArticleGenerationConfig):
    device = torch.device(
        "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model, tokenizer, generation_config, graph_config = GraphCausalLM.from_pretrained(
        config.model_dir,
        map_location=device,
    )
    generation_config.min_new_tokens = config.min_new_tokens
    generation_config.max_new_tokens = config.max_new_tokens
    generation_config.temperature = config.temperature
    generation_config.top_k = config.top_k
    generation_config.repetition_penalty = config.repetition_penalty
    model.to(device)

    graph_data, token_node_ids = GraphCausalLM.load_graph_artifacts(
        config.model_dir,
        map_location=device,
    )
    graph_memory = None
    graph_memory_config = GraphMemoryConfig(
        enabled=config.graph_memory,
        top_k_nodes=config.graph_memory_top_k,
        depth=config.graph_memory_depth,
        max_edges=config.graph_memory_max_edges,
        min_score=config.graph_memory_min_score,
        relation_weights=config.graph_memory_relation_weights,
    )
    dynamic_graph_config = None
    model_dir = Path(config.model_dir)
    corpus_path = model_dir / "corpus.txt"

    if model.config.graph_encoder != "none" and config.graph_memory:
        graph_memory, saved_memory_config = GraphMemoryArtifact.load(
            model_dir,
            map_location=device,
        )
        if graph_memory is None and corpus_path.exists():
            graph_memory = GraphMemoryArtifact.from_corpus(
                _load_corpus(corpus_path),
                tokenizer,
                graph_config,
            )
        elif (
            graph_memory is None
            and graph_data is not None
            and token_node_ids is not None
        ):
            graph_memory = GraphMemoryArtifact.from_pyg_data(
                graph_data,
                token_node_ids,
                tokenizer,
                graph_config,
            )
        graph_memory_config.enabled = (
            saved_memory_config.enabled and config.graph_memory
        )

    if (
        model.config.graph_encoder != "none"
        and bool(graph_config.get("dynamic_graph", False))
        and graph_memory is None
    ):
        graph_data = None
        token_node_ids = None
        dynamic_graph_config = graph_config
    elif graph_data is not None and token_node_ids is not None:
        graph_data = graph_data.to(device)
        token_node_ids = token_node_ids.to(device)
    elif model.config.graph_encoder != "none" and corpus_path.exists():
        graph = build_graph_lm_graph(
            _load_corpus(corpus_path),
            tokenizer,
            window_size=int(graph_config.get("window_size", 4)),
            min_count=int(graph_config.get("min_count", 1)),
            weighting=str(graph_config.get("weighting", "distance")),
            min_edge_weight=float(graph_config.get("min_edge_weight", 0.0)),
            top_k=graph_config.get("top_k"),  # type: ignore[arg-type]
            directed=bool(graph_config.get("directed", False)),
            graph_scope=str(graph_config.get("graph_scope", "document")),
            context_node_type=str(graph_config.get("context_node_type", "none")),
            graph_relations=graph_config.get(  # type: ignore[arg-type]
                "enabled_relations"
            ),
            relation_weights=graph_config.get(  # type: ignore[arg-type]
                "relation_weights"
            ),
            semantic_similarity_threshold=float(
                graph_config.get("semantic_similarity_threshold", 0.6)
            ),
            semantic_top_k=graph_config.get("semantic_top_k"),  # type: ignore[arg-type]
            semantic_method=str(graph_config.get("semantic_method", "distributional")),
            linguistic_backend=str(graph_config.get("linguistic_backend", "auto")),
            topic_top_k=int(graph_config.get("topic_top_k", 8)),
        )
        graph_data = graph.to_pyg_data().to(device)
        token_node_ids = graph.token_node_ids(tokenizer.vocab_size).to(device)
    else:
        graph_data = None
        token_node_ids = None

    return (
        model,
        tokenizer,
        generation_config,
        graph_data,
        token_node_ids,
        graph_memory,
        graph_memory_config,
        dynamic_graph_config,
    )


def generate_persian_article(config: ArticleGenerationConfig) -> PersianArticle:
    """Generate a structured Persian article from a trained article checkpoint."""

    (
        model,
        tokenizer,
        generation_config,
        graph_data,
        token_node_ids,
        graph_memory,
        graph_memory_config,
        dynamic_graph_config,
    ) = _load_generation_artifacts(config)
    prompt = _article_prompt(config)
    generated = model.generate(
        prompt,
        tokenizer,
        graph_data=graph_data,
        token_node_ids=token_node_ids,
        graph_memory=graph_memory,
        graph_memory_config=graph_memory_config,
        generation_config=generation_config,
        dynamic_graph_config=dynamic_graph_config,
        max_new_tokens=config.max_new_tokens,
    )
    continuation = _strip_prompt(generated, prompt)
    return _parse_generated_article(continuation, config)

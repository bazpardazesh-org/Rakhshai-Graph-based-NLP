"""Native corpus building utilities for independent Persian LM training."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


_PERSIAN_RE = re.compile(r"[\u0600-\u06ff]")
_WORD_RE = re.compile(r"[\w\u0600-\u06ff]+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


@dataclass
class CorpusBuildConfig:
    """Configuration for local-only corpus cleaning and splitting."""

    input_paths: Sequence[str]
    output_dir: str
    input_format: str = "auto"
    text_fields: Sequence[str] = field(default_factory=lambda: ["text", "body"])
    source_id: str | None = None
    min_chars: int = 20
    min_persian_ratio: float = 0.35
    near_duplicate_threshold: float = 0.92
    validation_ratio: float = 0.1
    test_ratio: float = 0.05
    seed: int = 0
    eval_paths: Sequence[str] = field(default_factory=list)


def _clean_text(text: object) -> str:
    raw = "" if text is None else str(text)
    raw = raw.replace("\ufeff", " ").replace("\u200b", "")
    return _SPACE_RE.sub(" ", raw).strip()


def _normalise_for_hash(text: str) -> str:
    text = text.lower().replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")
    return _SPACE_RE.sub(" ", text).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested.lower()
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    return "txt"


def _extract_text(row: dict[str, Any], text_fields: Sequence[str]) -> str:
    parts = [_clean_text(row.get(field, "")) for field in text_fields]
    return "\n".join(part for part in parts if part).strip()


def _iter_records(path: Path, fmt: str, text_fields: Sequence[str]) -> Iterable[dict[str, Any]]:
    if fmt == "txt":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                yield {"text": line.strip(), "_line": line_no}
        return
    if fmt == "jsonl":
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = {"text": line, "_decode_error": True}
                if not isinstance(row, dict):
                    row = {"text": row}
                row["_line"] = line_no
                row["text"] = _extract_text(row, text_fields) or _clean_text(row.get("text"))
                yield row
        return
    if fmt == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                row = {"text": row}
            row["_line"] = idx + 1
            row["text"] = _extract_text(row, text_fields) or _clean_text(row.get("text"))
            yield row
        return
    if fmt in {"csv", "tsv"}:
        delimiter = "\t" if fmt == "tsv" else ","
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for line_no, row in enumerate(reader, start=2):
                row = dict(row)
                row["_line"] = line_no
                row["text"] = _extract_text(row, text_fields)
                yield row
        return
    raise ValueError("input_format must be one of: auto, txt, json, jsonl, csv, tsv")


def persian_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha() or _PERSIAN_RE.match(ch)]
    if not letters:
        return 0.0
    persian = sum(1 for ch in letters if _PERSIAN_RE.match(ch))
    return persian / len(letters)


def _simhash(text: str, bits: int = 64) -> int:
    vector = [0] * bits
    tokens = _WORD_RE.findall(_normalise_for_hash(text))
    for token in tokens:
        digest = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(bits):
            vector[bit] += 1 if digest & (1 << bit) else -1
    value = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            value |= 1 << bit
    return value


def _simhash_similarity(left: int, right: int, bits: int = 64) -> float:
    distance = (left ^ right).bit_count()
    return 1.0 - (distance / bits)


def _split_records(records: list[dict[str, Any]], config: CorpusBuildConfig) -> dict[str, list[dict[str, Any]]]:
    import random

    shuffled = list(records)
    random.Random(config.seed).shuffle(shuffled)
    total = len(shuffled)
    test_size = int(round(total * config.test_ratio)) if total > 2 else 0
    val_size = int(round(total * config.validation_ratio)) if total > 1 else 0
    if total > 2 and config.test_ratio > 0:
        test_size = max(1, min(test_size, total - 2))
    if total - test_size > 1 and config.validation_ratio > 0:
        val_size = max(1, min(val_size, total - test_size - 1))
    test = shuffled[:test_size]
    validation = shuffled[test_size : test_size + val_size]
    train = shuffled[test_size + val_size :]
    return {"train": train, "validation": validation, "test": test}


def _load_eval_hashes(paths: Sequence[str]) -> set[str]:
    hashes: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = _clean_text(line)
            if text:
                hashes.add(_sha256_text(_normalise_for_hash(text)))
    return hashes


def build_lm_corpus(config: CorpusBuildConfig) -> dict[str, Any]:
    """Build a cleaned native corpus and quality reports.

    The implementation is deliberately local-only: it performs no network calls
    and never loads external models.
    """

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_hashes = _load_eval_hashes(config.eval_paths)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    simhashes: list[tuple[int, str]] = []
    source_rows: list[dict[str, Any]] = []

    for path_str in config.input_paths:
        path = Path(path_str)
        fmt = _detect_format(path, config.input_format)
        source_id = config.source_id or path.stem
        source_total = 0
        source_accepted = 0
        source_rejected = 0
        for row in _iter_records(path, fmt, config.text_fields):
            source_total += 1
            text = _clean_text(row.get("text", ""))
            norm = _normalise_for_hash(text)
            reason = ""
            if not text:
                reason = "empty"
            elif len(text) < config.min_chars:
                reason = "too_short"
            elif persian_ratio(text) < config.min_persian_ratio:
                reason = "low_persian_ratio"
            text_hash = _sha256_text(norm) if norm else ""
            if not reason and text_hash in exact_seen:
                reason = "exact_duplicate"
            current_simhash = _simhash(text) if not reason else 0
            if not reason:
                for previous, _previous_hash in simhashes:
                    if _simhash_similarity(current_simhash, previous) >= config.near_duplicate_threshold:
                        reason = "near_duplicate"
                        break
            if not reason and text_hash in eval_hashes:
                reason = "eval_contamination"
            record = {
                "text": text,
                "text_hash": text_hash,
                "source_id": source_id,
                "path": str(path),
                "line": row.get("_line"),
                "chars": len(text),
                "persian_ratio": persian_ratio(text),
            }
            if reason:
                source_rejected += 1
                rejected.append({**record, "reason": reason})
                continue
            source_accepted += 1
            exact_seen.add(text_hash)
            simhashes.append((current_simhash, text_hash))
            accepted.append(record)
        source_rows.append(
            {
                "path": str(path),
                "source_id": source_id,
                "format": fmt,
                "file_sha256": _file_sha256(path),
                "records_total": source_total,
                "records_accepted": source_accepted,
                "records_rejected": source_rejected,
            }
        )

    splits = _split_records(accepted, config)
    for split, rows in splits.items():
        (output_dir / f"{split}.txt").write_text(
            "\n".join(row["text"] for row in rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    (output_dir / "corpus.txt").write_text(
        "\n".join(row["text"] for row in accepted) + ("\n" if accepted else ""),
        encoding="utf-8",
    )
    with (output_dir / "rejected_records.jsonl").open("w", encoding="utf-8") as f:
        for row in rejected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reject_counts = Counter(row["reason"] for row in rejected)
    split_hashes = {
        name: _sha256_text("\n".join(row["text_hash"] for row in rows))
        for name, rows in splits.items()
    }
    quality_report = {
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
        "records_total": len(accepted) + len(rejected),
        "records_accepted": len(accepted),
        "records_rejected": len(rejected),
        "reject_counts": dict(reject_counts),
        "avg_persian_ratio": (
            sum(row["persian_ratio"] for row in accepted) / max(1, len(accepted))
        ),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_hashes": split_hashes,
        "contamination_eval_paths": list(config.eval_paths),
    }
    manifest = {
        "config": asdict(config),
        "sources": source_rows,
        "outputs": {
            "corpus": "corpus.txt",
            "train": "train.txt",
            "validation": "validation.txt",
            "test": "test.txt",
            "quality_report": "quality_report.json",
            "rejected_records": "rejected_records.jsonl",
        },
        "quality_report": quality_report,
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


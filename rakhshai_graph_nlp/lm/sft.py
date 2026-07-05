"""Native supervised fine-tuning from human-authored Persian instruction data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .model import GraphLMConfig
from .trainer import LMTrainingConfig, train_graph_lm


@dataclass
class SFTConfig:
    input_path: str
    output_dir: str
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    messages_field: str = "messages"
    min_completion_chars: int = 1
    epochs: int = 1
    batch_size: int = 1
    block_size: int = 128
    validation_ratio: float = 0.1
    tokenizer_type: str = "unigram"
    tokenizer_unigram_num_pieces: int = 8000
    device: str = "cpu"
    seed: int = 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _messages_to_prompt_completion(messages: list[dict[str, Any]]) -> tuple[str, str]:
    prompt_parts: list[str] = []
    completion_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            completion_parts.append(content)
        else:
            prompt_parts.append(f"{role}: {content}")
    return "\n".join(prompt_parts), "\n".join(completion_parts)


def format_sft_record(row: dict[str, Any], config: SFTConfig) -> str:
    if isinstance(row.get(config.messages_field), list):
        prompt, completion = _messages_to_prompt_completion(row[config.messages_field])
    else:
        prompt = str(row.get(config.prompt_field, "")).strip()
        completion = str(row.get(config.completion_field, "")).strip()
    if len(completion) < config.min_completion_chars:
        return ""
    return f"دستور:\n{prompt}\n\nپاسخ:\n{completion}"


def load_sft_texts(config: SFTConfig) -> list[str]:
    rows = _read_jsonl(Path(config.input_path))
    texts = [format_sft_record(row, config) for row in rows]
    return [text for text in texts if text.strip()]


def train_sft(config: SFTConfig, model_config: GraphLMConfig | None = None) -> dict[str, Any]:
    texts = load_sft_texts(config)
    if not texts:
        raise ValueError("SFT input contains no usable prompt/completion pairs")
    training_config = LMTrainingConfig(
        output_dir=config.output_dir,
        epochs=config.epochs,
        batch_size=config.batch_size,
        block_size=config.block_size,
        validation_ratio=config.validation_ratio,
        task_losses="next_token",
        tokenizer_type=config.tokenizer_type,
        tokenizer_unigram_num_pieces=config.tokenizer_unigram_num_pieces,
        early_stopping_patience=0,
        device=config.device,
        seed=config.seed,
    )
    metrics = train_graph_lm(
        texts,
        training_config=training_config,
        model_config=model_config,
        graph_encoder="none" if model_config is None else model_config.graph_encoder,
    )
    manifest = {
        "config": asdict(config),
        "records_used": len(texts),
        "checkpoint_dir": metrics.get("checkpoint_dir"),
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics["sft_manifest"] = manifest
    return metrics


"""Native local evaluation for independent Persian LM checkpoints."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from .model import GraphCausalLM, perplexity
from .tokenizer import PersianTokenizer


@dataclass
class NativeEvalConfig:
    model_dir: str
    eval_path: str
    output_path: str | None = None
    text_field: str = "text"
    prompt_field: str = "prompt"
    choices_field: str = "choices"
    answer_field: str = "answer"
    prediction_field: str = "prediction"
    block_size: int = 128
    device: str = "cpu"
    generation_prompts: Sequence[str] = field(default_factory=list)
    max_new_tokens: int = 32


def _read_eval_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append(row if isinstance(row, dict) else {"text": row})
        return rows
    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f, delimiter=delimiter)]
    return [{"text": line} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalise_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return text.strip()


def _f1(prediction: str, answers: Sequence[str]) -> float:
    pred_tokens = _normalise_answer(prediction).split()
    best = 0.0
    for answer in answers:
        gold_tokens = _normalise_answer(answer).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        overlap = 0
        remaining = list(gold_tokens)
        for token in pred_tokens:
            if token in remaining:
                overlap += 1
                remaining.remove(token)
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _loss_for_ids(model: GraphCausalLM, ids: list[int], pad_id: int, block_size: int) -> tuple[float, int]:
    if len(ids) < 2:
        return 0.0, 0
    losses: list[float] = []
    tokens = 0
    for start in range(0, max(1, len(ids) - 1), block_size):
        chunk = ids[start : start + block_size + 1]
        if len(chunk) < 2:
            continue
        input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=next(model.parameters()).device)
        labels = torch.tensor([chunk[1:]], dtype=torch.long, device=input_ids.device)
        with torch.no_grad():
            output = model(input_ids, labels=labels)
        count = int(labels.ne(-100).sum().item())
        losses.append(float(output["loss"].detach().cpu()) * count)
        tokens += count
    return sum(losses), tokens


def score_texts(
    model: GraphCausalLM,
    tokenizer: PersianTokenizer,
    texts: Sequence[str],
    *,
    block_size: int = 128,
) -> dict[str, float]:
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        loss, tokens = _loss_for_ids(
            model,
            tokenizer.encode(text, add_special_tokens=True),
            tokenizer.pad_id,
            block_size,
        )
        total_loss += loss
        total_tokens += tokens
    avg_loss = total_loss / max(1, total_tokens)
    return {
        "next_token_loss": avg_loss,
        "perplexity": perplexity(avg_loss),
        "tokens": float(total_tokens),
    }


def score_prompt_completion(
    model: GraphCausalLM,
    tokenizer: PersianTokenizer,
    prompt: str,
    completion: str,
    *,
    block_size: int = 128,
) -> dict[str, float]:
    ids = tokenizer.encode(f"{prompt} {completion}", add_special_tokens=True)
    loss, tokens = _loss_for_ids(model, ids, tokenizer.pad_id, block_size)
    return {
        "log_likelihood": -loss,
        "avg_loss": loss / max(1, tokens),
        "tokens": float(tokens),
    }


def evaluate_lm_checkpoint(config: NativeEvalConfig) -> dict[str, Any]:
    device = torch.device(
        "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model, tokenizer, generation_config, _graph_config = GraphCausalLM.from_pretrained(
        config.model_dir,
        map_location=device,
    )
    model.to(device)
    model.eval()
    rows = _read_eval_rows(Path(config.eval_path))
    texts = [str(row.get(config.text_field, "")).strip() for row in rows if row.get(config.text_field)]
    report: dict[str, Any] = {
        "config": asdict(config),
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
        "row_count": len(rows),
    }
    if texts:
        report["perplexity"] = score_texts(
            model,
            tokenizer,
            texts,
            block_size=config.block_size,
        )
    multiple_choice: list[dict[str, Any]] = []
    for row in rows:
        choices = row.get(config.choices_field)
        prompt = row.get(config.prompt_field)
        answer = row.get(config.answer_field)
        if not prompt or not isinstance(choices, list) or not choices:
            continue
        scored = [
            {
                "choice": choice,
                **score_prompt_completion(
                    model,
                    tokenizer,
                    str(prompt),
                    str(choice),
                    block_size=config.block_size,
                ),
            }
            for choice in choices
        ]
        prediction = max(scored, key=lambda item: item["log_likelihood"])
        multiple_choice.append(
            {
                "prompt": prompt,
                "prediction": prediction["choice"],
                "answer": answer,
                "correct": prediction["choice"] == answer,
                "scores": scored,
            }
        )
    if multiple_choice:
        report["multiple_choice"] = {
            "items": multiple_choice,
            "accuracy": sum(1 for row in multiple_choice if row["correct"]) / len(multiple_choice),
        }
    qa_rows = []
    for row in rows:
        prediction = row.get(config.prediction_field)
        answer = row.get(config.answer_field)
        if prediction is None or answer is None:
            continue
        answers = answer if isinstance(answer, list) else [str(answer)]
        qa_rows.append(
            {
                "prediction": prediction,
                "answers": answers,
                "exact": _normalise_answer(str(prediction))
                in {_normalise_answer(str(item)) for item in answers},
                "f1": _f1(str(prediction), [str(item) for item in answers]),
            }
        )
    if qa_rows:
        report["extractive_qa"] = {
            "exact_match": sum(1 for row in qa_rows if row["exact"]) / len(qa_rows),
            "f1": sum(float(row["f1"]) for row in qa_rows) / len(qa_rows),
            "items": qa_rows,
        }
    probes = []
    if config.generation_prompts:
        generation_config.max_new_tokens = config.max_new_tokens
        for prompt in config.generation_prompts:
            probes.append({"prompt": prompt, "output": model.generate(prompt, tokenizer, generation_config)})
        report["generation_probes"] = probes
    output_path = Path(config.output_path) if config.output_path else Path(config.model_dir) / "native_eval_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def export_human_review(
    prompts: Sequence[str],
    outputs: Sequence[str],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write human-review templates without invoking an external judge."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "human_review_items.jsonl"
    csv_path = root / "human_review_items.csv"
    schema_path = root / "human_review_schema.json"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for idx, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
            f.write(
                json.dumps(
                    {"id": idx, "prompt": prompt, "output": output, "fluency": "", "factuality": "", "notes": ""},
                    ensure_ascii=False,
                )
                + "\n"
            )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "output", "fluency", "factuality", "notes"])
        writer.writeheader()
        for idx, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
            writer.writerow({"id": idx, "prompt": prompt, "output": output, "fluency": "", "factuality": "", "notes": ""})
    schema_path.write_text(
        json.dumps(
            {
                "fluency": "Human 1-5 score.",
                "factuality": "Human 1-5 score against known references.",
                "notes": "Free-form human reviewer notes.",
                "uses_external_llm_judge": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "schema": str(schema_path),
    }

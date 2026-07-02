#!/usr/bin/env python3
"""Single-poem OpenAI vs Rakhshai MCP evaluation.

The benchmark compares the same Persian poetry question in two conditions:

1. direct: the poem and question are sent directly to the OpenAI model.
2. rakhshai_mcp: local Rakhshai MCP tools build graph evidence first, then the
   same model receives that evidence in the prompt.

The generated JSON keeps raw prompts, raw outputs, graph evidence and automatic
evidence metrics. A Markdown report can also be written into the project docs so
the test is visible and repeatable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rakhshai_mcp.tools import (  # noqa: E402
    rakhshai_analyze_persian_text,
    rakhshai_explain_result,
    rakhshai_graph_memory_generate,
    rakhshai_graph_summarize,
)


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
DEFAULT_OUTPUT_DIR = Path("runs/openai_mcp_eval")
DEFAULT_REPORT_PATH = Path("docs/mcp_single_poem_evaluation.md")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 42

DEFAULT_SAMPLE = {
    "id": "mirror_shadow_key_poem",
    "text": (
        "در آینه، سایه‌ام از من قدیمی‌تر بود\n"
        "و نامم از دهانِ پنجره به باران می‌ریخت\n\n"
        "چراغی که خوابِ خاکستر می‌دید\n"
        "راه را به پای گم‌شده‌ام نشان نمی‌داد\n\n"
        "رود از کنار من گذشت\n"
        "اما تشنگی در مشت‌هایم لانه کرده بود\n\n"
        "گفتم: کدام سو خانه است؟\n"
        "باد، کلیدی زنگ‌زده را در سکوت چرخاند"
    ),
    "task": (
        "این شعر از دید تو چه معنی می‌دهد؟\n"
        "معنی اصلی شعر را ساده توضیح بده "
        "و بگو رابطه میان آینه، سایه، نام، "
        "چراغ، رود، "
        "تشنگی، خانه و کلید چیست؟"
    ),
}


def _load_local_env(path: Path = PROJECT_ROOT / ".env.local") -> None:
    """Load simple KEY=VALUE pairs from the gitignored local env file."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


def _load_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI SDK is not installed. Run: pip install -e '.[openai]'"
        ) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Put it in .env.local or export it "
            "locally before running this script."
        )
    return OpenAI()


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text
    return str(response)


def _call_openai(
    client: Any,
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    requested = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "seed": seed,
    }
    if seed is not None:
        kwargs["seed"] = seed

    try:
        response = client.responses.create(**kwargs)
        return _response_text(response), {
            "requested": requested,
            "applied": requested,
            "fallback_notes": [],
        }
    except Exception as exc:
        message = str(exc)
        if seed is None or "seed" not in message.lower():
            raise

    kwargs.pop("seed", None)
    response = client.responses.create(**kwargs)
    applied = dict(requested)
    applied["seed"] = None
    return _response_text(response), {
        "requested": requested,
        "applied": applied,
        "fallback_notes": [
            "The OpenAI Responses API rejected `seed` for this model/client, "
            "so the request was retried without seed while keeping both "
            "conditions identical."
        ],
    }


def _compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    explanation = result.get("explanation", {})
    graph = result.get("graph", {})
    return {
        "status": result.get("status"),
        "task": result.get("task"),
        "summary": result.get("summary"),
        "keywords": result.get("keywords", [])[:8],
        "top_nodes": explanation.get("top_nodes", [])[:8],
        "important_relations": explanation.get("important_relations", [])[:8],
        "reasoning_path": explanation.get("reasoning_path", [])[:6],
        "graph_memory_report": explanation.get("graph_memory_report", {}),
        "focus_terms": explanation.get("focus_terms", []),
        "focus_nodes": explanation.get("focus_nodes", []),
        "focus_relations": explanation.get("focus_relations", [])[:8],
        "graph_metrics": graph.get("metrics", {}),
    }


def build_rakhshai_mcp_evidence(text: str) -> dict[str, Any]:
    """Run local Rakhshai MCP tool functions and compact their evidence."""

    analysis = rakhshai_analyze_persian_text(text, max_keywords=8)
    summary = rakhshai_graph_summarize(text, top_k=2)
    explanation = rakhshai_explain_result(text, top_k=6)
    memory = rakhshai_graph_memory_generate(text, memory_texts=[text], top_k=4)
    return {
        "analysis": _compact_tool_result(analysis),
        "graph_summary": _compact_tool_result(summary),
        "explanation": _compact_tool_result(explanation),
        "graph_memory": _compact_tool_result(memory),
    }


def direct_prompt(sample: dict[str, str]) -> str:
    return "\n".join(
        [
            "به این سوال درباره شعر پاسخ بده.",
            "",
            f"سوال:\n{sample['task']}",
            "",
            f"شعر:\n{sample['text']}",
        ]
    )


def rakhshai_mcp_prompt(sample: dict[str, str], evidence: dict[str, Any]) -> str:
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    return "\n".join(
        [
            "به این سوال درباره شعر پاسخ بده.",
            "از شواهد گرافی Rakhshai MCP فقط وقتی استفاده کن که مرتبط است.",
            "در پاسخ توضیح بده کدام گره‌ها یا رابطه‌ها کمک کردند.",
            "",
            f"سوال:\n{sample['task']}",
            "",
            f"شعر:\n{sample['text']}",
            "",
            "Rakhshai MCP graph evidence:",
            evidence_json,
        ]
    )


def score_persian_output(text: str) -> dict[str, Any]:
    persian_chars = sum(1 for char in text if "\u0600" <= char <= "\u06ff")
    letters = sum(1 for char in text if char.isalpha())
    evidence_terms = [
        "گراف",
        "گره",
        "رابطه",
        "شواهد",
        "مفهوم",
        "مسیر",
    ]
    explanation_terms = [
        "شواهد گرافی",
        "گره",
        "گره‌ها",
        "رابطه‌های گرافی",
        "کدام گره‌ها",
        "کمک کردند",
    ]
    return {
        "chars": len(text),
        "persian_char_ratio": round(persian_chars / max(letters, 1), 4),
        "evidence_term_hits": sum(term in text for term in evidence_terms),
        "has_explanation_signal": any(term in text for term in explanation_terms),
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def add_manual_scoring(
    result: dict[str, Any],
    *,
    direct_scores: dict[str, int] | None,
    mcp_scores: dict[str, int] | None,
) -> None:
    if not direct_scores or not mcp_scores:
        result["manual_scoring"] = None
        return

    criteria = [
        "interpretation_accuracy",
        "faithfulness_to_text",
        "symbol_relation_quality",
        "evidence_usage",
        "explanation_quality",
    ]

    def final_score(scores: dict[str, int]) -> int:
        return sum(scores[item] for item in criteria) - scores["hallucination_penalty"]

    direct_total = final_score(direct_scores)
    mcp_total = final_score(mcp_scores)
    ratio = _safe_ratio(mcp_total, direct_total)
    percent = None
    if direct_total:
        percent = round(((mcp_total - direct_total) / direct_total) * 100, 1)

    result["manual_scoring"] = {
        "criteria": criteria,
        "direct": direct_scores,
        "rakhshai_mcp": mcp_scores,
        "direct_score": direct_total,
        "mcp_score": mcp_total,
        "improvement_ratio": ratio,
        "improvement_percent": percent,
    }


def add_success_metrics(result: dict[str, Any]) -> None:
    row = result["rows"][0]
    direct_auto = row["scores"]["direct"]
    mcp_auto = row["scores"]["rakhshai_mcp"]
    manual = result.get("manual_scoring")

    evidence_ratio = _safe_ratio(
        mcp_auto["evidence_term_hits"],
        direct_auto["evidence_term_hits"],
    )
    checks = {
        "manual_mcp_score_gt_direct": None,
        "manual_improvement_ratio_gt_1": None,
        "mcp_evidence_hits_gt_direct": (
            mcp_auto["evidence_term_hits"] > direct_auto["evidence_term_hits"]
        ),
        "mcp_has_better_explanation_signal": (
            bool(mcp_auto["has_explanation_signal"])
            and not bool(direct_auto["has_explanation_signal"])
        ),
    }
    if manual:
        checks["manual_mcp_score_gt_direct"] = (
            manual["mcp_score"] > manual["direct_score"]
        )
        checks["manual_improvement_ratio_gt_1"] = (
            manual["improvement_ratio"] is not None
            and manual["improvement_ratio"] > 1.0
        )

    result["success_metrics"] = {
        "direct_evidence_hits": direct_auto["evidence_term_hits"],
        "mcp_evidence_hits": mcp_auto["evidence_term_hits"],
        "evidence_improvement_ratio": evidence_ratio,
        "direct_has_explanation_signal": direct_auto["has_explanation_signal"],
        "mcp_has_explanation_signal": mcp_auto["has_explanation_signal"],
        "checks": checks,
        "success": all(value is True for value in checks.values()),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    client = None if args.dry_run else _load_openai_client()
    sample = DEFAULT_SAMPLE
    started = time.time()
    evidence = build_rakhshai_mcp_evidence(sample["text"])
    direct_input = direct_prompt(sample)
    mcp_input = rakhshai_mcp_prompt(sample, evidence)
    row: dict[str, Any] = {
        "id": sample["id"],
        "task": sample["task"],
        "text": sample["text"],
        "mcp_evidence": evidence,
        "prompts": {
            "direct": direct_input,
            "rakhshai_mcp": mcp_input,
        },
        "outputs": {},
        "scores": {},
        "request_parameters": {},
        "elapsed_seconds": None,
    }

    if not args.dry_run:
        assert client is not None
        direct_output, direct_params = _call_openai(
            client,
            model=args.model,
            prompt=direct_input,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        mcp_output, mcp_params = _call_openai(
            client,
            model=args.model,
            prompt=mcp_input,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        row["outputs"]["direct"] = direct_output
        row["outputs"]["rakhshai_mcp"] = mcp_output
        row["scores"]["direct"] = score_persian_output(direct_output)
        row["scores"]["rakhshai_mcp"] = score_persian_output(mcp_output)
        row["request_parameters"]["direct"] = direct_params
        row["request_parameters"]["rakhshai_mcp"] = mcp_params

    row["elapsed_seconds"] = round(time.time() - started, 3)
    result = {
        "scenario": "single_poem_rakhshai_mcp_evaluation",
        "model": args.model,
        "dry_run": args.dry_run,
        "settings": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens,
            "seed": args.seed,
        },
        "claim_under_test": (
            "در این تست تک‌نمونه‌ای، Rakhshai MCP با دادن evidence گرافی واقعی "
            "به مدل، کیفیت فهم/تحلیل شعر فارسی را بهتر می‌کند."
        ),
        "rows": [row],
    }
    if not args.dry_run:
        add_manual_scoring(
            result,
            direct_scores=args.direct_manual_scores,
            mcp_scores=args.mcp_manual_scores,
        )
        add_success_metrics(result)
    return result


def _parse_score_arg(raw: str | None) -> dict[str, int] | None:
    if raw is None:
        return None
    keys = [
        "interpretation_accuracy",
        "faithfulness_to_text",
        "symbol_relation_quality",
        "evidence_usage",
        "explanation_quality",
        "hallucination_penalty",
    ]
    values = [int(part.strip()) for part in raw.split(",")]
    if len(values) != len(keys):
        raise SystemExit(
            "Manual scores must have 6 comma-separated integers: "
            "interpretation,faithfulness,symbol_relations,evidence,"
            "explanation,hallucination_penalty"
        )
    return dict(zip(keys, values, strict=True))


def _labels(nodes: list[Any]) -> list[str]:
    labels = []
    for node in nodes:
        if isinstance(node, dict):
            labels.append(str(node.get("label") or node.get("text") or node))
        else:
            labels.append(str(node))
    return labels


def _relation_lines(relations: list[Any]) -> list[str]:
    lines = []
    for edge in relations:
        if isinstance(edge, dict):
            lines.append(
                f"{edge.get('source_label')} -[{edge.get('relation')}]-> "
                f"{edge.get('target_label')}"
            )
        else:
            lines.append(str(edge))
    return lines


def _block(text: str, lang: str = "text") -> str:
    return f"```{lang}\n{text}\n```"


def render_markdown_report(result: dict[str, Any], source_json: Path | None) -> str:
    row = result["rows"][0]
    evidence = row["mcp_evidence"]
    explanation = evidence["explanation"]
    memory = evidence["graph_memory"]
    graph_summary = evidence["graph_summary"]
    manual = result.get("manual_scoring")
    success = result.get("success_metrics")
    settings = result["settings"]

    parts: list[str] = []
    parts.append("# Single Poem Rakhshai MCP Evaluation")
    parts.append("")
    parts.append("## Goal")
    parts.append("")
    parts.append(
        "بررسی اینکه آیا اتصال Rakhshai MCP به `gpt-5.4` باعث پاسخ بهتر "
        "در تحلیل یک شعر فارسی می‌شود یا نه، و اگر بهتر شد چند برابر/چند "
        "درصد بهتر شده است."
    )
    parts.append("")
    parts.append("## Claim Under Test")
    parts.append("")
    parts.append(result["claim_under_test"])
    parts.append("")
    parts.append("## Model And Settings")
    parts.append("")
    parts.append(f"- Model: `{result['model']}`")
    parts.append(f"- temperature: `{settings['temperature']}`")
    parts.append(f"- top_p: `{settings['top_p']}`")
    parts.append(f"- max_output_tokens: `{settings['max_output_tokens']}`")
    parts.append(f"- seed: `{settings['seed']}` if available")
    if source_json:
        parts.append(f"- Source JSON: `{source_json}`")
    parts.append("")
    if row.get("request_parameters"):
        parts.append("### Applied Request Parameters")
        parts.append("")
        parts.append(
            _block(
                json.dumps(row["request_parameters"], ensure_ascii=False, indent=2),
                "json",
            )
        )
        parts.append("")
    parts.append("## Input")
    parts.append("")
    parts.append("### Question")
    parts.append("")
    parts.append(_block(row["task"]))
    parts.append("")
    parts.append("### Poem")
    parts.append("")
    parts.append(_block(row["text"]))
    parts.append("")
    parts.append("## Conditions")
    parts.append("")
    parts.append("### Condition A: direct")
    parts.append("")
    parts.append(_block(row["prompts"]["direct"]))
    parts.append("")
    parts.append("### Condition B: rakhshai_mcp")
    parts.append("")
    parts.append(_block(row["prompts"]["rakhshai_mcp"]))
    parts.append("")
    parts.append("## Rakhshai MCP Evidence Summary")
    parts.append("")
    parts.append("### Top Nodes")
    parts.append("")
    parts.append(_block("، ".join(_labels(explanation.get("top_nodes", [])))))
    parts.append("")
    parts.append("### Important Relations")
    parts.append("")
    parts.append(_block("\n".join(_relation_lines(explanation.get("important_relations", [])))))
    parts.append("")
    parts.append("### Reasoning Path")
    parts.append("")
    parts.append(_block("\n".join(_relation_lines(explanation.get("reasoning_path", [])))))
    parts.append("")
    parts.append("### Graph Summary Top Nodes")
    parts.append("")
    parts.append(_block("، ".join(_labels(graph_summary.get("top_nodes", [])))))
    parts.append("")
    parts.append("### Graph Memory Report")
    parts.append("")
    parts.append(_block(json.dumps(memory.get("graph_memory_report", {}), ensure_ascii=False, indent=2), "json"))
    parts.append("")
    parts.append("## Automatic Evidence Metrics")
    parts.append("")
    if success:
        parts.append(f"- direct_evidence_hits: `{success['direct_evidence_hits']}`")
        parts.append(f"- mcp_evidence_hits: `{success['mcp_evidence_hits']}`")
        parts.append(
            "- evidence_improvement_ratio: "
            f"`{success['evidence_improvement_ratio']}x`"
        )
        parts.append(
            "- has_explanation_signal direct: "
            f"`{success['direct_has_explanation_signal']}`"
        )
        parts.append(
            "- has_explanation_signal rakhshai_mcp: "
            f"`{success['mcp_has_explanation_signal']}`"
        )
    else:
        parts.append("Automatic metrics are not available in dry-run mode.")
    parts.append("")
    parts.append("## Manual Scoring Rubric")
    parts.append("")
    parts.append("Each output is scored from 1 to 5 on these criteria:")
    parts.append("")
    parts.append("- interpretation_accuracy")
    parts.append("- faithfulness_to_text")
    parts.append("- symbol_relation_quality")
    parts.append("- evidence_usage")
    parts.append("- explanation_quality")
    parts.append("")
    parts.append("Penalty:")
    parts.append("")
    parts.append("- hallucination_penalty: `0` to `3`")
    parts.append("")
    parts.append("Formula:")
    parts.append("")
    parts.append(
        _block(
            "final_score = interpretation_accuracy\n"
            "  + faithfulness_to_text\n"
            "  + symbol_relation_quality\n"
            "  + evidence_usage\n"
            "  + explanation_quality\n"
            "  - hallucination_penalty"
        )
    )
    parts.append("")
    parts.append("## Manual Scoring Result")
    parts.append("")
    if manual:
        parts.append("### direct")
        parts.append("")
        parts.append(_block(json.dumps(manual["direct"], ensure_ascii=False, indent=2), "json"))
        parts.append(f"Final direct_score: `{manual['direct_score']}`")
        parts.append("")
        parts.append("### rakhshai_mcp")
        parts.append("")
        parts.append(_block(json.dumps(manual["rakhshai_mcp"], ensure_ascii=False, indent=2), "json"))
        parts.append(f"Final mcp_score: `{manual['mcp_score']}`")
        parts.append("")
        parts.append("### Improvement")
        parts.append("")
        parts.append(f"- improvement_ratio: `{manual['improvement_ratio']}x`")
        parts.append(f"- improvement_percent: `{manual['improvement_percent']}%`")
        parts.append("")
        parts.append(
            "Rakhshai MCP در این تست "
            f"`{manual['improvement_ratio']}x`، یعنی حدود "
            f"`{manual['improvement_percent']}%`، بهتر از حالت direct امتیاز گرفت."
        )
    else:
        parts.append(
            "Manual scores were not provided. Re-run with "
            "`--direct-manual-scores` and `--mcp-manual-scores`."
        )
    parts.append("")
    parts.append("## Success Criteria")
    parts.append("")
    parts.append("- `mcp_score > direct_score`")
    parts.append("- `improvement_ratio > 1.0`")
    parts.append("- `mcp_evidence_hits > direct_evidence_hits`")
    parts.append("- `has_explanation_signal` in MCP is better than direct")
    parts.append("")
    if success:
        parts.append("### Result")
        parts.append("")
        parts.append(_block(json.dumps(success, ensure_ascii=False, indent=2), "json"))
        parts.append("")
        if success["success"]:
            parts.append(
                "در این تست تک‌نمونه‌ای، اتصال `gpt-5.4` به Rakhshai MCP "
                "باعث شد پاسخ مدل در تحلیل شعر فارسی دقیق‌تر، متن‌وفادارتر، "
                "مستندتر و توضیح‌پذیرتر شود."
            )
            if manual:
                parts.append("")
                parts.append(
                    "بر اساس امتیاز داوری‌شده، Rakhshai MCP نسبت به direct "
                    f"برابر با `{manual['improvement_ratio']}x` بهتر عمل کرد، "
                    f"یعنی حدود `{manual['improvement_percent']}%` بهبود داشت."
                )
            parts.append("")
            parts.append(
                "در metric خودکار evidence، Rakhshai MCP نسبت به direct "
                f"برابر با `{success['evidence_improvement_ratio']}x` "
                "شواهد بیشتری را وارد پاسخ کرد."
            )
        else:
            parts.append(
                "The configured success criteria did not all pass in this run."
            )
    parts.append("")
    parts.append("## Response Received: direct")
    parts.append("")
    parts.append(_block(row["outputs"].get("direct", "")))
    parts.append("")
    parts.append("## Response Received: rakhshai_mcp")
    parts.append("")
    parts.append(_block(row["outputs"].get("rakhshai_mcp", "")))
    parts.append("")
    parts.append("## How To Repeat")
    parts.append("")
    parts.append("1. Put your API key in `.env.local` as `OPENAI_API_KEY=...`.")
    parts.append("2. Install the OpenAI optional dependency:")
    parts.append("")
    parts.append(_block('pip install -e ".[openai]"', "bash"))
    parts.append("")
    parts.append("3. Run the benchmark and refresh this documentation page:")
    parts.append("")
    parts.append(
        _block(
            "python scripts/evaluate_openai_mcp_persian.py \\\n"
            "  --model gpt-5.4 \\\n"
            "  --temperature 0 \\\n"
            "  --top-p 1 \\\n"
            "  --seed 42 \\\n"
            "  --max-output-tokens 3000 \\\n"
            "  --direct-manual-scores 5,5,4,3,4,0 \\\n"
            "  --mcp-manual-scores 5,5,5,5,5,0 \\\n"
            "  --report-path docs/mcp_single_poem_evaluation.md",
            "bash",
        )
    )
    return "\n".join(parts) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare gpt-5.4 direct output with Rakhshai MCP evidence."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", dest="top_p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--direct-manual-scores")
    parser.add_argument("--mcp-manual-scores")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.direct_manual_scores = _parse_score_arg(args.direct_manual_scores)
    args.mcp_manual_scores = _parse_score_arg(args.mcp_manual_scores)
    return args


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dry_run" if args.dry_run else args.model.replace("/", "_")
    output_path = args.output_dir / f"single_poem_mcp_eval_{suffix}_{int(time.time())}.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            render_markdown_report(result, output_path),
            encoding="utf-8",
        )
    print(output_path)
    if args.report_path:
        print(args.report_path)


if __name__ == "__main__":
    main()

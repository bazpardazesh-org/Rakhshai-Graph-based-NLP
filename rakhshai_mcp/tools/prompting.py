"""Prompt optimization tools for Persian OpenAI/agent workflows."""

from __future__ import annotations

import re
from typing import Any

from ..schemas.outputs import error_response, standard_response
from ..security import RATE_LIMITER, validate_text
from .common import (
    important_relations,
    is_meaningful_label,
    relation_paths_from_edges,
    tokenize_documents,
)
from .graph import rakhshai_build_knowledge_graph


def _format_nodes(nodes: list[dict[str, Any]], *, limit: int = 8) -> str:
    labels = [str(node.get("label")) for node in nodes[:limit] if node.get("label")]
    return "، ".join(labels)


def _format_relations(relations: list[dict[str, Any]], *, limit: int = 6) -> str:
    lines = []
    for edge in relations[:limit]:
        source = edge.get("source_label")
        target = edge.get("target_label")
        relation = edge.get("relation", "relation")
        if source and target:
            lines.append(f"- {source} -[{relation}]-> {target}")
    return "\n".join(lines)


def _concept_root(label: str) -> str:
    root = label.strip().strip("،.;:؟!()[]{}\"'")
    suffixes = (
        "هایم",
        "هایت",
        "هایش",
        "هایمان",
        "هایتان",
        "هایشان",
        "ها",
        "ام",
        "ات",
        "اش",
        "مان",
        "تان",
        "شان",
        "ی",
        "م",
    )
    for suffix in suffixes:
        if suffix == "ی" and root.endswith("گی"):
            continue
        if root.endswith(suffix) and len(root) > len(suffix) + 2:
            return root[: -len(suffix)].rstrip("\u200c")
    return root.rstrip("\u200c")


def _extract_focus_terms(task: str) -> list[str]:
    match = re.search(
        r"رابطه\s+میان\s+(.+?)(?:چیست|است|هست|[؟?]|$)",
        task,
        flags=re.DOTALL,
    )
    if match:
        raw = match.group(1)
        parts = re.split(r"[،,\n]|(?:\s+و\s+)", raw)
        terms = [part.strip() for part in parts if part.strip()]
    else:
        terms = tokenize_documents([task])[0]
    seen: set[str] = set()
    focus_terms = []
    for term in terms:
        root = _concept_root(term)
        if not is_meaningful_label(root) or root in seen:
            continue
        seen.add(root)
        focus_terms.append(root)
    return focus_terms[:12]


def _focus_evidence(
    focus_terms: list[str],
    graph_payload: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    if not focus_terms:
        return [], []
    focus_roots = {_concept_root(term) for term in focus_terms}
    matched_nodes = []
    for node in graph_payload.get("nodes", []):
        label = str(node.get("label", ""))
        if _concept_root(label) in focus_roots and label not in matched_nodes:
            matched_nodes.append(label)
    matched_roots = {_concept_root(label) for label in matched_nodes}
    relation_hints = []
    for edge in graph_payload.get("edges", []):
        source = str(edge.get("source_label", ""))
        target = str(edge.get("target_label", ""))
        if not is_meaningful_label(source) or not is_meaningful_label(target):
            continue
        if (
            _concept_root(source) in matched_roots
            or _concept_root(target) in matched_roots
        ):
            relation_hints.append(edge)
        if len(relation_hints) >= 12:
            break
    return matched_nodes, relation_hints


def _infer_response_plan(task: str) -> list[str]:
    task_lower = task.lower()
    plan = [
        "پاسخ را به فارسی روان، "
        "دقیق و قابل اتکا بنویس."
    ]
    if "خلاصه" in task:
        plan.append(
            "ابتدا یک خلاصه کوتاه و روشن ارائه کن."
        )
    if "تحلیل" in task or "مفهوم" in task:
        plan.append(
            "مفاهیم مرکزی را جداگانه استخراج "
            "و توضیح بده."
        )
    if "رابطه" in task:
        plan.append(
            "رابطه میان تصویرها "
            "و نمادهای اصلی را روشن کن."
        )
    if "شواهد" in task or "توضیح" in task or "چرا" in task_lower:
        plan.append(
            "برای هر نتیجه، شواهد متنی "
            "یا گرافی مرتبط را ذکر کن."
        )
    if "مقایسه" in task:
        plan.append(
            "مقایسه را با معیارهای روشن "
            "و نتیجه نهایی انجام بده."
        )
    if len(plan) == 1:
        plan.append(
            "پاسخ را ساختارمند بنویس "
            "و از متن داده‌شده فراتر نرو."
        )
    return plan


def rakhshai_optimize_persian_prompt(
    task: str,
    text: str,
    *,
    include_graph_evidence: bool = True,
) -> dict[str, Any]:
    """Rewrite a raw Persian task into a graph-aware prompt for LLMs."""

    tool_task = "persian_prompt_optimization"
    try:
        RATE_LIMITER.check()
        cleaned_task = validate_text(task)
        cleaned_text = validate_text(text)
        graph_result = (
            rakhshai_build_knowledge_graph(text=cleaned_text)
            if include_graph_evidence
            else {}
        )
        explanation = graph_result.get("explanation", {})
        top_nodes = explanation.get("top_nodes", [])
        relations = explanation.get("important_relations", [])
        reasoning_path = explanation.get("reasoning_path", [])
        focus_terms = _extract_focus_terms(cleaned_task)
        focus_nodes, focus_relations = _focus_evidence(
            focus_terms,
            graph_result.get("graph", {}),
        )
        relation_lines = _format_relations(relations)
        focus_relation_lines = _format_relations(focus_relations, limit=10)
        plan = _infer_response_plan(cleaned_task)

        sections = [
            "به فارسی روان، دقیق و قابل اتکا پاسخ بده.",
            "از متن داده‌شده فراتر نرو "
            "مگر اینکه صریحا لازم باشد.",
            "اگر شواهد گرافی مرتبط بودند، "
            "آن‌ها را برای توضیح تصمیم به کار ببر.",
            "",
            f"وظیفه بازنویسی‌شده:\n{cleaned_task}",
            "",
            "انتظار از پاسخ:",
            *[f"- {item}" for item in plan],
            "",
            f"متن:\n{cleaned_text}",
        ]
        if include_graph_evidence and graph_result:
            sections.extend(
                [
                    "",
                    "شواهد گرافی Rakhshai برای grounding پاسخ:",
                    f"- گره‌های مرکزی: {_format_nodes(top_nodes)}",
                ]
            )
            if focus_terms:
                sections.append(
                    "- نمادهای خواسته‌شده در سؤال: "
                    + "، ".join(focus_terms)
                )
            if focus_nodes:
                sections.append(
                    "- گره‌های متناظر با نمادهای سؤال: "
                    + "، ".join(focus_nodes)
                )
            if focus_relation_lines:
                sections.extend(
                    [
                        "- رابطه‌های پیرامون نمادهای سؤال:",
                        focus_relation_lines,
                    ]
                )
            if relation_lines:
                sections.extend(["- رابطه‌های مهم:", relation_lines])
            if reasoning_path:
                sections.append(
                    "- مسیرهای استدلالی: "
                    + " | ".join(str(path) for path in reasoning_path[:5])
                )
        optimized_prompt = "\n".join(sections)
        evidence_relations = important_relations(
            graph_result.get("graph", {"edges": []})
        )
        return standard_response(
            task=tool_task,
            summary=optimized_prompt,
            keywords=graph_result.get("keywords", []),
            graph=graph_result.get("graph", {"nodes": [], "edges": []}),
            explanation={
                "top_nodes": top_nodes,
                "important_relations": evidence_relations or relations,
                "reasoning_path": relation_paths_from_edges(evidence_relations)
                or reasoning_path,
                "focus_terms": focus_terms,
                "focus_nodes": focus_nodes,
                "focus_relations": focus_relations,
                "method": "Persian task rewrite + Rakhshai graph evidence",
            },
            artifacts={
                "optimized_prompt": optimized_prompt,
                "original_task": cleaned_task,
                "include_graph_evidence": include_graph_evidence,
            },
        )
    except Exception as exc:
        return error_response(tool_task, exc)

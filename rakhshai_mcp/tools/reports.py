"""MCP tools for explainable graph-NLP reports."""

from __future__ import annotations

from typing import Any

from ..schemas.outputs import error_response, standard_response
from ..security import RATE_LIMITER, validate_text, validate_top_k
from .common import (
    important_relations,
    new_artifact_id,
    relation_paths_from_edges,
    relation_path_from_nodes,
)
from .graph import rakhshai_build_knowledge_graph


def rakhshai_explain_result(
    text: str,
    result_summary: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Explain a Rakhshai result with important nodes and relation evidence."""

    task = "explainable_graph_nlp"
    try:
        RATE_LIMITER.check()
        cleaned = validate_text(text)
        top_k = validate_top_k(top_k, maximum=20)
        graph_result = rakhshai_build_knowledge_graph(text=cleaned)
        graph_payload = graph_result.get("graph", {"nodes": [], "edges": []})
        top_nodes = graph_result.get("explanation", {}).get("top_nodes", [])[:top_k]
        relations = important_relations(graph_payload, limit=top_k)
        reasoning_path = relation_paths_from_edges(relations, limit=top_k)
        if not reasoning_path:
            reasoning_path = relation_path_from_nodes(top_nodes, limit=top_k)
        summary = result_summary or " ".join(
            [
                "خروجی بر پایه گره‌های مرکزی",
                "و ارتباطات پرتکرار گراف توضیح داده شد.",
            ]
        )
        return standard_response(
            task=task,
            summary=summary,
            keywords=graph_result.get("keywords", [])[:top_k],
            graph=graph_payload,
            explanation={
                "top_nodes": top_nodes,
                "important_relations": relations,
                "reasoning_path": reasoning_path,
                "method": "multi-relation evidence over the generated graph",
            },
            artifacts={
                "report_id": new_artifact_id("report"),
                "source_graph_id": graph_result.get("artifacts", {}).get("graph_id"),
            },
        )
    except Exception as exc:
        return error_response(task, exc)

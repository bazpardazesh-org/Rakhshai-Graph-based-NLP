"""MCP tool for Persian text analysis."""

from __future__ import annotations

from typing import Any

from rakhshai_graph_nlp.graphs.co_occurrence import build_cooccurrence_graph

from ..config import MCPConfig
from ..schemas.outputs import error_response, standard_response
from ..security import MCPInputError, RATE_LIMITER, validate_text, validate_top_k
from .common import (
    extract_entities,
    graph_to_payload,
    important_relations,
    keyword_scores,
    relation_path_from_nodes,
    tokenize_documents,
    top_graph_nodes,
)
from .graph import rakhshai_build_knowledge_graph


def rakhshai_analyze_persian_text(
    text: str,
    max_keywords: int = 10,
) -> dict[str, Any]:
    """Analyze Persian text and expose keywords, entities and graph signals."""

    task = "persian_text_analysis"
    try:
        RATE_LIMITER.check()
        cleaned = validate_text(text)
        max_keywords = validate_top_k(max_keywords, maximum=25)
        token_lists = tokenize_documents([cleaned])
        tokens = token_lists[0]
        if not tokens:
            raise MCPInputError("text did not produce tokens")
        graph_result = rakhshai_build_knowledge_graph(text=cleaned)
        graph_payload = graph_result.get("graph", {"nodes": [], "edges": []})
        explanation = graph_result.get("explanation", {})
        top_nodes = explanation.get("top_nodes", [])[:max_keywords]
        return standard_response(
            task=task,
            summary="Persian text analyzed with Rakhshai graph signals.",
            keywords=keyword_scores(tokens, top_k=max_keywords),
            entities=extract_entities(cleaned),
            graph=graph_payload,
            explanation={
                "top_nodes": top_nodes,
                "important_relations": explanation.get("important_relations", []),
                "reasoning_path": explanation.get("reasoning_path", []),
                "method": explanation.get("method", "Rakhshai graph analysis"),
            },
        )
    except Exception as exc:
        return error_response(task, exc)


def analyze_with_config(
    text: str,
    *,
    config: MCPConfig,
    max_keywords: int = 10,
) -> dict[str, Any]:
    """Internal variant used by tests or embedded servers with custom limits."""

    task = "persian_text_analysis"
    try:
        cleaned = validate_text(text, config=config)
        max_keywords = validate_top_k(max_keywords, maximum=25)
        token_lists = tokenize_documents([cleaned])
        graph = build_cooccurrence_graph(token_lists, window_size=2, min_count=1)
        graph_payload = graph_to_payload(graph, config=config)
        top_nodes = top_graph_nodes(graph, top_k=max_keywords)
        return standard_response(
            task=task,
            summary="Persian text analyzed with lexical graph signals.",
            keywords=keyword_scores(token_lists[0], top_k=max_keywords),
            entities=extract_entities(cleaned),
            graph=graph_payload,
            explanation={
                "top_nodes": top_nodes,
                "important_relations": important_relations(graph_payload),
                "reasoning_path": relation_path_from_nodes(top_nodes),
                "method": "tokenization + cooccurrence graph + weighted degree",
            },
        )
    except Exception as exc:
        return error_response(task, exc)

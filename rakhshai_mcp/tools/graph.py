"""MCP tools for graph building and graph-based summarisation."""

from __future__ import annotations

from typing import Any

from rakhshai_graph_nlp.graphs.co_occurrence import build_cooccurrence_graph
from rakhshai_graph_nlp.lm.graph_builder import build_graph_lm_graph
from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer
from rakhshai_graph_nlp.tasks.summarization import textrank_summarise

from ..config import DEFAULT_CONFIG
from ..schemas.outputs import error_response, standard_response
from ..security import (
    MCPInputError,
    RATE_LIMITER,
    validate_documents,
    validate_text,
    validate_top_k,
    validate_window_size,
)
from .common import (
    graph_to_payload,
    graph_lm_to_payload,
    important_relations,
    keyword_scores,
    new_artifact_id,
    relation_paths_from_edges,
    relation_path_from_nodes,
    tokenize_documents,
    top_graph_lm_nodes,
    top_graph_nodes,
)


def _resolve_documents(text: str | None, documents: list[str] | None) -> list[str]:
    if documents is not None:
        return validate_documents(documents)
    if text is None:
        raise MCPInputError("provide either text or documents")
    return [validate_text(text)]


def rakhshai_build_knowledge_graph(
    text: str | None = None,
    documents: list[str] | None = None,
    window_size: int = 2,
    min_count: int = 1,
    graph_type: str = "graph_lm",
    graph_relations: list[str] | None = None,
) -> dict[str, Any]:
    """Build a bounded Persian knowledge graph from Rakhshai core builders."""

    task = "knowledge_graph_building"
    try:
        RATE_LIMITER.check()
        docs = _resolve_documents(text, documents)
        window_size = validate_window_size(window_size)
        if not isinstance(min_count, int) or min_count < 1:
            raise MCPInputError("min_count must be a positive integer")
        token_lists = tokenize_documents(docs)
        if not any(token_lists):
            raise MCPInputError("documents did not produce tokens")
        graph_type = graph_type.lower().strip()
        if graph_type not in {"graph_lm", "cooccurrence"}:
            raise MCPInputError("graph_type must be one of: graph_lm, cooccurrence")
        if graph_type == "cooccurrence":
            graph = build_cooccurrence_graph(
                token_lists,
                window_size=window_size,
                min_count=min_count,
            )
            graph_payload = graph_to_payload(graph, config=DEFAULT_CONFIG)
            top_nodes = top_graph_nodes(graph, top_k=10)
            method = "cooccurrence graph over Persian tokens"
            tokenizer_vocab_size = len(
                {token for tokens in token_lists for token in tokens}
            )
        else:
            tokenizer = PersianTokenizer(min_freq=min_count).fit(docs)
            graph = build_graph_lm_graph(
                docs,
                tokenizer,
                window_size=max(2, window_size),
                min_count=min_count,
                weighting="distance",
                top_k=12,
                directed=False,
                graph_scope="sentence",
                graph_relations=graph_relations,
                semantic_top_k=4,
                topic_top_k=6,
            )
            graph_payload = graph_lm_to_payload(graph, config=DEFAULT_CONFIG)
            top_nodes = top_graph_lm_nodes(graph, top_k=10)
            method = "Graph-LM multi-relation graph over Persian tokens"
            tokenizer_vocab_size = tokenizer.vocab_size
        relations = important_relations(graph_payload)
        reasoning_path = relation_paths_from_edges(relations)
        if not reasoning_path:
            reasoning_path = relation_path_from_nodes(top_nodes)
        return standard_response(
            task=task,
            summary="Persian text converted into a Rakhshai knowledge graph.",
            keywords=keyword_scores(
                [token for tokens in token_lists for token in tokens],
                top_k=10,
            ),
            graph=graph_payload,
            explanation={
                "top_nodes": top_nodes,
                "important_relations": relations,
                "reasoning_path": reasoning_path,
                "method": method,
            },
            artifacts={
                "graph_id": new_artifact_id("graph"),
                "run_id": new_artifact_id("run"),
                "storage": "in_response",
                "graph_type": graph_type,
                "tokenizer_vocab_size": tokenizer_vocab_size,
            },
        )
    except Exception as exc:
        return error_response(task, exc)


def rakhshai_graph_summarize(text: str, top_k: int = 3) -> dict[str, Any]:
    """Summarize Persian text using graph-based sentence ranking."""

    task = "graph_summarization"
    try:
        RATE_LIMITER.check()
        cleaned = validate_text(text)
        top_k = validate_top_k(top_k, maximum=10)
        summary = textrank_summarise(cleaned, top_k=top_k)
        graph_result = rakhshai_build_knowledge_graph(text=cleaned)
        graph_payload = graph_result.get("graph", {"nodes": [], "edges": []})
        top_nodes = graph_result.get("explanation", {}).get("top_nodes", [])
        relations = important_relations(graph_payload)
        reasoning_path = relation_paths_from_edges(relations)
        if not reasoning_path:
            reasoning_path = relation_path_from_nodes(top_nodes)
        return standard_response(
            task=task,
            summary=summary,
            keywords=graph_result.get("keywords", []),
            graph=graph_payload,
            explanation={
                "top_nodes": top_nodes,
                "important_relations": relations,
                "reasoning_path": reasoning_path,
                "method": "TextRank sentence graph plus token-level graph evidence",
            },
            artifacts={
                "run_id": new_artifact_id("run"),
                "source_graph_id": graph_result.get("artifacts", {}).get("graph_id"),
            },
        )
    except Exception as exc:
        return error_response(task, exc)

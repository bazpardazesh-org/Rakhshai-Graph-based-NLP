"""Shared output builders for Rakhshai MCP tools."""

from __future__ import annotations

from typing import Any


def standard_response(
    *,
    task: str,
    input_language: str = "fa",
    summary: str = "",
    keywords: list[dict[str, Any]] | None = None,
    entities: list[dict[str, Any]] | None = None,
    graph: dict[str, Any] | None = None,
    explanation: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical MCP response envelope."""

    return {
        "status": "success",
        "task": task,
        "input_language": input_language,
        "summary": summary,
        "keywords": keywords or [],
        "entities": entities or [],
        "graph": graph or {"nodes": [], "edges": []},
        "explanation": explanation
        or {"top_nodes": [], "important_relations": [], "reasoning_path": []},
        "artifacts": artifacts or {},
    }


def error_response(task: str, error: Exception) -> dict[str, Any]:
    """Return a controlled error payload for MCP clients."""

    return {
        "status": "error",
        "task": task,
        "input_language": "fa",
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
        "summary": "",
        "keywords": [],
        "entities": [],
        "graph": {"nodes": [], "edges": []},
        "explanation": {
            "top_nodes": [],
            "important_relations": [],
            "reasoning_path": [],
        },
        "artifacts": {},
    }


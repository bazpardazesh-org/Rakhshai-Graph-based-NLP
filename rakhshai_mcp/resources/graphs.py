"""Read-only graph artifact resources for the MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from ..config import DEFAULT_CONFIG, MCPConfig
from ..security import resolve_whitelisted_path
from .models import _read_json


def _graph_files(config: MCPConfig = DEFAULT_CONFIG) -> list[Path]:
    files: list[Path] = []
    for root in config.allowed_graph_dirs:
        if root.exists():
            files.extend(root.rglob("graph*.pt"))
    return sorted(set(files))


def list_graphs(config: MCPConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    graphs = []
    for path in _graph_files(config):
        rel = path.relative_to(config.project_root)
        graph_id = quote(rel.as_posix(), safe="")
        graphs.append(
            {
                "graph_id": rel.as_posix(),
                "resource_id": graph_id,
                "uri": f"rakhshai://graphs/{graph_id}",
                "kind": path.stem,
                "bytes": path.stat().st_size,
            }
        )
    return {"graphs": graphs}


def get_graph_info(
    graph_id: str,
    config: MCPConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    path = resolve_whitelisted_path(
        unquote(graph_id),
        config.allowed_graph_dirs,
        base_dir=config.project_root,
    )
    if not path.exists():
        return {"status": "not_found", "graph_id": graph_id}
    run_dir = path.parent
    return {
        "status": "success",
        "graph_id": path.relative_to(config.project_root).as_posix(),
        "kind": path.stem,
        "bytes": path.stat().st_size,
        "config": _read_json(run_dir / "graph_config.json"),
        "memory_config": _read_json(run_dir / "graph_memory_config.json"),
        "metrics": _read_json(run_dir / "metrics.json"),
    }

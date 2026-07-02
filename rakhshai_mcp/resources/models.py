"""Read-only model metadata resources for the MCP server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from ..config import DEFAULT_CONFIG, MCPConfig
from ..security import resolve_whitelisted_path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {"value": data}


def _model_dirs(config: MCPConfig = DEFAULT_CONFIG) -> list[Path]:
    dirs: list[Path] = []
    for root in config.allowed_model_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and (
                (path / "model.pt").exists() or (path / "config.json").exists()
            ):
                dirs.append(path)
    return sorted(set(dirs))


def list_models(config: MCPConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """List available model-like run directories without loading checkpoints."""

    models = []
    for path in _model_dirs(config):
        rel = path.relative_to(config.project_root)
        model_id = quote(rel.as_posix(), safe="")
        models.append(
            {
                "name": rel.as_posix(),
                "model_id": model_id,
                "uri": f"rakhshai://models/{model_id}",
                "has_model": (path / "model.pt").exists(),
                "has_config": (path / "config.json").exists(),
                "has_metrics": (path / "metrics.json").exists(),
            }
        )
    return {"models": models}


def get_model_info(
    model_name: str,
    config: MCPConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Return safe JSON metadata for a whitelisted model directory."""

    path = resolve_whitelisted_path(
        unquote(model_name),
        config.allowed_model_dirs,
        base_dir=config.project_root,
    )
    if not path.exists() or not path.is_dir():
        return {"status": "not_found", "model": model_name}
    return {
        "status": "success",
        "name": path.relative_to(config.project_root).as_posix(),
        "config": _read_json(path / "config.json"),
        "generation_config": _read_json(path / "generation_config.json"),
        "graph_config": _read_json(path / "graph_config.json"),
        "graph_memory_config": _read_json(path / "graph_memory_config.json"),
        "metrics": _read_json(path / "metrics.json"),
        "artifacts": {
            "model": (path / "model.pt").exists(),
            "graph": (path / "graph.pt").exists(),
            "graph_memory": (path / "graph_memory.pt").exists(),
            "tokenizer": (path / "tokenizer.json").exists(),
        },
    }

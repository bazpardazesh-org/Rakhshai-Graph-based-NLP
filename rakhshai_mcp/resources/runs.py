"""Read-only run resources for the MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from ..config import DEFAULT_CONFIG, MCPConfig
from ..security import resolve_whitelisted_path
from .models import _read_json


def _run_dirs(config: MCPConfig = DEFAULT_CONFIG) -> list[Path]:
    runs: list[Path] = []
    for root in config.allowed_run_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_dir() and (
                (path / "metrics.json").exists()
                or (path / "training_state.pt").exists()
                or (path / "model.pt").exists()
            ):
                runs.append(path)
    return sorted(set(runs))


def list_runs(config: MCPConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    runs = []
    for path in _run_dirs(config):
        rel = path.relative_to(config.project_root)
        run_id = quote(rel.as_posix(), safe="")
        runs.append(
            {
                "run_id": rel.as_posix(),
                "resource_id": run_id,
                "uri": f"rakhshai://runs/{run_id}",
                "has_metrics": (path / "metrics.json").exists(),
                "has_training_state": (path / "training_state.pt").exists(),
                "has_model": (path / "model.pt").exists(),
            }
        )
    return {"runs": runs}


def get_run_metrics(
    run_id: str,
    config: MCPConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    path = resolve_whitelisted_path(
        unquote(run_id),
        config.allowed_run_dirs,
        base_dir=config.project_root,
    )
    if not path.exists() or not path.is_dir():
        return {"status": "not_found", "run_id": run_id}
    return {
        "status": "success",
        "run_id": path.relative_to(config.project_root).as_posix(),
        "metrics": _read_json(path / "metrics.json"),
        "config": _read_json(path / "config.json"),
        "graph_config": _read_json(path / "graph_config.json"),
        "generation_config": _read_json(path / "generation_config.json"),
    }

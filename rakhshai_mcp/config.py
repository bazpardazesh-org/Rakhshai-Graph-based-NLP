"""Configuration for the Rakhshai MCP integration layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _project_root() -> Path:
    raw = os.environ.get("RAKHSHAI_MCP_ROOT") or os.environ.get("RAKHSHAI_ROOT")
    return Path(raw).expanduser().resolve() if raw else Path.cwd().resolve()


@dataclass(frozen=True)
class MCPConfig:
    """Runtime limits and whitelisted read locations for the MCP adapter."""

    name: str = "Rakhshai Graph Intelligence MCP Server"
    project_root: Path = _project_root()
    max_text_chars: int = _env_int("RAKHSHAI_MCP_MAX_TEXT_CHARS", 20_000)
    max_documents: int = _env_int("RAKHSHAI_MCP_MAX_DOCUMENTS", 50)
    max_document_chars: int = _env_int("RAKHSHAI_MCP_MAX_DOCUMENT_CHARS", 10_000)
    max_nodes: int = _env_int("RAKHSHAI_MCP_MAX_NODES", 250)
    max_edges: int = _env_int("RAKHSHAI_MCP_MAX_EDGES", 1_000)
    rate_limit_per_minute: int = _env_int("RAKHSHAI_MCP_RATE_LIMIT", 60)

    @property
    def allowed_model_dirs(self) -> tuple[Path, ...]:
        return (self.project_root / "runs", self.project_root / "models")

    @property
    def allowed_graph_dirs(self) -> tuple[Path, ...]:
        return (self.project_root / "runs", self.project_root / "graphs")

    @property
    def allowed_run_dirs(self) -> tuple[Path, ...]:
        return (self.project_root / "runs",)


DEFAULT_CONFIG = MCPConfig()


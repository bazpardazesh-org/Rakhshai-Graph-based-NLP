"""Safety checks for the Rakhshai MCP adapter."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_CONFIG, MCPConfig


class MCPInputError(ValueError):
    """Raised when an MCP request violates adapter input limits."""


class RateLimiter:
    """Small in-process sliding-window rate limiter for local MCP use."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: deque[float] = deque()

    def check(self) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
        if len(self._events) >= self.limit:
            raise MCPInputError("rate limit exceeded")
        self._events.append(now)


RATE_LIMITER = RateLimiter(DEFAULT_CONFIG.rate_limit_per_minute)


def validate_text(text: str, *, config: MCPConfig = DEFAULT_CONFIG) -> str:
    """Validate a single Persian text payload."""

    if not isinstance(text, str):
        raise MCPInputError("text must be a string")
    cleaned = text.strip()
    if not cleaned:
        raise MCPInputError("text must not be empty")
    if len(cleaned) > config.max_text_chars:
        raise MCPInputError(
            f"text is too large; max {config.max_text_chars} characters"
        )
    return cleaned


def validate_documents(
    documents: Iterable[str],
    *,
    config: MCPConfig = DEFAULT_CONFIG,
) -> list[str]:
    """Validate a bounded list of documents."""

    if isinstance(documents, str):
        raise MCPInputError("documents must be a list of strings")
    cleaned = [validate_text(doc, config=config) for doc in documents]
    if not cleaned:
        raise MCPInputError("documents must not be empty")
    if len(cleaned) > config.max_documents:
        raise MCPInputError(
            f"too many documents; max {config.max_documents} documents"
        )
    for doc in cleaned:
        if len(doc) > config.max_document_chars:
            raise MCPInputError(
                f"document is too large; max {config.max_document_chars} characters"
            )
    return cleaned


def validate_top_k(value: int, *, minimum: int = 1, maximum: int = 50) -> int:
    if not isinstance(value, int):
        raise MCPInputError("top_k must be an integer")
    if value < minimum or value > maximum:
        raise MCPInputError(f"top_k must be between {minimum} and {maximum}")
    return value


def validate_window_size(value: int, *, maximum: int = 20) -> int:
    if not isinstance(value, int):
        raise MCPInputError("window_size must be an integer")
    if value < 1 or value > maximum:
        raise MCPInputError(f"window_size must be between 1 and {maximum}")
    return value


def resolve_whitelisted_path(
    path: str | Path,
    roots: Iterable[Path],
    *,
    base_dir: Path = DEFAULT_CONFIG.project_root,
) -> Path:
    """Resolve a path only if it is inside one of the whitelisted roots."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    candidate = candidate.resolve()
    allowed_roots = [root.expanduser().resolve() for root in roots]
    if not any(
        candidate == root or root in candidate.parents
        for root in allowed_roots
    ):
        raise MCPInputError("path is outside the MCP whitelist")
    return candidate

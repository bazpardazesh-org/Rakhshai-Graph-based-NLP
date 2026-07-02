"""MCP integration layer for Rakhshai Graph-based NLP.

The MCP package intentionally stays outside the core graph-NLP package.  It
adapts stable Rakhshai APIs into agent-facing tools, resources and prompts
without changing the core architecture.
"""

from __future__ import annotations

from .config import MCPConfig

__all__ = ["MCPConfig"]


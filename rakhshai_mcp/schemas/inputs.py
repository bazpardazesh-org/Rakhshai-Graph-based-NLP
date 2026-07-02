"""Typed input shapes for the Rakhshai MCP adapter."""

from __future__ import annotations

from typing import TypedDict


class TextInput(TypedDict):
    text: str


class DocumentsInput(TypedDict, total=False):
    text: str
    documents: list[str]
    window_size: int
    min_count: int


class GenerationInput(TypedDict, total=False):
    prompt: str
    memory_texts: list[str]
    top_k: int


"""Reusable prompts exposed by the Rakhshai MCP server."""

from __future__ import annotations

from .graph_reasoning import graph_reasoning_prompt
from .persian_analysis import (
    explainable_nlp_prompt,
    graph_memory_generation_prompt,
    model_comparison_prompt,
    persian_text_analysis_prompt,
    research_report_prompt,
)

__all__ = [
    "explainable_nlp_prompt",
    "graph_memory_generation_prompt",
    "graph_reasoning_prompt",
    "model_comparison_prompt",
    "persian_text_analysis_prompt",
    "research_report_prompt",
]


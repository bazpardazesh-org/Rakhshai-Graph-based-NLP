"""Graph reasoning prompt templates for MCP clients."""

from __future__ import annotations


def graph_reasoning_prompt(text: str, question: str) -> str:
    """Guide an agent from Persian text to graph-backed reasoning."""

    return (
        "مسیر زیر را دنبال کن: Persian Text -> Knowledge Graph -> "
        "Graph Reasoning -> Explainable Output. ابتدا گراف را با "
        "`rakhshai_build_knowledge_graph` بساز، بعد با تکیه "
        "بر گره‌های مرکزی "
        "و رابطه‌های مهم به پرسش پاسخ بده.\n\n"
        f"متن:\n{text}\n\nپرسش:\n{question}"
    )

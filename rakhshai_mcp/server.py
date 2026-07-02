"""MCP server wiring for Rakhshai Graph Intelligence."""

from __future__ import annotations

import json
import os
from typing import Any

from .config import DEFAULT_CONFIG
from .prompts import (
    explainable_nlp_prompt,
    graph_memory_generation_prompt,
    graph_reasoning_prompt,
    model_comparison_prompt,
    persian_text_analysis_prompt,
    research_report_prompt,
)
from .resources import (
    get_graph_info,
    get_model_info,
    get_run_metrics,
    list_graphs,
    list_models,
    list_runs,
)
from .tools import (
    rakhshai_analyze_persian_text,
    rakhshai_build_knowledge_graph,
    rakhshai_explain_result,
    rakhshai_graph_memory_generate,
    rakhshai_graph_summarize,
    rakhshai_optimize_persian_prompt,
)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def create_server() -> Any:
    """Create a FastMCP server with Rakhshai tools, resources and prompts."""

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "Install the MCP extra first: pip install -e '.[mcp]'"
        ) from exc

    mcp = FastMCP(DEFAULT_CONFIG.name, json_response=True)

    mcp.tool()(rakhshai_analyze_persian_text)
    mcp.tool()(rakhshai_build_knowledge_graph)
    mcp.tool()(rakhshai_graph_summarize)
    mcp.tool()(rakhshai_graph_memory_generate)
    mcp.tool()(rakhshai_explain_result)
    mcp.tool()(rakhshai_optimize_persian_prompt)

    @mcp.resource("rakhshai://models")
    def rakhshai_models() -> str:
        """List safe model metadata resources."""

        return _json(list_models())

    @mcp.resource("rakhshai://models/{model_name}")
    def rakhshai_model(model_name: str) -> str:
        """Read safe metadata for one model directory."""

        return _json(get_model_info(model_name))

    @mcp.resource("rakhshai://graphs")
    def rakhshai_graphs() -> str:
        """List safe graph artifact resources."""

        return _json(list_graphs())

    @mcp.resource("rakhshai://graphs/{graph_id}")
    def rakhshai_graph(graph_id: str) -> str:
        """Read safe metadata for one graph artifact."""

        return _json(get_graph_info(graph_id))

    @mcp.resource("rakhshai://runs")
    def rakhshai_runs() -> str:
        """List safe run resources."""

        return _json(list_runs())

    @mcp.resource("rakhshai://runs/{run_id}/metrics")
    def rakhshai_run_metrics(run_id: str) -> str:
        """Read safe metrics for one run."""

        return _json(get_run_metrics(run_id))

    @mcp.resource("rakhshai://docs/api")
    def rakhshai_api_docs() -> str:
        """Describe the stable API surface exposed through MCP."""

        return _json(
            {
                "message": (
                    "MCP exposes Rakhshai's stable graph-NLP APIs as tools; "
                    "the core architecture remains in rakhshai_graph_nlp."
                ),
                "tools": [
                    "rakhshai_analyze_persian_text",
                    "rakhshai_build_knowledge_graph",
                    "rakhshai_graph_summarize",
                    "rakhshai_graph_memory_generate",
                    "rakhshai_explain_result",
                    "rakhshai_optimize_persian_prompt",
                ],
            }
        )

    @mcp.resource("rakhshai://docs/examples")
    def rakhshai_examples() -> str:
        """Return small MCP usage examples."""

        return _json(
            {
                "demo": (
                    "From Persian Text to Explainable Graph Intelligence: "
                    "analyze text, build graph, summarize, retrieve memory, "
                    "then explain top nodes and relations."
                )
            }
        )

    mcp.prompt()(persian_text_analysis_prompt)
    mcp.prompt()(graph_reasoning_prompt)
    mcp.prompt()(graph_memory_generation_prompt)
    mcp.prompt()(research_report_prompt)
    mcp.prompt()(model_comparison_prompt)
    mcp.prompt()(explainable_nlp_prompt)

    return mcp


def main() -> None:
    """Run the MCP server over stdio unless another transport is requested."""

    server = create_server()
    transport = os.environ.get("RAKHSHAI_MCP_TRANSPORT", "stdio")
    server.run(transport=transport)


if __name__ == "__main__":
    main()

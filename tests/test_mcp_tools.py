from rakhshai_mcp.resources.graphs import list_graphs
from rakhshai_mcp.resources.models import list_models
from rakhshai_mcp.resources.runs import list_runs
from rakhshai_mcp.tools import (
    rakhshai_analyze_persian_text,
    rakhshai_build_knowledge_graph,
    rakhshai_explain_result,
    rakhshai_graph_memory_generate,
    rakhshai_graph_summarize,
    rakhshai_optimize_persian_prompt,
)


SAMPLE_TEXT = (
    "در آینه، سایه‌ام از من قدیمی‌تر بود. "
    "نامم از دهان پنجره به باران می‌ریخت. "
    "چراغ خاکستر می‌دید و راه را نشان نمی‌داد. "
    "رود گذشت اما تشنگی در مشت‌هایم بود. "
    "باد کلیدی زنگ‌زده را در سکوت چرخاند."
)


def test_mcp_analyze_persian_text_returns_standard_envelope():
    result = rakhshai_analyze_persian_text(SAMPLE_TEXT, max_keywords=5)

    assert result["status"] == "success"
    assert result["task"] == "persian_text_analysis"
    assert result["input_language"] == "fa"
    assert result["keywords"]
    assert result["graph"]["nodes"]
    assert "top_nodes" in result["explanation"]


def test_mcp_build_knowledge_graph_accepts_documents():
    result = rakhshai_build_knowledge_graph(
        documents=[
            "آینه سایه و نام را نشان می‌دهد.",
            "رود می‌گذرد اما تشنگی و کلید می‌مانند.",
        ],
        window_size=2,
    )

    assert result["status"] == "success"
    assert result["task"] == "knowledge_graph_building"
    assert result["graph"]["metrics"]["node_count"] > 0
    assert result["artifacts"]["graph_type"] == "graph_lm"
    assert len(result["graph"]["metrics"]["enabled_relations"]) > 1
    assert "را" not in {node["label"] for node in result["explanation"]["top_nodes"]}
    assert result["artifacts"]["storage"] == "in_response"


def test_mcp_graph_summarize_and_explain_result():
    summary = rakhshai_graph_summarize(SAMPLE_TEXT, top_k=1)
    explanation = rakhshai_explain_result(SAMPLE_TEXT, top_k=3)

    assert summary["status"] == "success"
    assert summary["summary"]
    assert explanation["status"] == "success"
    assert explanation["explanation"]["important_relations"]


def test_mcp_graph_memory_generate_retrieves_evidence():
    result = rakhshai_graph_memory_generate(
        "رابطه آینه، سایه، رود و کلید در شعر چیست؟",
        memory_texts=[
            "آینه با سایه و نام پیوند دارد.",
            "رود از کنار تشنگی می‌گذرد.",
            "کلید خانه را یادآوری می‌کند.",
        ],
    )

    assert result["status"] == "success"
    assert result["summary"]
    assert result["explanation"]["graph_memory_report"]["top_nodes"]


def test_mcp_optimize_persian_prompt_adds_graph_grounding():
    result = rakhshai_optimize_persian_prompt(
        "رابطه آینه، سایه، رود و کلید را بگو.",
        SAMPLE_TEXT,
    )

    prompt = result["artifacts"]["optimized_prompt"]
    assert result["status"] == "success"
    assert "وظیفه بازنویسی‌شده" in prompt
    assert "شواهد گرافی Rakhshai" in prompt
    assert result["explanation"]["top_nodes"]


def test_mcp_resources_are_read_only_metadata_payloads():
    assert "models" in list_models()
    assert "graphs" in list_graphs()
    assert "runs" in list_runs()

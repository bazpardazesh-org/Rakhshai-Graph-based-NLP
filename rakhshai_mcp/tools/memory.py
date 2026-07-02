"""MCP tool for lightweight Graph Memory-style generation."""

from __future__ import annotations

from typing import Any

from rakhshai_graph_nlp.lm.graph_memory import (
    GraphMemoryArtifact,
    GraphMemoryConfig,
)
from rakhshai_graph_nlp.lm.model import GraphCausalLM
from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer
from rakhshai_graph_nlp.tasks.summarization import split_sentences

from ..config import DEFAULT_CONFIG
from ..schemas.outputs import error_response, standard_response
from ..security import (
    RATE_LIMITER,
    resolve_whitelisted_path,
    validate_documents,
    validate_text,
    validate_top_k,
)
from .common import (
    is_meaningful_label,
    important_relations,
    keyword_scores,
    new_artifact_id,
    relation_paths_from_edges,
    relation_path_from_nodes,
    tokenize_documents,
)
from .graph import rakhshai_build_knowledge_graph


def _retrieve_memory_sentences(
    prompt: str,
    memory_texts: list[str],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    prompt_tokens = {
        token
        for token in tokenize_documents([prompt])[0]
        if is_meaningful_label(token)
    }
    scored: list[dict[str, Any]] = []
    for doc_id, text in enumerate(memory_texts):
        for sentence in split_sentences(text):
            sentence_tokens = {
                token
                for token in tokenize_documents([sentence])[0]
                if is_meaningful_label(token)
            }
            overlap = prompt_tokens & sentence_tokens
            if overlap:
                scored.append(
                    {
                        "doc_id": doc_id,
                        "text": sentence,
                        "score": round(len(overlap) / max(len(prompt_tokens), 1), 6),
                        "matched_nodes": sorted(overlap),
                    }
                )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _graph_memory_report(
    prompt: str,
    memory_texts: list[str],
    *,
    top_k: int,
) -> dict[str, Any]:
    tokenizer = PersianTokenizer(min_freq=1).fit([prompt, *memory_texts])
    memory = GraphMemoryArtifact.from_corpus(
        memory_texts,
        tokenizer,
        {
            "window_size": 4,
            "min_count": 1,
            "weighting": "distance",
            "graph_scope": "sentence",
            "enabled_relations": [
                "cooccurrence",
                "pmi",
                "dependency",
                "stem",
                "subword",
                "word_document",
                "topic_document",
            ],
            "topic_top_k": 6,
        },
    )
    context = memory.retrieve(
        prompt,
        tokenizer,
        config=GraphMemoryConfig(
            top_k_nodes=max(8, top_k * 4),
            depth=1,
            max_edges=64,
        ),
    )
    return _sanitize_memory_report(context.report)


def _sanitize_memory_report(report: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(report)
    prompt_tokens = [
        str(token)
        for token in cleaned.pop("prompt_tokens", [])
        if is_meaningful_label(str(token))
    ]
    top_nodes = [
        str(node)
        for node in cleaned.get("top_nodes", [])
        if is_meaningful_label(str(node))
    ]
    cleaned["prompt_concepts"] = prompt_tokens[:12]
    cleaned["top_nodes"] = top_nodes[:10]
    return cleaned


def _generate_from_checkpoint(
    prompt: str,
    model_dir: str,
    *,
    max_new_tokens: int,
) -> tuple[str, dict[str, Any]]:
    model_path = resolve_whitelisted_path(
        model_dir,
        DEFAULT_CONFIG.allowed_model_dirs,
        base_dir=DEFAULT_CONFIG.project_root,
    )
    model, tokenizer, generation_config, _graph_config = GraphCausalLM.from_pretrained(
        model_path,
        map_location="cpu",
    )
    graph_data, token_node_ids = GraphCausalLM.load_graph_artifacts(
        model_path,
        map_location="cpu",
    )
    graph_memory, graph_memory_config = GraphMemoryArtifact.load(
        model_path,
        map_location="cpu",
    )
    generated = model.generate(
        prompt,
        tokenizer,
        graph_data=graph_data,
        token_node_ids=token_node_ids,
        graph_memory=graph_memory,
        graph_memory_config=graph_memory_config,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )
    report: dict[str, Any] = {}
    if graph_memory is not None:
        report = graph_memory.retrieve(
            prompt,
            tokenizer,
            config=graph_memory_config,
        ).report
    return generated, _sanitize_memory_report(report)


def rakhshai_graph_memory_generate(
    prompt: str,
    memory_texts: list[str] | None = None,
    top_k: int = 3,
    model_dir: str | None = None,
    max_new_tokens: int = 80,
) -> dict[str, Any]:
    """Generate an explainable response from prompt-linked graph memory evidence."""

    task = "graph_memory_generation"
    try:
        RATE_LIMITER.check()
        cleaned_prompt = validate_text(prompt)
        top_k = validate_top_k(top_k, maximum=10)
        memory = validate_documents(memory_texts or [cleaned_prompt])
        max_new_tokens = validate_top_k(max_new_tokens, minimum=1, maximum=256)
        retrieved = _retrieve_memory_sentences(cleaned_prompt, memory, top_k=top_k)
        memory_report = _graph_memory_report(
            cleaned_prompt,
            memory,
            top_k=top_k,
        )
        corpus = [cleaned_prompt, *memory]
        graph_result = rakhshai_build_knowledge_graph(documents=corpus)
        graph_payload = graph_result.get("graph", {"nodes": [], "edges": []})
        top_nodes = graph_result.get("explanation", {}).get("top_nodes", [])
        relations = important_relations(graph_payload)
        reasoning_path = relation_paths_from_edges(relations)
        if not reasoning_path:
            reasoning_path = relation_path_from_nodes(top_nodes)
        all_tokens = [
            token
            for tokens in tokenize_documents(corpus)
            for token in tokens
        ]
        concepts = [item["text"] for item in keyword_scores(all_tokens, top_k=top_k)]
        evidence_text = " ".join(item["text"] for item in retrieved)
        generated = " ".join(
            [
                "بر اساس حافظه گرافی،",
                "مفاهیم محوری عبارت‌اند از:",
                "، ".join(concepts),
            ]
        )
        if evidence_text:
            generated += f". شواهد مرتبط: {evidence_text}"
        checkpoint_report: dict[str, Any] = {}
        if model_dir:
            generated, checkpoint_report = _generate_from_checkpoint(
                cleaned_prompt,
                model_dir,
                max_new_tokens=max_new_tokens,
            )
        return standard_response(
            task=task,
            summary=generated,
            keywords=keyword_scores(all_tokens, top_k=top_k),
            graph=graph_payload,
            explanation={
                "top_nodes": top_nodes,
                "important_relations": relations,
                "reasoning_path": reasoning_path,
                "retrieved_memory": retrieved,
                "graph_memory_report": memory_report,
                "checkpoint_memory_report": checkpoint_report,
                "method": (
                    "GraphMemoryArtifact retrieval"
                    + (" + GraphCausalLM checkpoint generation" if model_dir else "")
                ),
            },
            artifacts={
                "memory_session_id": new_artifact_id("memory"),
                "run_id": new_artifact_id("run"),
                "source_graph_id": graph_result.get("artifacts", {}).get("graph_id"),
            },
        )
    except Exception as exc:
        return error_response(task, exc)

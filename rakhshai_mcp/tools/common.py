"""Shared graph-NLP helpers for MCP tools."""

from __future__ import annotations

import re
import uuid
from collections import Counter
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from rakhshai_graph_nlp.features.tokenizer import tokenize
from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.lm.graph_builder import GraphLMGraph

from ..config import DEFAULT_CONFIG, MCPConfig

PERSIAN_STOPWORDS = {
    "",
    "،",
    ".",
    "؛",
    ":",
    "؟",
    "!",
    "از",
    "اگر",
    "اما",
    "اند",
    "است",
    "باشد",
    "باشند",
    "این",
    "آن",
    "او",
    "من",
    "تو",
    "با",
    "به",
    "برای",
    "بر",
    "پس",
    "سپس",
    "چو",
    "چون",
    "کدام",
    "ز",
    "دگر",
    "تا",
    "در",
    "را",
    "شد",
    "شود",
    "شده",
    "کرده",
    "کند",
    "کنند",
    "قابل",
    "نماند",
    "تواند",
    "توانند",
    "که",
    "کرد",
    "می",
    "می‌کند",
    "می‌کنند",
    "می‌تواند",
    "می‌توانند",
    "نمی‌کند",
    "گفتم",
    "بود",
    "ام",
    "ات",
    "اش",
    "مان",
    "تان",
    "شان",
    "هم",
    "نیز",
    "و",
    "یا",
    "یک",
    "ها",
    "های",
    "هایی",
    "تر",
    "ترین",
}
NON_CONCEPT_NODE_TYPES = {
    "doc",
    "document",
    "sentence",
    "topic",
    "context",
}

_ENTITY_HINT_RE = re.compile(
    r"(?:شرکت|سازمان|دانشگاه|وزارت|"
    r"پژوهشگاه|مرکز|دکتر|مهندس)"
    r"\s+([\u0600-\u06FF\u200c]+(?:\s+[\u0600-\u06FF\u200c]+){0,2})"
)


def new_artifact_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def tokenize_documents(documents: Iterable[str]) -> list[list[str]]:
    return [tokenize(document) for document in documents]


def keyword_scores(tokens: Iterable[str], *, top_k: int = 10) -> list[dict[str, Any]]:
    counts = Counter(
        token
        for token in tokens
        if len(token) > 1 and token not in PERSIAN_STOPWORDS and not token.isdigit()
    )
    total = sum(counts.values()) or 1
    return [
        {"text": token, "score": round(count / total, 6), "count": count}
        for token, count in counts.most_common(top_k)
    ]


def is_meaningful_label(label: str, node_type: str | None = None) -> bool:
    """Return whether a graph node should be treated as conceptual evidence."""

    cleaned = label.strip().strip("،.;:؟!()[]{}\"'")
    if node_type and node_type.lower() in NON_CONCEPT_NODE_TYPES:
        return False
    if (
        cleaned in PERSIAN_STOPWORDS
        or cleaned.startswith("doc_")
        or cleaned.startswith("document:")
    ):
        return False
    if cleaned.startswith("<") and cleaned.endswith(">"):
        return False
    if cleaned.startswith("##"):
        return False
    if cleaned.isdigit() or len(cleaned) < 2:
        return False
    return True


def extract_entities(text: str, *, limit: int = 12) -> list[dict[str, Any]]:
    """Extract simple Persian named-entity candidates with conservative rules."""

    seen: set[str] = set()
    entities: list[dict[str, Any]] = []
    for match in _ENTITY_HINT_RE.finditer(text):
        entity = match.group(0).strip()
        if entity in seen:
            continue
        seen.add(entity)
        entities.append(
            {
                "text": entity,
                "type": "entity_candidate",
                "confidence": 0.45,
                "method": "lexical_hint",
            }
        )
        if len(entities) >= limit:
            break
    return entities


def top_graph_nodes(graph: Graph, *, top_k: int = 10) -> list[dict[str, Any]]:
    if len(graph.nodes) == 0:
        return []
    degrees = np.asarray(graph.adjacency).sum(axis=1)
    order = [
        int(idx)
        for idx in np.argsort(degrees)[::-1]
        if is_meaningful_label(
            str(graph.nodes[int(idx)]),
            None if graph.node_types is None else graph.node_types[int(idx)],
        )
    ][:top_k]
    max_degree = float(degrees[order[0]]) if len(order) else 0.0
    normalizer = max(max_degree, 1e-12)
    node_types = graph.node_types or [None] * len(graph.nodes)
    return [
        {
            "id": int(idx),
            "label": str(graph.nodes[int(idx)]),
            "type": node_types[int(idx)],
            "score": round(float(degrees[int(idx)] / normalizer), 6),
            "weighted_degree": round(float(degrees[int(idx)]), 6),
        }
        for idx in order
    ]


def top_graph_lm_nodes(
    graph: GraphLMGraph,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    if not graph.nodes:
        return []
    degrees = np.zeros((len(graph.nodes),), dtype=float)
    if graph.edge_index.size:
        for edge_idx, (src, dst) in enumerate(graph.edge_index.T):
            weight = float(graph.edge_weight[edge_idx])
            degrees[int(src)] += weight
            degrees[int(dst)] += weight
    order = [
        int(idx)
        for idx in np.argsort(degrees)[::-1]
        if is_meaningful_label(
            str(graph.nodes[int(idx)]),
            None if graph.node_types is None else graph.node_types[int(idx)],
        )
    ][:top_k]
    max_degree = float(degrees[order[0]]) if order else 0.0
    normalizer = max(max_degree, 1e-12)
    node_types = graph.node_types or [None] * len(graph.nodes)
    return [
        {
            "id": idx,
            "label": str(graph.nodes[idx]),
            "type": node_types[idx],
            "score": round(float(degrees[idx] / normalizer), 6),
            "weighted_degree": round(float(degrees[idx]), 6),
        }
        for idx in order
    ]


def graph_to_payload(
    graph: Graph,
    *,
    config: MCPConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Serialize a Rakhshai graph into a bounded MCP-safe payload."""

    node_types = graph.node_types or [None] * len(graph.nodes)
    nodes = [
        {
            "id": idx,
            "label": str(label),
            "type": node_types[idx],
        }
        for idx, label in enumerate(graph.nodes[: config.max_nodes])
    ]
    edges: list[dict[str, Any]] = []
    for source, target, weight in graph.to_edge_list():
        try:
            source_idx = list(graph.nodes).index(source)
            target_idx = list(graph.nodes).index(target)
        except ValueError:
            continue
        if source_idx >= config.max_nodes or target_idx >= config.max_nodes:
            continue
        edges.append(
            {
                "source": source_idx,
                "target": target_idx,
                "source_label": str(source),
                "target_label": str(target),
                "weight": round(float(weight), 6),
                "relation": "cooccurrence",
            }
        )
    edges.sort(key=lambda edge: abs(float(edge["weight"])), reverse=True)
    return {
        "nodes": nodes,
        "edges": edges[: config.max_edges],
        "metrics": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.to_edge_list()),
            "directed": bool(graph.directed),
        },
    }


def graph_lm_to_payload(
    graph: GraphLMGraph,
    *,
    config: MCPConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Serialize a Graph-LM multi-relation graph into an MCP-safe payload."""

    edge_types = graph.graph_config.get("edge_types", {})
    relation_names: Mapping[int, str] = {}
    if isinstance(edge_types, Mapping):
        relation_names = {
            int(value): str(name)
            for name, value in edge_types.items()
        }
    node_types = graph.node_types or [None] * len(graph.nodes)
    nodes = [
        {
            "id": idx,
            "label": str(label),
            "type": node_types[idx],
        }
        for idx, label in enumerate(graph.nodes[: config.max_nodes])
    ]
    edges: list[dict[str, Any]] = []
    for edge_idx, (source, target) in enumerate(graph.edge_index.T):
        source_idx = int(source)
        target_idx = int(target)
        if source_idx >= config.max_nodes or target_idx >= config.max_nodes:
            continue
        relation_id = (
            int(graph.edge_type[edge_idx])
            if graph.edge_type is not None
            else 0
        )
        relation = relation_names.get(relation_id, "relation")
        edges.append(
            {
                "source": source_idx,
                "target": target_idx,
                "source_label": str(graph.nodes[source_idx]),
                "target_label": str(graph.nodes[target_idx]),
                "weight": round(float(graph.edge_weight[edge_idx]), 6),
                "relation": relation,
            }
        )
    edges.sort(key=lambda edge: abs(float(edge["weight"])), reverse=True)
    return {
        "nodes": nodes,
        "edges": edges[: config.max_edges],
        "metrics": {
            "node_count": len(graph.nodes),
            "edge_count": int(graph.edge_index.shape[1]),
            "directed": bool(graph.directed),
            "relation_edge_counts": graph.graph_config.get(
                "relation_edge_counts",
                {},
            ),
            "node_type_counts": graph.graph_config.get("node_type_counts", {}),
            "enabled_relations": graph.graph_config.get("enabled_relations", []),
        },
    }


def relation_path_from_nodes(
    nodes: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[str]:
    labels = [str(node["label"]) for node in nodes[:limit]]
    if len(labels) < 2:
        return labels
    return [f"{labels[idx]} -> {labels[idx + 1]}" for idx in range(len(labels) - 1)]


def important_relations(
    graph_payload: dict[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    relations = []
    for edge in graph_payload.get("edges", []):
        if not is_meaningful_label(str(edge.get("source_label", ""))):
            continue
        if not is_meaningful_label(str(edge.get("target_label", ""))):
            continue
        relations.append(edge)
        if len(relations) >= limit:
            break
    return relations


def relation_paths_from_edges(
    edges: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[str]:
    paths = []
    for edge in edges[:limit]:
        source = edge.get("source_label")
        target = edge.get("target_label")
        relation = edge.get("relation", "relation")
        if source and target:
            paths.append(f"{source} -[{relation}]-> {target}")
    return paths

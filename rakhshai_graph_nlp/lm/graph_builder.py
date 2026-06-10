"""Vocabulary and multi-relation graph builders for Graph-LM."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from ..graphs.graph import Graph
from .tokenizer import PersianTokenizer

SENTENCE_SPLIT_PATTERN = re.compile(r"[.!؟?؛\n]+")
MAX_DENSE_GRAPH_CELLS = 25_000_000
DEFAULT_GRAPH_RELATIONS = (
    "cooccurrence",
    "pmi",
    "stem",
    "subword",
    "word_document",
    "topic_document",
)
MULTI_RELATION_NAMES = {
    "cooccurrence",
    "pmi",
    "ppmi",
    "dependency",
    "stem",
    "subword",
    "semantic_similarity",
    "word_document",
    "topic_document",
}
LIGHT_VERB_HINTS = (
    "کرد",
    "شد",
    "گرفت",
    "داد",
    "داشت",
    "آورد",
    "رفت",
    "بود",
    "است",
)
STEM_SUFFIXES = (
    "هایمان",
    "هایتان",
    "هایشان",
    "هایی",
    "های",
    "ها",
    "ترین",
    "تر",
    "مان",
    "تان",
    "شان",
    "ام",
    "ات",
    "اش",
)


@dataclass
class GraphLMGraph:
    nodes: list[str]
    token_to_node: dict[int, int]
    graph_config: dict[str, object]
    edge_index: np.ndarray
    edge_weight: np.ndarray
    edge_type: np.ndarray | None = None
    node_types: list[str] | None = None
    directed: bool = False
    _graph: Graph | None = field(default=None, repr=False)

    @property
    def graph(self) -> Graph:
        """Materialise a dense compatibility graph for small LM graphs."""

        if self._graph is None:
            n = len(self.nodes)
            if n * n > MAX_DENSE_GRAPH_CELLS:
                raise ValueError(
                    "GraphLMGraph is too large to materialise as a dense Graph; "
                    "use to_pyg_data() for sparse access"
                )
            adjacency = np.zeros((n, n), dtype=np.float32)
            if self.edge_index.size:
                src = self.edge_index[0]
                dst = self.edge_index[1]
                adjacency[src, dst] = self.edge_weight
            self._graph = Graph(
                nodes=self.nodes,
                adjacency=adjacency,
                node_types=self.node_types,
                directed=self.directed,
            )
        return self._graph

    def to_pyg_data(self):
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_weight = torch.tensor(self.edge_weight, dtype=torch.float32)
        data = Data(edge_index=edge_index, num_nodes=len(self.nodes))
        data.edge_weight = edge_weight
        if self.edge_type is not None:
            data.edge_type = torch.tensor(self.edge_type, dtype=torch.long)
        return data

    def token_node_ids(self, vocab_size: int) -> torch.Tensor:
        ids = [self.token_to_node.get(i, -1) for i in range(vocab_size)]
        return torch.tensor(ids, dtype=torch.long)

    def save_config(self, path: str | Path) -> None:
        payload = {
            "token_to_node": {str(k): v for k, v in self.token_to_node.items()},
            **self.graph_config,
        }
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def _split_units(texts: Sequence[str], scope: str) -> list[str]:
    if scope == "corpus":
        return [" ".join(texts)]
    if scope == "document":
        return [text for text in texts if text.strip()]
    if scope == "sentence":
        units: list[str] = []
        for text in texts:
            units.extend(
                part.strip()
                for part in SENTENCE_SPLIT_PATTERN.split(text)
                if part.strip()
            )
        return units
    raise ValueError("graph_scope must be one of: corpus, document, sentence")


def _token_ids_for_units(
    units: Sequence[str],
    tokenizer: PersianTokenizer,
    token_to_node: dict[int, int],
) -> list[list[int]]:
    tokenised = [tokenizer.tokenize(unit) for unit in units]
    return [
        [
            tokenizer.token_to_id[token]
            for token in tokens
            if tokenizer.token_to_id.get(token) in token_to_node
        ]
        for tokens in tokenised
    ]


def _cooccurrence_edges(
    unit_ids: Sequence[Sequence[int]],
    token_to_node: dict[int, int],
    *,
    window_size: int,
    directed: bool,
) -> tuple[dict[tuple[int, int], float], Counter[int], float]:
    edges: dict[tuple[int, int], float] = {}
    counts: Counter[int] = Counter()
    total_tokens = 0
    for ids in unit_ids:
        counts.update(ids)
        total_tokens += len(ids)
        for i, src_id in enumerate(ids):
            src = token_to_node[src_id]
            right = min(len(ids), i + window_size + 1)
            for j in range(i + 1, right):
                dst_id = ids[j]
                if src_id == dst_id:
                    continue
                dst = token_to_node[dst_id]
                weight = 1.0 / (j - i)
                edges[(src, dst)] = edges.get((src, dst), 0.0) + weight
                if not directed:
                    edges[(dst, src)] = edges.get((dst, src), 0.0) + weight
    return edges, counts, float(max(1, total_tokens))


def _normalise_relations(relations: Sequence[str] | str | None) -> list[str]:
    if relations is None:
        return list(DEFAULT_GRAPH_RELATIONS)
    if isinstance(relations, str):
        parts = [part.strip() for part in relations.split(",")]
    else:
        parts = [str(part).strip() for part in relations]
    normalised: list[str] = []
    aliases = {
        "semantic": "semantic_similarity",
        "word-doc": "word_document",
        "word_document": "word_document",
        "topic-doc": "topic_document",
        "topic_document": "topic_document",
    }
    for relation in parts:
        if not relation:
            continue
        name = aliases.get(relation.lower().replace("-", "_"), relation.lower())
        if name not in MULTI_RELATION_NAMES:
            raise ValueError(
                "graph_relations must contain only: "
                + ", ".join(sorted(MULTI_RELATION_NAMES))
            )
        if name not in normalised:
            normalised.append(name)
    return normalised or list(DEFAULT_GRAPH_RELATIONS)


def _normalise_relation_weights(
    relation_weights: dict[str, float] | None,
    relations: Sequence[str],
) -> dict[str, float]:
    weights = {relation: 1.0 for relation in relations}
    if relation_weights:
        for relation, weight in relation_weights.items():
            key = relation.lower().replace("-", "_")
            if key == "semantic":
                key = "semantic_similarity"
            weights[key] = float(weight)
    return weights


def _merge_relation_edges(
    merged_edges: dict[tuple[int, int], float],
    edge_types: dict[tuple[int, int], int],
    relation_edge_counts: dict[str, int],
    relation_edges: dict[tuple[int, int], float],
    *,
    relation: str,
    relation_id: int,
    relation_weight: float,
) -> None:
    count = 0
    for edge, weight in relation_edges.items():
        scaled = float(weight) * relation_weight
        if scaled == 0:
            continue
        merged_edges[edge] = merged_edges.get(edge, 0.0) + scaled
        edge_types[edge] = relation_id
        count += 1
    relation_edge_counts[relation] = relation_edge_counts.get(relation, 0) + count


def _surface_stem(token: str) -> str:
    token = token.replace("##", "").strip("\u200c")
    for suffix in STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)].rstrip("\u200c")
    return token


def _stem_edges(
    token_to_node: dict[int, int],
    tokenizer: PersianTokenizer,
    *,
    directed: bool,
) -> dict[tuple[int, int], float]:
    groups: dict[str, list[int]] = {}
    for token_id, node_id in token_to_node.items():
        stem = _surface_stem(tokenizer.id_to_token[token_id])
        if len(stem) >= 2:
            groups.setdefault(stem, []).append(node_id)
    edges: dict[tuple[int, int], float] = {}
    for node_ids in groups.values():
        if len(node_ids) < 2:
            continue
        for i, src in enumerate(node_ids):
            for dst in node_ids[i + 1 :]:
                edges[(src, dst)] = 1.0
                if not directed:
                    edges[(dst, src)] = 1.0
    return edges


def _subword_edges(
    token_to_node: dict[int, int],
    tokenizer: PersianTokenizer,
    *,
    directed: bool,
) -> dict[tuple[int, int], float]:
    nodes_by_piece: dict[str, list[int]] = {}
    for token_id, node_id in token_to_node.items():
        token = tokenizer.id_to_token[token_id]
        piece = token.replace("##", "")
        if len(piece) >= 3:
            nodes_by_piece.setdefault(piece, []).append(node_id)
        if token.startswith("##") and len(piece) >= 2:
            nodes_by_piece.setdefault(piece[:2], []).append(node_id)
    edges: dict[tuple[int, int], float] = {}
    for node_ids in nodes_by_piece.values():
        unique_ids = sorted(set(node_ids))
        for i, src in enumerate(unique_ids):
            for dst in unique_ids[i + 1 :]:
                edges[(src, dst)] = 1.0
                if not directed:
                    edges[(dst, src)] = 1.0
    return edges


def _char_ngrams(token: str, n: int = 3) -> set[str]:
    clean = token.replace("##", "")
    if len(clean) <= n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def _semantic_similarity_edges(
    token_to_node: dict[int, int],
    tokenizer: PersianTokenizer,
    *,
    directed: bool,
    threshold: float,
    top_k: int | None,
) -> dict[tuple[int, int], float]:
    token_items = [
        (token_id, node_id, _char_ngrams(tokenizer.id_to_token[token_id]))
        for token_id, node_id in token_to_node.items()
    ]
    rows: dict[int, list[tuple[int, float]]] = {}
    for i, (_, src, src_ngrams) in enumerate(token_items):
        if not src_ngrams:
            continue
        for _, dst, dst_ngrams in token_items[i + 1 :]:
            if not dst_ngrams:
                continue
            union = src_ngrams | dst_ngrams
            score = len(src_ngrams & dst_ngrams) / max(1, len(union))
            if score >= threshold:
                rows.setdefault(src, []).append((dst, score))
                if not directed:
                    rows.setdefault(dst, []).append((src, score))
    edges: dict[tuple[int, int], float] = {}
    for src, neighbours in rows.items():
        selected = sorted(neighbours, key=lambda item: item[1], reverse=True)
        if top_k is not None and top_k > 0:
            selected = selected[:top_k]
        for dst, score in selected:
            edges[(src, dst)] = score
    return edges


def _dependency_edges(
    unit_ids: Sequence[Sequence[int]],
    token_to_node: dict[int, int],
    tokenizer: PersianTokenizer,
    *,
    directed: bool,
) -> dict[tuple[int, int], float]:
    edges: dict[tuple[int, int], float] = {}
    for ids in unit_ids:
        node_ids = [token_to_node[token_id] for token_id in ids if token_id in token_to_node]
        if len(node_ids) < 2:
            continue
        token_surfaces = [tokenizer.id_to_token[token_id] for token_id in ids]
        for idx, src in enumerate(node_ids):
            head_idx = min(idx + 1, len(node_ids) - 1)
            for lookahead in range(idx + 1, min(len(node_ids), idx + 5)):
                surface = token_surfaces[lookahead]
                if surface.startswith(("می", "نمی")) or any(
                    surface.endswith(hint) for hint in LIGHT_VERB_HINTS
                ):
                    head_idx = lookahead
                    break
            dst = node_ids[head_idx]
            if src == dst:
                continue
            edges[(src, dst)] = edges.get((src, dst), 0.0) + 1.0
            if not directed:
                edges[(dst, src)] = edges.get((dst, src), 0.0) + 1.0
    return edges


def _apply_association_weighting(
    edges: dict[tuple[int, int], float],
    counts: Counter[int],
    token_to_node: dict[int, int],
    total_tokens: float,
    *,
    weighting: str,
) -> dict[tuple[int, int], float]:
    weighting = weighting.lower()
    if weighting in {"distance", "count", "raw"}:
        return edges
    if weighting not in {"pmi", "ppmi"}:
        raise ValueError("graph_weighting must be one of: distance, count, raw, pmi, ppmi")

    weighted: dict[tuple[int, int], float] = {}
    total_edges = float(max(sum(edges.values()), 1.0))
    node_counts = np.zeros((len(token_to_node),), dtype=np.float32)
    for token_id, node_id in token_to_node.items():
        node_counts[node_id] = float(counts[token_id])

    for (i, j), weight in edges.items():
        p_ij = float(weight) / total_edges
        p_i = max(float(node_counts[i]) / total_tokens, 1e-12)
        p_j = max(float(node_counts[j]) / total_tokens, 1e-12)
        score = math.log(max(p_ij, 1e-12) / (p_i * p_j))
        if weighting == "ppmi":
            if score > 0:
                weighted[(i, j)] = score
        else:
            weighted[(i, j)] = score
    return weighted


def _prune_edges(
    edges: dict[tuple[int, int], float],
    min_edge_weight: float,
    top_k: int | None,
    *,
    directed: bool,
) -> dict[tuple[int, int], float]:
    pruned = {
        edge: weight
        for edge, weight in edges.items()
        if min_edge_weight <= 0 or weight >= min_edge_weight
    }
    if top_k is not None and top_k > 0:
        rows: dict[int, list[tuple[int, float]]] = {}
        for (src, dst), weight in pruned.items():
            rows.setdefault(src, []).append((dst, weight))
        kept: dict[tuple[int, int], float] = {}
        for src, neighbours in rows.items():
            for dst, weight in sorted(neighbours, key=lambda item: item[1], reverse=True)[:top_k]:
                kept[(src, dst)] = weight
        pruned = kept
    if not directed:
        symmetric: dict[tuple[int, int], float] = {}
        for (src, dst), weight in pruned.items():
            reciprocal = pruned.get((dst, src), weight)
            sym_weight = max(weight, reciprocal)
            symmetric[(src, dst)] = sym_weight
            symmetric[(dst, src)] = sym_weight
        pruned = symmetric
    return pruned


def _add_context_nodes(
    edges: dict[tuple[int, int], float],
    edge_types: dict[tuple[int, int], int],
    nodes: list[str],
    unit_ids: Sequence[Sequence[int]],
    token_to_node: dict[int, int],
    *,
    context_node_type: str,
    edge_weight: float,
    relation_id: int = 1,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int], list[str], list[str]]:
    node_types = ["token"] * len(nodes)
    if context_node_type == "none":
        return edges, edge_types, nodes, node_types
    if context_node_type not in {"document", "sentence"}:
        raise ValueError("context_node_type must be one of: none, document, sentence")

    context_units = [ids for ids in unit_ids if ids]
    old_n = len(nodes)
    for idx, ids in enumerate(context_units):
        context_node = old_n + idx
        nodes.append(f"{context_node_type}:{idx}")
        node_types.append(context_node_type)
        for token_id in set(ids):
            token_node = token_to_node[token_id]
            edges[(context_node, token_node)] = edge_weight
            edges[(token_node, context_node)] = edge_weight
            edge_types[(context_node, token_node)] = relation_id
            edge_types[(token_node, context_node)] = relation_id
    return edges, edge_types, nodes, node_types


def _add_word_document_nodes(
    edges: dict[tuple[int, int], float],
    edge_types: dict[tuple[int, int], int],
    nodes: list[str],
    node_types: list[str],
    document_unit_ids: Sequence[Sequence[int]],
    token_to_node: dict[int, int],
    *,
    relation_id: int,
    relation_weight: float,
    link_tokens: bool = True,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int], list[int]]:
    document_nodes: list[int] = []
    for doc_idx, ids in enumerate(document_unit_ids):
        if not ids:
            continue
        document_node = len(nodes)
        document_nodes.append(document_node)
        nodes.append(f"document:{doc_idx}")
        node_types.append("document")
        if not link_tokens:
            continue
        counts = Counter(ids)
        max_count = max(counts.values(), default=1)
        for token_id, count in counts.items():
            if token_id not in token_to_node:
                continue
            token_node = token_to_node[token_id]
            weight = relation_weight * (count / max_count)
            edges[(document_node, token_node)] = edges.get((document_node, token_node), 0.0) + weight
            edges[(token_node, document_node)] = edges.get((token_node, document_node), 0.0) + weight
            edge_types[(document_node, token_node)] = relation_id
            edge_types[(token_node, document_node)] = relation_id
    return edges, edge_types, document_nodes


def _add_topic_document_nodes(
    edges: dict[tuple[int, int], float],
    edge_types: dict[tuple[int, int], int],
    nodes: list[str],
    node_types: list[str],
    document_nodes: Sequence[int],
    document_unit_ids: Sequence[Sequence[int]],
    token_to_node: dict[int, int],
    tokenizer: PersianTokenizer,
    *,
    relation_id: int,
    relation_weight: float,
    max_topics: int = 8,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    global_counts: Counter[int] = Counter()
    for ids in document_unit_ids:
        global_counts.update(ids)
    topic_token_ids = [
        token_id
        for token_id, _ in global_counts.most_common(max_topics)
        if token_id in token_to_node
    ]
    for token_id in topic_token_ids:
        topic_node = len(nodes)
        nodes.append(f"topic:{tokenizer.id_to_token[token_id]}")
        node_types.append("topic")
        token_node = token_to_node[token_id]
        edges[(topic_node, token_node)] = edges.get((topic_node, token_node), 0.0) + relation_weight
        edges[(token_node, topic_node)] = edges.get((token_node, topic_node), 0.0) + relation_weight
        edge_types[(topic_node, token_node)] = relation_id
        edge_types[(token_node, topic_node)] = relation_id
        for doc_node, ids in zip(document_nodes, document_unit_ids):
            if token_id not in ids:
                continue
            edges[(topic_node, doc_node)] = edges.get((topic_node, doc_node), 0.0) + relation_weight
            edges[(doc_node, topic_node)] = edges.get((doc_node, topic_node), 0.0) + relation_weight
            edge_types[(topic_node, doc_node)] = relation_id
            edge_types[(doc_node, topic_node)] = relation_id
    return edges, edge_types


def _edges_to_arrays(
    edges: dict[tuple[int, int], float],
    edge_types: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not edges:
        empty_index = np.empty((2, 0), dtype=np.int64)
        empty_weight = np.empty((0,), dtype=np.float32)
        empty_type = np.empty((0,), dtype=np.int64) if edge_types is not None else None
        return empty_index, empty_weight, empty_type
    ordered = sorted(edges)
    edge_index = np.array(ordered, dtype=np.int64).T
    edge_weight = np.array([edges[edge] for edge in ordered], dtype=np.float32)
    edge_type = (
        np.array([edge_types.get(edge, 0) for edge in ordered], dtype=np.int64)
        if edge_types is not None
        else None
    )
    return edge_index, edge_weight, edge_type


def build_graph_lm_graph(
    texts: Sequence[str],
    tokenizer: PersianTokenizer,
    *,
    window_size: int = 4,
    min_count: int = 1,
    weighting: str = "distance",
    min_edge_weight: float = 0.0,
    top_k: int | None = None,
    directed: bool = False,
    graph_scope: str = "document",
    context_node_type: str = "none",
    graph_relations: Sequence[str] | str | None = None,
    relation_weights: dict[str, float] | None = None,
    semantic_similarity_threshold: float = 0.6,
    semantic_top_k: int | None = 4,
    topic_top_k: int = 8,
) -> GraphLMGraph:
    """Build a weighted token graph for LM fusion.

    The default relation set is the Phase 3 multi-relation preset. Use
    ``graph_relations=["cooccurrence"]`` to reproduce the v1 simple graph
    baseline.
    """

    if window_size < 1:
        raise ValueError("window_size must be positive")

    special_ids = {
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
    }
    enabled_relations = _normalise_relations(graph_relations)
    relation_weight_map = _normalise_relation_weights(relation_weights, enabled_relations)
    relation_to_id = {name: idx for idx, name in enumerate(enabled_relations)}
    units = _split_units(texts, graph_scope)
    document_units = _split_units(texts, "document")
    tokenised = [tokenizer.tokenize(text) for text in units]
    counts: Counter[int] = Counter()
    for tokens in tokenised:
        counts.update(
            tokenizer.token_to_id[token]
            for token in tokens
            if token in tokenizer.token_to_id
        )

    kept_token_ids = [
        idx
        for token, idx in tokenizer.token_to_id.items()
        if idx not in special_ids and counts[idx] >= min_count
    ]
    if not kept_token_ids:
        kept_token_ids = [idx for idx in range(tokenizer.vocab_size) if idx not in special_ids]
    if not kept_token_ids:
        raise ValueError("graph vocabulary is empty")

    token_to_node = {token_id: node_id for node_id, token_id in enumerate(kept_token_ids)}
    nodes = [tokenizer.id_to_token[token_id] for token_id in kept_token_ids]
    unit_ids = _token_ids_for_units(units, tokenizer, token_to_node)
    document_unit_ids = _token_ids_for_units(document_units, tokenizer, token_to_node)
    base_edges, counts, total_tokens = _cooccurrence_edges(
        unit_ids,
        token_to_node,
        window_size=window_size,
        directed=directed,
    )
    edges: dict[tuple[int, int], float] = {}
    edge_types: dict[tuple[int, int], int] = {}
    relation_edge_counts: dict[str, int] = {}

    if "cooccurrence" in enabled_relations:
        weighted_edges = _apply_association_weighting(
            dict(base_edges),
            counts,
            token_to_node,
            total_tokens,
            weighting=weighting,
        )
        weighted_edges = _prune_edges(
            weighted_edges,
            min_edge_weight,
            top_k,
            directed=directed,
        )
        _merge_relation_edges(
            edges,
            edge_types,
            relation_edge_counts,
            weighted_edges,
            relation="cooccurrence",
            relation_id=relation_to_id["cooccurrence"],
            relation_weight=relation_weight_map.get("cooccurrence", 1.0),
        )

    for association_relation in ("pmi", "ppmi"):
        if association_relation not in enabled_relations:
            continue
        weighted_edges = _apply_association_weighting(
            dict(base_edges),
            counts,
            token_to_node,
            total_tokens,
            weighting=association_relation,
        )
        weighted_edges = _prune_edges(
            weighted_edges,
            min_edge_weight,
            top_k,
            directed=directed,
        )
        _merge_relation_edges(
            edges,
            edge_types,
            relation_edge_counts,
            weighted_edges,
            relation=association_relation,
            relation_id=relation_to_id[association_relation],
            relation_weight=relation_weight_map.get(association_relation, 1.0),
        )

    if "dependency" in enabled_relations:
        dependency_edges = _dependency_edges(
            unit_ids,
            token_to_node,
            tokenizer,
            directed=directed,
        )
        dependency_edges = _prune_edges(dependency_edges, min_edge_weight, top_k, directed=directed)
        _merge_relation_edges(
            edges,
            edge_types,
            relation_edge_counts,
            dependency_edges,
            relation="dependency",
            relation_id=relation_to_id["dependency"],
            relation_weight=relation_weight_map.get("dependency", 1.0),
        )

    if "stem" in enabled_relations:
        stem_edges = _stem_edges(token_to_node, tokenizer, directed=directed)
        _merge_relation_edges(
            edges,
            edge_types,
            relation_edge_counts,
            stem_edges,
            relation="stem",
            relation_id=relation_to_id["stem"],
            relation_weight=relation_weight_map.get("stem", 1.0),
        )

    if "subword" in enabled_relations:
        subword_edges = _subword_edges(token_to_node, tokenizer, directed=directed)
        _merge_relation_edges(
            edges,
            edge_types,
            relation_edge_counts,
            subword_edges,
            relation="subword",
            relation_id=relation_to_id["subword"],
            relation_weight=relation_weight_map.get("subword", 1.0),
        )

    if "semantic_similarity" in enabled_relations:
        semantic_edges = _semantic_similarity_edges(
            token_to_node,
            tokenizer,
            directed=directed,
            threshold=semantic_similarity_threshold,
            top_k=semantic_top_k,
        )
        _merge_relation_edges(
            edges,
            edge_types,
            relation_edge_counts,
            semantic_edges,
            relation="semantic_similarity",
            relation_id=relation_to_id["semantic_similarity"],
            relation_weight=relation_weight_map.get("semantic_similarity", 1.0),
        )

    node_types = ["token"] * len(nodes)
    document_nodes: list[int] = []
    if "word_document" in enabled_relations:
        before = len(edges)
        edges, edge_types, document_nodes = _add_word_document_nodes(
            edges,
            edge_types,
            nodes,
            node_types,
            document_unit_ids,
            token_to_node,
            relation_id=relation_to_id["word_document"],
            relation_weight=relation_weight_map.get("word_document", 1.0),
        )
        relation_edge_counts["word_document"] = len(edges) - before

    if "topic_document" in enabled_relations:
        if not document_nodes:
            before_docs = len(edges)
            edges, edge_types, document_nodes = _add_word_document_nodes(
                edges,
                edge_types,
                nodes,
                node_types,
                document_unit_ids,
                token_to_node,
                relation_id=relation_to_id["topic_document"],
                relation_weight=0.0,
                link_tokens=False,
            )
            relation_edge_counts["word_document"] = relation_edge_counts.get(
                "word_document",
                0,
            ) + len(edges) - before_docs
        before = len(edges)
        edges, edge_types = _add_topic_document_nodes(
            edges,
            edge_types,
            nodes,
            node_types,
            document_nodes,
            document_unit_ids,
            token_to_node,
            tokenizer,
            relation_id=relation_to_id["topic_document"],
            relation_weight=relation_weight_map.get("topic_document", 1.0),
            max_topics=topic_top_k,
        )
        relation_edge_counts["topic_document"] = len(edges) - before

    if not edges:
        fallback_edges = _apply_association_weighting(
            dict(base_edges),
        counts,
        token_to_node,
        total_tokens,
            weighting=weighting,
        )
        fallback_edges = _prune_edges(
            fallback_edges,
            min_edge_weight,
            top_k,
            directed=directed,
        )
        fallback_relation = enabled_relations[0]
        _merge_relation_edges(
            edges,
            edge_types,
            relation_edge_counts,
            fallback_edges,
            relation=fallback_relation,
            relation_id=relation_to_id[fallback_relation],
            relation_weight=relation_weight_map.get(fallback_relation, 1.0),
        )

    if context_node_type != "none":
        if context_node_type not in relation_to_id:
            relation_to_id[context_node_type] = len(relation_to_id)
        before = len(edges)
        context_relation_id = relation_to_id[context_node_type]
        edges, edge_types, nodes, node_types = _add_context_nodes(
            edges,
            edge_types,
            nodes,
            unit_ids,
            token_to_node,
            context_node_type=context_node_type,
            edge_weight=1.0,
            relation_id=context_relation_id,
        )
        relation_edge_counts[context_node_type] = len(edges) - before

    edge_index, edge_weight, edge_type = _edges_to_arrays(edges, edge_types)
    relation_weights_payload = {
        relation: relation_weight_map.get(relation, 1.0)
        for relation in enabled_relations
    }

    return GraphLMGraph(
        nodes=nodes,
        token_to_node=token_to_node,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        node_types=node_types,
        directed=directed,
        graph_config={
            "window_size": window_size,
            "min_count": min_count,
            "weighting": weighting,
            "min_edge_weight": min_edge_weight,
            "top_k": top_k,
            "directed": directed,
            "graph_scope": graph_scope,
            "context_node_type": context_node_type,
            "num_nodes": len(nodes),
            "num_edges": int(len(edges) if directed else len(edges) // 2),
            "enabled_relations": enabled_relations,
            "relation_weights": relation_weights_payload,
            "edge_types": relation_to_id,
            "relation_edge_counts": relation_edge_counts,
            "node_type_counts": dict(Counter(node_types)),
            "semantic_similarity_threshold": semantic_similarity_threshold,
            "semantic_top_k": semantic_top_k,
            "topic_top_k": topic_top_k,
        },
    )


def build_graph_lm_graph_from_token_ids(
    token_id_sequences: Sequence[Sequence[int]],
    tokenizer: PersianTokenizer,
    *,
    window_size: int = 4,
    weighting: str = "distance",
    min_edge_weight: float = 0.0,
    top_k: int | None = None,
    directed: bool = True,
) -> GraphLMGraph:
    """Build a small local graph from the token ids in a batch or generation context."""

    special_ids = {tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id}
    kept_token_ids = sorted(
        {
            int(token_id)
            for ids in token_id_sequences
            for token_id in ids
            if int(token_id) not in special_ids and int(token_id) in tokenizer.id_to_token
        }
    )
    if not kept_token_ids:
        kept_token_ids = [idx for idx in range(tokenizer.vocab_size) if idx not in special_ids][:1]
    token_to_node = {token_id: node_id for node_id, token_id in enumerate(kept_token_ids)}
    unit_ids = [
        [int(token_id) for token_id in ids if int(token_id) in token_to_node]
        for ids in token_id_sequences
    ]
    edges, counts, total_tokens = _cooccurrence_edges(
        unit_ids,
        token_to_node,
        window_size=window_size,
        directed=directed,
    )
    edges = _apply_association_weighting(
        edges,
        counts,
        token_to_node,
        total_tokens,
        weighting=weighting,
    )
    edges = _prune_edges(edges, min_edge_weight, top_k, directed=directed)
    nodes = [tokenizer.id_to_token[token_id] for token_id in kept_token_ids]
    edge_types: dict[tuple[int, int], int] = {}
    edge_index, edge_weight, edge_type = _edges_to_arrays(edges, edge_types)
    return GraphLMGraph(
        nodes=nodes,
        token_to_node=token_to_node,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        node_types=["token"] * len(nodes),
        directed=directed,
        graph_config={
            "mode": "dynamic",
            "window_size": window_size,
            "weighting": weighting,
            "min_edge_weight": min_edge_weight,
            "top_k": top_k,
            "directed": directed,
            "num_nodes": len(nodes),
            "num_edges": int(len(edges) if directed else len(edges) // 2),
        },
    )

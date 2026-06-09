"""Vocabulary co-occurrence graph builders for Graph-LM."""

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
            edge_types[(context_node, token_node)] = 1
            edge_types[(token_node, context_node)] = 1
    return edges, edge_types, nodes, node_types


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
) -> GraphLMGraph:
    """Build a weighted token graph for LM fusion."""

    if window_size < 1:
        raise ValueError("window_size must be positive")

    special_ids = {
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
    }
    units = _split_units(texts, graph_scope)
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
    edge_types: dict[tuple[int, int], int] = {}
    edges, edge_types, nodes, node_types = _add_context_nodes(
        edges,
        edge_types,
        nodes,
        unit_ids,
        token_to_node,
        context_node_type=context_node_type,
        edge_weight=1.0,
    )
    edge_index, edge_weight, edge_type = _edges_to_arrays(edges, edge_types)

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
            "edge_types": {"cooccurrence": 0, context_node_type: 1}
            if context_node_type != "none"
            else {"cooccurrence": 0},
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

"""Vocabulary co-occurrence graph builders for Graph-LM."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..features.pyg_data import graph_to_data
from ..graphs.graph import Graph
from .tokenizer import PersianTokenizer

SENTENCE_SPLIT_PATTERN = re.compile(r"[.!؟?؛\n]+")


@dataclass
class GraphLMGraph:
    graph: Graph
    token_to_node: dict[int, int]
    graph_config: dict[str, object]
    edge_type_adjacency: np.ndarray | None = None

    def to_pyg_data(self):
        features = np.eye(len(self.graph.nodes), dtype=np.float32)
        data = graph_to_data(self.graph, features=features)
        if self.edge_type_adjacency is not None and data.edge_index.numel() > 0:
            src = data.edge_index[0].cpu().numpy()
            dst = data.edge_index[1].cpu().numpy()
            data.edge_type = torch.tensor(
                self.edge_type_adjacency[src, dst], dtype=torch.long
            )
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
            units.extend(part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip())
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


def _cooccurrence_adjacency(
    unit_ids: Sequence[Sequence[int]],
    token_to_node: dict[int, int],
    *,
    window_size: int,
    directed: bool,
) -> tuple[np.ndarray, Counter[int], float]:
    n = len(token_to_node)
    adjacency = np.zeros((n, n), dtype=np.float32)
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
                adjacency[src, dst] += weight
                if not directed:
                    adjacency[dst, src] += weight
    return adjacency, counts, float(max(1, total_tokens))


def _apply_association_weighting(
    adjacency: np.ndarray,
    counts: Counter[int],
    token_to_node: dict[int, int],
    total_tokens: float,
    *,
    weighting: str,
) -> np.ndarray:
    weighting = weighting.lower()
    if weighting in {"distance", "count", "raw"}:
        return adjacency
    if weighting not in {"pmi", "ppmi"}:
        raise ValueError("graph_weighting must be one of: distance, count, raw, pmi, ppmi")

    weighted = np.zeros_like(adjacency)
    total_edges = float(max(adjacency.sum(), 1.0))
    node_counts = np.zeros((len(token_to_node),), dtype=np.float32)
    for token_id, node_id in token_to_node.items():
        node_counts[node_id] = float(counts[token_id])

    src, dst = np.nonzero(adjacency)
    for i, j in zip(src, dst, strict=False):
        p_ij = float(adjacency[i, j]) / total_edges
        p_i = max(float(node_counts[i]) / total_tokens, 1e-12)
        p_j = max(float(node_counts[j]) / total_tokens, 1e-12)
        score = math.log(max(p_ij, 1e-12) / (p_i * p_j))
        weighted[i, j] = max(0.0, score) if weighting == "ppmi" else score
    weighted[weighted < 0] = 0.0 if weighting == "ppmi" else weighted[weighted < 0]
    return weighted


def _prune_edges(adjacency: np.ndarray, min_edge_weight: float, top_k: int | None) -> np.ndarray:
    pruned = adjacency.copy()
    if min_edge_weight > 0:
        pruned[pruned < min_edge_weight] = 0.0
    if top_k is not None and top_k > 0:
        for row in range(pruned.shape[0]):
            nonzero = np.flatnonzero(pruned[row])
            if len(nonzero) > top_k:
                keep = nonzero[np.argsort(pruned[row, nonzero])[-top_k:]]
                drop = np.setdiff1d(nonzero, keep, assume_unique=False)
                pruned[row, drop] = 0.0
    return pruned


def _add_context_nodes(
    adjacency: np.ndarray,
    edge_types: np.ndarray,
    nodes: list[str],
    unit_ids: Sequence[Sequence[int]],
    token_to_node: dict[int, int],
    *,
    context_node_type: str,
    edge_weight: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    node_types = ["token"] * len(nodes)
    if context_node_type == "none":
        return adjacency, edge_types, nodes, node_types
    if context_node_type not in {"document", "sentence"}:
        raise ValueError("context_node_type must be one of: none, document, sentence")

    context_units = [ids for ids in unit_ids if ids]
    old_n = adjacency.shape[0]
    new_n = old_n + len(context_units)
    expanded = np.zeros((new_n, new_n), dtype=np.float32)
    expanded[:old_n, :old_n] = adjacency
    expanded_types = np.zeros((new_n, new_n), dtype=np.int64)
    expanded_types[:old_n, :old_n] = edge_types
    for idx, ids in enumerate(context_units):
        context_node = old_n + idx
        nodes.append(f"{context_node_type}:{idx}")
        node_types.append(context_node_type)
        for token_id in set(ids):
            token_node = token_to_node[token_id]
            expanded[context_node, token_node] = edge_weight
            expanded[token_node, context_node] = edge_weight
            expanded_types[context_node, token_node] = 1
            expanded_types[token_node, context_node] = 1
    return expanded, expanded_types, nodes, node_types


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
        counts.update(tokenizer.token_to_id[token] for token in tokens if token in tokenizer.token_to_id)

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
    adjacency, counts, total_tokens = _cooccurrence_adjacency(
        unit_ids,
        token_to_node,
        window_size=window_size,
        directed=directed,
    )
    adjacency = _apply_association_weighting(
        adjacency,
        counts,
        token_to_node,
        total_tokens,
        weighting=weighting,
    )
    adjacency = _prune_edges(adjacency, min_edge_weight, top_k)
    edge_types = np.zeros_like(adjacency, dtype=np.int64)
    adjacency, edge_types, nodes, node_types = _add_context_nodes(
        adjacency,
        edge_types,
        nodes,
        unit_ids,
        token_to_node,
        context_node_type=context_node_type,
        edge_weight=1.0,
    )

    graph = Graph(
        nodes=nodes,
        adjacency=adjacency,
        node_types=node_types,
        directed=directed,
    )
    return GraphLMGraph(
        graph=graph,
        token_to_node=token_to_node,
        edge_type_adjacency=edge_types,
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
            "num_edges": int((adjacency > 0).sum() if directed else (adjacency > 0).sum() // 2),
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
    adjacency, counts, total_tokens = _cooccurrence_adjacency(
        unit_ids,
        token_to_node,
        window_size=window_size,
        directed=directed,
    )
    adjacency = _apply_association_weighting(
        adjacency,
        counts,
        token_to_node,
        total_tokens,
        weighting=weighting,
    )
    adjacency = _prune_edges(adjacency, min_edge_weight, top_k)
    nodes = [tokenizer.id_to_token[token_id] for token_id in kept_token_ids]
    graph = Graph(
        nodes=nodes,
        adjacency=adjacency,
        node_types=["token"] * len(nodes),
        directed=directed,
    )
    return GraphLMGraph(
        graph=graph,
        token_to_node=token_to_node,
        edge_type_adjacency=np.zeros_like(adjacency, dtype=np.int64),
        graph_config={
            "mode": "dynamic",
            "window_size": window_size,
            "weighting": weighting,
            "min_edge_weight": min_edge_weight,
            "top_k": top_k,
            "directed": directed,
            "num_nodes": len(nodes),
            "num_edges": int((adjacency > 0).sum() if directed else (adjacency > 0).sum() // 2),
        },
    )

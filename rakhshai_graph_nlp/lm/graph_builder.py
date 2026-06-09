"""Vocabulary co-occurrence graph builder for Graph-LM."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..features.pyg_data import graph_to_data
from ..graphs.graph import Graph
from .tokenizer import PersianTokenizer


@dataclass
class GraphLMGraph:
    graph: Graph
    token_to_node: dict[int, int]
    graph_config: dict[str, object]

    def to_pyg_data(self):
        features = np.eye(len(self.graph.nodes), dtype=np.float32)
        return graph_to_data(self.graph, features=features)

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


def build_graph_lm_graph(
    texts: Sequence[str],
    tokenizer: PersianTokenizer,
    *,
    window_size: int = 4,
    min_count: int = 1,
) -> GraphLMGraph:
    """Build a weighted token co-occurrence graph for LM fusion."""

    if window_size < 1:
        raise ValueError("window_size must be positive")

    special_ids = {
        tokenizer.pad_id,
        tokenizer.unk_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
    }
    tokenised = [tokenizer.tokenize(text) for text in texts]
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
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float32)

    for tokens in tokenised:
        ids = [
            tokenizer.token_to_id[token]
            for token in tokens
            if tokenizer.token_to_id.get(token) in token_to_node
        ]
        for i, src_id in enumerate(ids):
            src = token_to_node[src_id]
            right = min(len(ids), i + window_size + 1)
            for j in range(i + 1, right):
                dst_id = ids[j]
                if src_id == dst_id:
                    continue
                dst = token_to_node[dst_id]
                distance = j - i
                weight = 1.0 / distance
                adjacency[src, dst] += weight
                adjacency[dst, src] += weight

    graph = Graph(
        nodes=nodes,
        adjacency=adjacency,
        node_types=["token"] * len(nodes),
        directed=False,
    )
    return GraphLMGraph(
        graph=graph,
        token_to_node=token_to_node,
        graph_config={
            "window_size": window_size,
            "min_count": min_count,
            "num_nodes": len(nodes),
            "num_edges": int((adjacency > 0).sum() // 2),
        },
    )

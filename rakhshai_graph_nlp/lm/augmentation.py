"""Low-data augmentation and graph regularisation utilities for Graph-LM."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Sequence

import torch
from torch_geometric.data import Data

PUNCTUATION_PATTERN = re.compile(r"[،؛:,.!?؟]+")


@dataclass
class TextAugmentationConfig:
    enabled: bool = True
    ratio: float = 0.5
    token_dropout: float = 0.05
    punctuation_dropout: float = 0.5
    min_tokens: int = 3


def _drop_tokens(tokens: list[str], probability: float, min_tokens: int) -> list[str]:
    if probability <= 0 or len(tokens) <= min_tokens:
        return tokens
    kept = [token for token in tokens if random.random() >= probability]
    if len(kept) < min_tokens:
        return tokens
    return kept


def augment_text(text: str, config: TextAugmentationConfig) -> str:
    """Create a conservative Persian text augmentation without adding new tokens."""

    augmented = text
    if config.punctuation_dropout > 0 and random.random() < config.punctuation_dropout:
        augmented = PUNCTUATION_PATTERN.sub(" ", augmented)
    tokens = augmented.split()
    tokens = _drop_tokens(tokens, config.token_dropout, config.min_tokens)
    return " ".join(tokens).strip()


def augment_corpus(
    corpus: Sequence[str],
    config: TextAugmentationConfig,
    *,
    seed: int,
) -> list[str]:
    """Return original corpus plus augmented training-only examples."""

    base = [text for text in corpus if text.strip()]
    if not config.enabled or config.ratio <= 0 or not base:
        return list(base)
    state = random.getstate()
    random.seed(seed)
    try:
        sample_count = max(1, int(round(len(base) * config.ratio)))
        augmented: list[str] = []
        for index in range(sample_count):
            text = base[index % len(base)]
            candidate = augment_text(text, config)
            if candidate and candidate != text:
                augmented.append(candidate)
        return [*base, *augmented]
    finally:
        random.setstate(state)


def augment_graph_data(
    graph_data: Data | None,
    *,
    edge_dropout: float,
    node_dropout: float,
    subgraph_ratio: float,
    training: bool,
) -> Data | None:
    """Apply edge/node dropout and subgraph edge sampling to a PyG graph view."""

    if graph_data is None or not training:
        return graph_data
    if graph_data.edge_index.numel() == 0:
        return graph_data

    edge_count = graph_data.edge_index.size(1)
    keep = torch.ones(edge_count, dtype=torch.bool, device=graph_data.edge_index.device)

    if edge_dropout > 0:
        keep &= torch.rand(edge_count, device=keep.device) >= edge_dropout

    if 0 < subgraph_ratio < 1:
        keep &= torch.rand(edge_count, device=keep.device) < subgraph_ratio

    if node_dropout > 0 and int(graph_data.num_nodes) > 1:
        keep_nodes = torch.rand(
            int(graph_data.num_nodes),
            device=graph_data.edge_index.device,
        ) >= node_dropout
        edge_nodes = graph_data.edge_index
        keep &= keep_nodes[edge_nodes[0]] & keep_nodes[edge_nodes[1]]

    if not keep.any():
        # Keep at least one edge so GNN layers and graph losses retain a valid graph.
        keep[torch.randint(0, edge_count, (1,), device=keep.device)] = True

    data = Data(
        edge_index=graph_data.edge_index[:, keep],
        num_nodes=int(graph_data.num_nodes),
    )
    edge_weight = getattr(graph_data, "edge_weight", None)
    if edge_weight is not None:
        data.edge_weight = edge_weight[keep]
    edge_type = getattr(graph_data, "edge_type", None)
    if edge_type is not None:
        data.edge_type = edge_type[keep]
    node_type_id = getattr(graph_data, "node_type_id", None)
    if node_type_id is not None:
        data.node_type_id = node_type_id
    return data


def mean_pool_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    mask = ~input_ids.eq(pad_token_id)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom

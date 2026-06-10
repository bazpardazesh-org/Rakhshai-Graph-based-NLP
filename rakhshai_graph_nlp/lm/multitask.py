"""Multi-task losses for Graph-LM training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .model import GraphCausalLM


DEFAULT_TASK_LOSSES = (
    "next_token",
    "masked_token",
    "edge",
    "neighbor",
    "node_relation",
    "graph_text",
    "sentence_graph",
)


@dataclass
class MultiTaskLossConfig:
    task_losses: str = ",".join(DEFAULT_TASK_LOSSES)
    next_token_weight: float = 1.0
    masked_token_weight: float = 0.25
    edge_prediction_weight: float = 0.1
    neighbor_prediction_weight: float = 0.1
    node_relation_weight: float = 0.1
    graph_text_alignment_weight: float = 0.1
    sentence_graph_alignment_weight: float = 0.1
    mask_probability: float = 0.15
    negative_samples: int = 1


def parse_task_losses(task_losses: str | None) -> set[str]:
    if not task_losses:
        return set()
    aliases = {
        "all": ",".join(DEFAULT_TASK_LOSSES),
        "edge_prediction": "edge",
        "neighbor_prediction": "neighbor",
        "relation": "node_relation",
        "relation_prediction": "node_relation",
        "graph_text_alignment": "graph_text",
        "sentence_graph_alignment": "sentence_graph",
        "mlm": "masked_token",
        "causal": "next_token",
    }
    parsed: set[str] = set()
    for raw in task_losses.replace("+", ",").split(","):
        name = raw.strip().lower().replace("-", "_")
        if not name:
            continue
        expanded = aliases.get(name, name)
        for part in expanded.split(","):
            if part:
                parsed.add(part)
    invalid = parsed - set(DEFAULT_TASK_LOSSES)
    if invalid:
        raise ValueError(
            "task_losses must contain only: " + ", ".join(DEFAULT_TASK_LOSSES)
        )
    return parsed


def _task_weight(config: MultiTaskLossConfig, task: str) -> float:
    return {
        "next_token": config.next_token_weight,
        "masked_token": config.masked_token_weight,
        "edge": config.edge_prediction_weight,
        "neighbor": config.neighbor_prediction_weight,
        "node_relation": config.node_relation_weight,
        "graph_text": config.graph_text_alignment_weight,
        "sentence_graph": config.sentence_graph_alignment_weight,
    }[task]


def _mean_pool(hidden: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    mask = ~input_ids.eq(pad_token_id)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
    return (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    edges = edge_index.detach().cpu().t().tolist()
    return {(int(src), int(dst)) for src, dst in edges}


def _negative_edges(
    positive_edges: torch.Tensor,
    *,
    num_nodes: int,
    multiplier: int,
) -> torch.Tensor:
    count = positive_edges.size(1) * max(1, multiplier)
    if count == 0 or num_nodes < 2:
        return positive_edges.new_empty((2, 0))
    existing = _edge_set(positive_edges)
    negatives: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(count * 20, 50)
    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        src = int(torch.randint(0, num_nodes, (1,), device=positive_edges.device).item())
        dst = int(torch.randint(0, num_nodes, (1,), device=positive_edges.device).item())
        if src == dst or (src, dst) in existing:
            continue
        negatives.append((src, dst))
    if not negatives:
        return positive_edges.new_empty((2, 0))
    return torch.tensor(negatives, dtype=torch.long, device=positive_edges.device).t()


def _binary_edge_loss(
    node_embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    negative_samples: int,
) -> torch.Tensor | None:
    if edge_index.numel() == 0 or node_embeddings.size(0) < 2:
        return None
    negative_index = _negative_edges(
        edge_index,
        num_nodes=node_embeddings.size(0),
        multiplier=negative_samples,
    )
    if negative_index.numel() == 0:
        return None
    pos_score = (
        node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
    ).sum(dim=-1)
    neg_score = (
        node_embeddings[negative_index[0]] * node_embeddings[negative_index[1]]
    ).sum(dim=-1)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    return F.binary_cross_entropy_with_logits(scores, labels)


def _node_relation_loss(
    model: GraphCausalLM,
    node_embeddings: torch.Tensor,
    graph_data: Data,
    *,
    num_relations: int,
) -> torch.Tensor | None:
    edge_type = getattr(graph_data, "edge_type", None)
    if edge_type is None or graph_data.edge_index.numel() == 0 or num_relations <= 1:
        return None
    edge_index = graph_data.edge_index.to(node_embeddings.device)
    edge_type = edge_type.to(node_embeddings.device).long().clamp(0, num_relations - 1)
    pair_embeddings = node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]
    logits = model.node_relation_head(pair_embeddings)
    return F.cross_entropy(logits, edge_type)


def _masked_token_loss(
    model: GraphCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    graph_data: Data | None,
    token_node_ids: torch.Tensor | None,
    graph_embeddings: torch.Tensor | None,
    mask_probability: float,
) -> torch.Tensor | None:
    valid = labels.ne(-100) & ~input_ids.eq(model.config.pad_token_id)
    if not valid.any() or mask_probability <= 0:
        return None
    selected = torch.rand(input_ids.shape, device=input_ids.device) < mask_probability
    selected &= valid
    if not selected.any():
        first = valid.nonzero(as_tuple=False)[0]
        selected[first[0], first[1]] = True
    masked_input = input_ids.clone()
    masked_input[selected] = 1
    masked_labels = torch.full_like(labels, -100)
    masked_labels[selected] = input_ids[selected]
    masked_graph_embeddings = None
    if graph_embeddings is not None:
        masked_graph_embeddings = graph_embeddings.detach()
    output = model(
        masked_input,
        graph_data=graph_data,
        token_node_ids=token_node_ids,
        graph_embeddings=masked_graph_embeddings,
        labels=masked_labels,
    )
    return output.get("loss")


def compute_multitask_losses(
    model: GraphCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    output: dict[str, torch.Tensor],
    *,
    graph_data: Data | None,
    token_node_ids: torch.Tensor | None,
    config: MultiTaskLossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, str]]:
    tasks = parse_task_losses(config.task_losses)
    device = input_ids.device
    base_loss = output.get("loss")
    total = input_ids.new_tensor(0.0, dtype=torch.float32)
    losses: dict[str, torch.Tensor] = {}
    status: dict[str, str] = {}
    node_embeddings_cache: torch.Tensor | None = None

    for task in DEFAULT_TASK_LOSSES:
        weight = _task_weight(config, task)
        if task not in tasks or weight <= 0:
            status[task] = "disabled"
            continue
        loss: torch.Tensor | None = None
        if task == "next_token":
            loss = base_loss
        elif task == "masked_token":
            loss = _masked_token_loss(
                model,
                input_ids,
                labels,
                graph_data=graph_data,
                token_node_ids=token_node_ids,
                graph_embeddings=output.get("graph_embeddings"),
                mask_probability=config.mask_probability,
            )
        else:
            hidden = output.get("hidden")
            graph_embeddings = output.get("graph_embeddings")
            graph_ready = (
                hidden is not None
                and graph_data is not None
                and token_node_ids is not None
                and model.graph_encoder is not None
            )
            if not graph_ready:
                status[task] = "skipped"
                continue
            if task in {"edge", "neighbor", "node_relation", "graph_text"}:
                if node_embeddings_cache is None:
                    node_embeddings_cache = model._encode_graph_nodes(
                        graph_data,
                        token_node_ids,
                    )
                node_embeddings = node_embeddings_cache
            else:
                node_embeddings = None
            if task == "edge" and node_embeddings is not None:
                loss = _binary_edge_loss(
                    node_embeddings,
                    graph_data.edge_index.to(device),
                    negative_samples=config.negative_samples,
                )
            elif task == "neighbor" and node_embeddings is not None:
                loss = _binary_edge_loss(
                    node_embeddings,
                    graph_data.edge_index.flip(0).to(device),
                    negative_samples=config.negative_samples,
                )
            elif task == "node_relation" and node_embeddings is not None:
                loss = _node_relation_loss(
                    model,
                    node_embeddings,
                    graph_data,
                    num_relations=model.config.graph_edge_types,
                )
            elif task == "graph_text" and hidden is not None and node_embeddings is not None:
                text_repr = _mean_pool(hidden, input_ids, model.config.pad_token_id)
                graph_repr = node_embeddings.mean(dim=0, keepdim=True).expand_as(text_repr)
                loss = 1.0 - F.cosine_similarity(text_repr, graph_repr, dim=-1).mean()
            elif (
                task == "sentence_graph"
                and hidden is not None
                and graph_embeddings is not None
            ):
                text_repr = _mean_pool(hidden, input_ids, model.config.pad_token_id)
                graph_repr = _mean_pool(
                    graph_embeddings,
                    input_ids,
                    model.config.pad_token_id,
                )
                if graph_repr.abs().sum() > 0:
                    loss = 1.0 - F.cosine_similarity(text_repr, graph_repr, dim=-1).mean()
        if loss is None:
            status[task] = "skipped"
            continue
        weighted = loss * float(weight)
        losses[task] = loss
        total = total + weighted
        status[task] = "active"
    if not losses and base_loss is not None:
        losses["next_token"] = base_loss
        total = base_loss
        status["next_token"] = "active"
    return total, losses, status

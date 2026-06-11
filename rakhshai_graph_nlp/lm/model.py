"""Graph-fused Transformer causal language model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, MessagePassing, RGCNConv

from .graph_memory import GraphMemoryArtifact, GraphMemoryConfig
from .tokenizer import PersianTokenizer


@dataclass
class GraphLMConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    graph_encoder: str = "gat"
    graph_hidden_dim: int = 128
    graph_heads: int = 4
    graph_relation_mode: str = "bias"
    graph_pooling: str = "none"
    graph_node_importance: bool = False
    fusion: str = "gated"
    fusion_layers: str = "input"
    fusion_levels: str = "token"
    graph_fusion_scale: float = 1.0
    graph_fusion_dropout: float = 0.0
    graph_edge_types: int = 1
    pad_token_id: int = 0


@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    min_new_tokens: int = 0
    temperature: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    eos_token_id: int = 3


class WeightedSAGEConv(MessagePassing):
    """GraphSAGE-style weighted mean aggregation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin_root = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            edge_weight = x.new_ones((edge_index.size(1),))
        edge_weight = edge_weight.to(dtype=x.dtype, device=x.device)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        dst = edge_index[1]
        denom = x.new_zeros((x.size(0),))
        denom.scatter_add_(0, dst, edge_weight.abs())
        out = out / denom.clamp_min(1e-12).unsqueeze(-1)
        return self.lin_root(x) + self.lin_neigh(out)

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return x_j * edge_weight.unsqueeze(-1)


class RelationWeightedSAGEConv(MessagePassing):
    """GraphSAGE aggregation with relation-aware edge attributes."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr="add")
        self.lin_root = nn.Linear(in_channels, out_channels)
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        self.edge_gate = nn.Linear(edge_dim, in_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_weight is None:
            edge_weight = x.new_ones((edge_index.size(1),))
        edge_weight = edge_weight.to(dtype=x.dtype, device=x.device)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
        )
        dst = edge_index[1]
        denom = x.new_zeros((x.size(0),))
        denom.scatter_add_(0, dst, edge_weight.abs())
        out = out / denom.clamp_min(1e-12).unsqueeze(-1)
        return self.lin_root(x) + self.lin_neigh(out)

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor | None,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if edge_attr is not None:
            x_j = x_j * torch.sigmoid(self.edge_gate(edge_attr))
        return x_j * edge_weight.unsqueeze(-1)


class RakhshaiGraphEncoder(nn.Module):
    """GNN encoder that creates graph-aware token embeddings."""

    def __init__(self, config: GraphLMConfig):
        super().__init__()
        encoder = config.graph_encoder.lower()
        relation_mode = config.graph_relation_mode.lower()
        if relation_mode not in {"bias", "embedding", "rgcn"}:
            raise ValueError("graph_relation_mode must be one of: bias, embedding, rgcn")
        if relation_mode == "rgcn":
            encoder = "rgcn"
        if encoder == "rgcn":
            relation_mode = "rgcn"
        if encoder == "gcn":
            self.conv1 = GCNConv(config.d_model, config.graph_hidden_dim)
            self.conv2 = GCNConv(config.graph_hidden_dim, config.d_model)
        elif encoder == "graphsage":
            if relation_mode == "embedding":
                self.conv1 = RelationWeightedSAGEConv(
                    config.d_model,
                    config.graph_hidden_dim,
                    config.d_model,
                )
                self.conv2 = RelationWeightedSAGEConv(
                    config.graph_hidden_dim,
                    config.d_model,
                    config.graph_hidden_dim,
                )
            else:
                self.conv1 = WeightedSAGEConv(config.d_model, config.graph_hidden_dim)
                self.conv2 = WeightedSAGEConv(config.graph_hidden_dim, config.d_model)
        elif encoder == "gat":
            edge_dim = config.d_model if relation_mode == "embedding" else 1
            self.conv1 = GATConv(
                config.d_model,
                config.graph_hidden_dim,
                heads=config.graph_heads,
                dropout=config.dropout,
                edge_dim=edge_dim,
            )
            self.conv2 = GATConv(
                config.graph_hidden_dim * config.graph_heads,
                config.d_model,
                heads=1,
                concat=False,
                dropout=config.dropout,
                edge_dim=config.graph_hidden_dim if relation_mode == "embedding" else 1,
            )
        elif encoder == "rgcn":
            self.conv1 = RGCNConv(
                config.d_model,
                config.graph_hidden_dim,
                num_relations=max(1, config.graph_edge_types),
            )
            self.conv2 = RGCNConv(
                config.graph_hidden_dim,
                config.d_model,
                num_relations=max(1, config.graph_edge_types),
            )
        else:
            raise ValueError("graph_encoder must be one of: gcn, graphsage, gat, rgcn")
        self.encoder = encoder
        self.relation_mode = relation_mode
        self.dropout = nn.Dropout(config.dropout)
        self.edge_type_bias = (
            nn.Embedding(config.graph_edge_types, 1)
            if config.graph_edge_types > 1
            else None
        )
        self.edge_type_embedding = (
            nn.Embedding(config.graph_edge_types, config.d_model)
            if relation_mode == "embedding" and config.graph_edge_types > 1
            else None
        )
        self.edge_type_hidden_embedding = (
            nn.Embedding(config.graph_edge_types, config.graph_hidden_dim)
            if relation_mode == "embedding" and config.graph_edge_types > 1
            else None
        )
        self.node_importance = (
            nn.Linear(config.d_model, 1) if config.graph_node_importance else None
        )
        if config.graph_pooling not in {"none", "mean", "attention"}:
            raise ValueError("graph_pooling must be one of: none, mean, attention")
        self.graph_pooling = config.graph_pooling
        self.pool_score = (
            nn.Linear(config.d_model, 1) if config.graph_pooling == "attention" else None
        )

    def _edge_type(self, graph_data: Data, device: torch.device) -> torch.Tensor:
        edge_type = getattr(graph_data, "edge_type", None)
        if edge_type is None:
            return torch.zeros(
                (graph_data.edge_index.size(1),),
                dtype=torch.long,
                device=device,
            )
        max_type = max(0, getattr(self.conv1, "num_relations", 1) - 1)
        if self.edge_type_bias is not None:
            max_type = self.edge_type_bias.num_embeddings - 1
        elif self.edge_type_embedding is not None:
            max_type = self.edge_type_embedding.num_embeddings - 1
        return edge_type.to(device).long().clamp(0, max_type)

    def _relation_edge_attr(
        self,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor | None,
        *,
        hidden: bool = False,
    ) -> torch.Tensor | None:
        embedding = self.edge_type_hidden_embedding if hidden else self.edge_type_embedding
        if self.relation_mode == "embedding" and embedding is not None:
            edge_attr = embedding(edge_type)
            if edge_weight is not None:
                edge_attr = edge_attr * edge_weight.unsqueeze(-1)
            return edge_attr
        if edge_weight is None:
            return None
        return edge_weight.unsqueeze(-1)

    def forward(self, node_features: torch.Tensor, graph_data: Data) -> torch.Tensor:
        edge_index = graph_data.edge_index.to(node_features.device)
        edge_weight = getattr(graph_data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(node_features.device).to(dtype=node_features.dtype)
        edge_type = self._edge_type(graph_data, node_features.device)
        if (
            self.relation_mode == "bias"
            and self.edge_type_bias is not None
            and edge_weight is not None
        ):
            edge_weight = edge_weight * torch.sigmoid(self.edge_type_bias(edge_type).squeeze(-1))

        if self.encoder == "gcn":
            x = self.conv1(node_features, edge_index, edge_weight)
        elif self.encoder == "gat":
            edge_attr = self._relation_edge_attr(edge_type, edge_weight)
            x = self.conv1(node_features, edge_index, edge_attr=edge_attr)
        elif self.encoder == "rgcn":
            x = self.conv1(node_features, edge_index, edge_type)
            if edge_weight is not None:
                x = x * edge_weight.mean().clamp_min(1e-6)
        elif self.relation_mode == "embedding":
            edge_attr = self._relation_edge_attr(edge_type, edge_weight)
            x = self.conv1(node_features, edge_index, edge_attr, edge_weight)
        else:
            x = self.conv1(node_features, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        if self.encoder == "gcn":
            node_embeddings = self.conv2(x, edge_index, edge_weight)
        elif self.encoder == "gat":
            edge_attr = self._relation_edge_attr(edge_type, edge_weight, hidden=True)
            node_embeddings = self.conv2(x, edge_index, edge_attr=edge_attr)
        elif self.encoder == "rgcn":
            node_embeddings = self.conv2(x, edge_index, edge_type)
            if edge_weight is not None:
                node_embeddings = node_embeddings * edge_weight.mean().clamp_min(1e-6)
        elif self.relation_mode == "embedding":
            edge_attr = self._relation_edge_attr(edge_type, edge_weight, hidden=True)
            node_embeddings = self.conv2(x, edge_index, edge_attr, edge_weight)
        else:
            node_embeddings = self.conv2(x, edge_index, edge_weight)
        if self.graph_pooling != "none":
            pooled = self.pool(node_embeddings, graph_data)
            node_embeddings = node_embeddings + pooled.unsqueeze(0)
        return node_embeddings

    def importance_scores(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if self.node_importance is None:
            return torch.empty((0,), device=node_embeddings.device)
        return torch.softmax(self.node_importance(node_embeddings).squeeze(-1), dim=0)

    def pool(self, node_embeddings: torch.Tensor, graph_data: Data) -> torch.Tensor:
        node_type_id = getattr(graph_data, "node_type_id", None)
        if node_type_id is not None:
            node_type_id = node_type_id.to(node_embeddings.device)
            mask = node_type_id != 0
            if not mask.any():
                mask = torch.ones(
                    node_embeddings.size(0),
                    dtype=torch.bool,
                    device=node_embeddings.device,
                )
        else:
            mask = torch.ones(
                node_embeddings.size(0),
                dtype=torch.bool,
                device=node_embeddings.device,
            )
        selected = node_embeddings[mask]
        if self.graph_pooling == "attention" and self.pool_score is not None:
            weights = torch.softmax(self.pool_score(selected).squeeze(-1), dim=0)
            return (selected * weights.unsqueeze(-1)).sum(dim=0)
        return selected.mean(dim=0)


class GraphTokenFusion(nn.Module):
    """Adaptive multi-level graph-text fusion."""

    def __init__(self, config: GraphLMConfig):
        super().__init__()
        self.fusion = config.fusion.lower()
        self.fusion_levels = self._parse_levels(config.fusion_levels)
        self.graph_fusion_scale = float(config.graph_fusion_scale)
        self.graph_dropout = nn.Dropout(config.graph_fusion_dropout)
        if self.fusion == "gated":
            self.gate = nn.Linear(config.d_model * 2, config.d_model)
        elif self.fusion in {"context_gated", "contextual"}:
            self.gate = nn.Sequential(
                nn.Linear(config.d_model * 3, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
        elif self.fusion in {"add", "sum"}:
            self.gate = None
        else:
            raise ValueError("fusion must be one of: gated, context_gated, add")
        self.sentence_gate = nn.Linear(config.d_model * 2, config.d_model)
        self.subgraph_gate = nn.Linear(config.d_model * 2, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        # Zero-init gating (Flamingo-style): each fusion level scales its graph
        # update by tanh(alpha) with alpha starting at zero, so an untrained
        # model is exactly equivalent to the no-graph baseline and graph
        # information only flows in once training finds it useful.
        self.token_alpha = nn.Parameter(torch.zeros(1))
        self.sentence_alpha = nn.Parameter(torch.zeros(1))
        self.subgraph_alpha = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _parse_levels(raw_levels: str) -> set[str]:
        aliases = {
            "all": "token,sentence,subgraph",
            "input": "token",
            "tokens": "token",
            "sent": "sentence",
            "sentences": "sentence",
            "graph": "subgraph",
            "subgraphs": "subgraph",
        }
        levels = {
            expanded
            for part in raw_levels.replace("+", ",").split(",")
            for expanded in aliases.get(
                part.strip().lower(),
                part.strip().lower(),
            ).split(",")
            if expanded
        }
        levels = {
            level
            for level in levels
            if level.strip()
        }
        levels = levels or {"token"}
        invalid = levels - {"token", "sentence", "subgraph"}
        if invalid:
            raise ValueError(
                "fusion_levels must contain only: token, sentence, subgraph"
            )
        return levels

    def _basic_fusion(
        self,
        token_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.fusion in {"add", "sum"}:
            return token_embeddings + graph_embeddings, None
        if self.fusion in {"context_gated", "contextual"}:
            context = (
                token_embeddings if context_embeddings is None else context_embeddings
            )
            gate_input = torch.cat(
                [token_embeddings, graph_embeddings, context],
                dim=-1,
            )
            gate = torch.sigmoid(self.gate(gate_input))
            fused = gate * token_embeddings + (1.0 - gate) * graph_embeddings
            return (
                token_embeddings
                + torch.tanh(self.token_alpha) * self.norm(fused),
                gate,
            )
        gate_input = torch.cat([token_embeddings, graph_embeddings], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        update = (1.0 - gate) * graph_embeddings
        return token_embeddings + torch.tanh(self.token_alpha) * update, gate

    @staticmethod
    def _gate_stats(prefix: str, gate: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if gate is None:
            return {}
        graph_share = 1.0 - gate.detach()
        return {
            f"{prefix}_text_gate_mean": gate.detach().mean(),
            f"{prefix}_graph_share_mean": graph_share.mean(),
            f"{prefix}_graph_share_min": graph_share.min(),
            f"{prefix}_graph_share_max": graph_share.max(),
        }

    def forward(
        self,
        token_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor | None = None,
        subgraph_embeddings: torch.Tensor | None = None,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        graph_embeddings = (
            self.graph_dropout(graph_embeddings) * self.graph_fusion_scale
        )
        stats: dict[str, torch.Tensor] = {}
        hidden = token_embeddings
        if "token" in self.fusion_levels:
            hidden, gate = self._basic_fusion(
                hidden,
                graph_embeddings,
                context_embeddings=context_embeddings,
            )
            stats.update(self._gate_stats("token", gate))
            stats["token_alpha_tanh"] = torch.tanh(self.token_alpha).reshape(())
        if "sentence" in self.fusion_levels:
            sentence_graph = graph_embeddings.mean(dim=1, keepdim=True).expand_as(hidden)
            sentence_input = torch.cat([hidden, sentence_graph], dim=-1)
            gate = torch.sigmoid(self.sentence_gate(sentence_input))
            hidden = hidden + torch.tanh(self.sentence_alpha) * (
                (1.0 - gate) * sentence_graph
            )
            stats.update(self._gate_stats("sentence", gate))
            stats["sentence_alpha_tanh"] = torch.tanh(self.sentence_alpha).reshape(())
        if "subgraph" in self.fusion_levels and subgraph_embeddings is not None:
            subgraph = self.graph_dropout(subgraph_embeddings) * self.graph_fusion_scale
            subgraph_input = torch.cat([hidden, subgraph], dim=-1)
            gate = torch.sigmoid(self.subgraph_gate(subgraph_input))
            hidden = hidden + torch.tanh(self.subgraph_alpha) * (
                (1.0 - gate) * subgraph
            )
            stats.update(self._gate_stats("subgraph", gate))
            stats["subgraph_alpha_tanh"] = torch.tanh(self.subgraph_alpha).reshape(())
        if return_stats:
            return hidden, stats
        return hidden


class GraphCausalLM(nn.Module):
    """Persian Graph-LM with Rakhshai graph encoder and gated token fusion."""

    def __init__(self, config: GraphLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.graph_encoder = (
            None
            if config.graph_encoder.lower() == "none"
            else RakhshaiGraphEncoder(config)
        )
        self.fusion = GraphTokenFusion(config)
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.node_relation_head = nn.Linear(
            config.d_model,
            max(1, config.graph_edge_types),
        )
        self.dropout = nn.Dropout(config.dropout)
        # The lm_head is tied to token_embedding, so the default N(0, 1)
        # embedding init produces logits on the order of sqrt(d_model) and an
        # initial cross-entropy far above ln(vocab_size). Small-std init keeps
        # the untrained model near the uniform distribution.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[config.pad_token_id].zero_()

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )

    def _graph_embedding_table(
        self,
        graph_data: Data | None,
        token_node_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        token_table = self.token_embedding.weight
        if self.graph_encoder is None or graph_data is None or token_node_ids is None:
            return torch.zeros_like(token_table)
        token_node_ids = token_node_ids.to(token_table.device)
        valid = token_node_ids >= 0
        if not valid.any():
            return torch.zeros_like(token_table)
        node_features = token_table.new_zeros((graph_data.num_nodes, token_table.size(1)))
        node_features[token_node_ids[valid]] = token_table[valid]
        node_embeddings = self.graph_encoder(node_features, graph_data)
        graph_table = torch.zeros_like(token_table)
        graph_table[valid] = node_embeddings[token_node_ids[valid]]
        return graph_table

    def _encode_graph_nodes(
        self,
        graph_data: Data | None,
        token_node_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        token_table = self.token_embedding.weight
        if self.graph_encoder is None or graph_data is None or token_node_ids is None:
            return None
        token_node_ids = token_node_ids.to(token_table.device)
        valid = token_node_ids >= 0
        if not valid.any():
            return None
        node_features = token_table.new_zeros((graph_data.num_nodes, token_table.size(1)))
        node_features[token_node_ids[valid]] = token_table[valid]
        return self.graph_encoder(node_features, graph_data)

    def _graph_embeddings_for_input(
        self,
        input_ids: torch.Tensor,
        graph_data: Data | None,
        token_node_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        graph_embeddings, _ = self._graph_context_for_input(
            input_ids,
            graph_data,
            token_node_ids,
        )
        return graph_embeddings

    def _graph_context_for_input(
        self,
        input_ids: torch.Tensor,
        graph_data: Data | None,
        token_node_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_table = self.token_embedding.weight
        output_shape = (*input_ids.shape, token_table.size(1))
        if self.graph_encoder is None or graph_data is None or token_node_ids is None:
            return token_table.new_zeros(output_shape), None

        token_node_ids = token_node_ids.to(token_table.device)
        valid_tokens = token_node_ids >= 0
        if not valid_tokens.any():
            return token_table.new_zeros(output_shape), None

        node_features = token_table.new_zeros((graph_data.num_nodes, token_table.size(1)))
        node_features[token_node_ids[valid_tokens]] = token_table[valid_tokens]
        node_embeddings = self.graph_encoder(node_features, graph_data)

        flat_input = input_ids.reshape(-1).to(token_table.device)
        flat_output = token_table.new_zeros((flat_input.numel(), token_table.size(1)))
        in_range = (flat_input >= 0) & (flat_input < token_node_ids.numel())
        node_ids = token_node_ids[flat_input.clamp(0, token_node_ids.numel() - 1)]
        valid_positions = in_range & (node_ids >= 0)
        if valid_positions.any():
            flat_output[valid_positions] = node_embeddings[node_ids[valid_positions]]
        graph_embeddings = flat_output.reshape(output_shape)

        node_type_id = getattr(graph_data, "node_type_id", None)
        if node_type_id is not None:
            node_type_id = node_type_id.to(token_table.device)
            mask = node_type_id != 0
            if not mask.any():
                mask = valid_tokens.new_zeros((node_embeddings.size(0),), dtype=torch.bool)
                mask[token_node_ids[valid_tokens]] = True
        else:
            mask = valid_tokens.new_zeros((node_embeddings.size(0),), dtype=torch.bool)
            mask[token_node_ids[valid_tokens]] = True
        if not mask.any():
            return graph_embeddings, None
        subgraph_embedding = node_embeddings[mask].mean(dim=0).view(1, 1, -1)
        subgraph_embedding = subgraph_embedding.expand(
            input_ids.size(0),
            input_ids.size(1),
            -1,
        )
        return graph_embeddings, subgraph_embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        graph_data: Data | None = None,
        token_node_ids: torch.Tensor | None = None,
        graph_embeddings: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len :]
            if labels is not None:
                labels = labels[:, -self.config.max_seq_len :]
            if graph_embeddings is not None:
                graph_embeddings = graph_embeddings[:, -self.config.max_seq_len :]
            seq_len = self.config.max_seq_len

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_embeddings = self.token_embedding(input_ids)
        has_graph = graph_embeddings is not None or (
            self.graph_encoder is not None
            and graph_data is not None
            and token_node_ids is not None
        )
        subgraph_embeddings = None
        if not has_graph:
            hidden = token_embeddings
            fusion_stats = {}
        else:
            if graph_embeddings is None:
                graph_embeddings, subgraph_embeddings = self._graph_context_for_input(
                    input_ids,
                    graph_data,
                    token_node_ids,
                )
            fused = self.fusion(
                token_embeddings,
                graph_embeddings,
                subgraph_embeddings=subgraph_embeddings,
                return_stats=True,
            )
            hidden, fusion_stats = fused
        hidden = self.dropout(hidden + self.position_embedding(positions))
        padding_mask = input_ids.eq(self.config.pad_token_id)
        mask = self._causal_mask(seq_len, input_ids.device)
        for layer in self.transformer_layers:
            if has_graph and self.config.fusion_layers == "all":
                fused = self.fusion(
                    hidden,
                    graph_embeddings,
                    context_embeddings=hidden,
                    subgraph_embeddings=subgraph_embeddings,
                    return_stats=True,
                )
                hidden, layer_stats = fused
                for key, value in layer_stats.items():
                    fusion_stats[f"layer_{key}"] = value
            hidden = layer(hidden, src_mask=mask, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        output = {"logits": logits}
        if return_hidden:
            output["hidden"] = hidden
            if has_graph:
                output["graph_embeddings"] = graph_embeddings
            if subgraph_embeddings is not None:
                output["subgraph_embeddings"] = subgraph_embeddings
        if fusion_stats:
            output["fusion_stats"] = fusion_stats
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tokenizer: PersianTokenizer,
        *,
        graph_data: Data | None = None,
        token_node_ids: torch.Tensor | None = None,
        graph_memory: GraphMemoryArtifact | None = None,
        graph_memory_config: GraphMemoryConfig | None = None,
        generation_config: GenerationConfig | None = None,
        dynamic_graph_config: dict[str, object] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        self.eval()
        cfg = generation_config or GenerationConfig(eos_token_id=tokenizer.eos_id)
        if max_new_tokens is not None:
            cfg.max_new_tokens = max_new_tokens
        device = next(self.parameters()).device
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        if ids and ids[-1] == tokenizer.eos_id:
            ids = ids[:-1]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        if graph_data is not None:
            graph_data = graph_data.to(device)

        memory_context = None
        if graph_memory is not None and self.graph_encoder is not None:
            memory_context = graph_memory.retrieve(
                prompt,
                tokenizer,
                config=graph_memory_config,
                device=device,
            )
            if memory_context.graph_data is not None and memory_context.token_node_ids is not None:
                graph_data = memory_context.graph_data
                token_node_ids = memory_context.token_node_ids

        for _ in range(cfg.max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            step_graph_data = graph_data
            step_token_node_ids = token_node_ids
            if dynamic_graph_config is not None and self.graph_encoder is not None:
                from .graph_builder import build_graph_lm_graph_from_token_ids

                local_graph = build_graph_lm_graph_from_token_ids(
                    context.detach().cpu().tolist(),
                    tokenizer,
                    window_size=int(dynamic_graph_config.get("window_size", 4)),
                    weighting=str(dynamic_graph_config.get("weighting", "distance")),
                    min_edge_weight=float(dynamic_graph_config.get("min_edge_weight", 0.0)),
                    top_k=dynamic_graph_config.get("top_k"),  # type: ignore[arg-type]
                    directed=True,
                )
                step_graph_data = local_graph.to_pyg_data().to(device)
                step_token_node_ids = local_graph.token_node_ids(tokenizer.vocab_size).to(device)
            logits = self(
                context,
                graph_data=step_graph_data,
                token_node_ids=step_token_node_ids,
            )["logits"][:, -1, :]
            if cfg.repetition_penalty and cfg.repetition_penalty != 1.0:
                seen = set(input_ids[0].tolist())
                for token_id in seen:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= cfg.repetition_penalty
                    else:
                        logits[0, token_id] *= cfg.repetition_penalty
            logits = logits / max(cfg.temperature, 1e-6)
            if cfg.top_k and cfg.top_k > 0:
                values, indices = torch.topk(logits, k=min(cfg.top_k, logits.size(-1)))
                filtered = torch.full_like(logits, float("-inf"))
                filtered.scatter_(1, indices, values)
                logits = filtered
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            generated_count = input_ids.size(1) - len(ids)
            if (
                generated_count >= cfg.min_new_tokens
                and int(next_id.item()) == cfg.eos_token_id
            ):
                break
        return tokenizer.decode(input_ids[0].tolist())

    def save_pretrained(
        self,
        output_dir: str | Path,
        *,
        tokenizer: PersianTokenizer,
        graph_config: dict[str, object],
        graph_data: Data | None = None,
        token_node_ids: torch.Tensor | None = None,
        graph_memory: GraphMemoryArtifact | None = None,
        graph_memory_config: GraphMemoryConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "model.pt")
        with (output_path / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)
        tokenizer.save(output_path / "tokenizer.json")
        with (output_path / "graph_config.json").open("w", encoding="utf-8") as f:
            json.dump(graph_config, f, ensure_ascii=False, indent=2)
        gen_cfg = generation_config or GenerationConfig(eos_token_id=tokenizer.eos_id)
        with (output_path / "generation_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(gen_cfg), f, ensure_ascii=False, indent=2)
        graph_artifact = output_path / "graph.pt"
        if graph_data is not None and token_node_ids is not None:
            payload: dict[str, object] = {
                "num_nodes": int(graph_data.num_nodes),
                "edge_index": graph_data.edge_index.detach().cpu(),
                "token_node_ids": token_node_ids.detach().cpu(),
            }
            edge_weight = getattr(graph_data, "edge_weight", None)
            if edge_weight is not None:
                payload["edge_weight"] = edge_weight.detach().cpu()
            edge_type = getattr(graph_data, "edge_type", None)
            if edge_type is not None:
                payload["edge_type"] = edge_type.detach().cpu()
            node_type_id = getattr(graph_data, "node_type_id", None)
            if node_type_id is not None:
                payload["node_type_id"] = node_type_id.detach().cpu()
            torch.save(payload, graph_artifact)
            memory = graph_memory or GraphMemoryArtifact.from_pyg_data(
                graph_data,
                token_node_ids,
                tokenizer,
                graph_config,
            )
            memory.save(output_path, graph_memory_config)
        elif graph_artifact.exists():
            graph_artifact.unlink()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["GraphCausalLM", PersianTokenizer, GenerationConfig, dict[str, object]]:
        model_path = Path(model_dir)
        with (model_path / "config.json").open(encoding="utf-8") as f:
            config = GraphLMConfig(**json.load(f))
        model = cls(config)
        state = torch.load(model_path / "model.pt", map_location=map_location)
        model.load_state_dict(state, strict=False)
        tokenizer = PersianTokenizer.load(model_path / "tokenizer.json")
        with (model_path / "generation_config.json").open(encoding="utf-8") as f:
            generation_config = GenerationConfig(**json.load(f))
        with (model_path / "graph_config.json").open(encoding="utf-8") as f:
            graph_config = json.load(f)
        return model, tokenizer, generation_config, graph_config

    @staticmethod
    def load_graph_artifacts(
        model_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple[Data | None, torch.Tensor | None]:
        graph_path = Path(model_dir) / "graph.pt"
        if not graph_path.exists():
            return None, None
        state = torch.load(graph_path, map_location=map_location)
        edge_index = state["edge_index"].long()
        data = Data(edge_index=edge_index, num_nodes=int(state["num_nodes"]))
        edge_weight = state.get("edge_weight")
        if edge_weight is not None:
            data.edge_weight = edge_weight.float()
        edge_type = state.get("edge_type")
        if edge_type is not None:
            data.edge_type = edge_type.long()
        node_type_id = state.get("node_type_id")
        if node_type_id is not None:
            data.node_type_id = node_type_id.long()
        return data, state["token_node_ids"].long()


def perplexity(loss: float) -> float:
    """Perplexity of a mean next-token cross-entropy loss.

    The clamp only guards against float overflow (exp overflows above ~709);
    any realistic loss is reported unclamped.
    """
    return float(math.exp(min(loss, 700.0)))

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
from torch_geometric.nn import GATConv, GCNConv, MessagePassing

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
    fusion: str = "gated"
    fusion_layers: str = "input"
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


class RakhshaiGraphEncoder(nn.Module):
    """GNN encoder that creates graph-aware token embeddings."""

    def __init__(self, config: GraphLMConfig):
        super().__init__()
        encoder = config.graph_encoder.lower()
        if encoder == "gcn":
            self.conv1 = GCNConv(config.d_model, config.graph_hidden_dim)
            self.conv2 = GCNConv(config.graph_hidden_dim, config.d_model)
        elif encoder == "graphsage":
            self.conv1 = WeightedSAGEConv(config.d_model, config.graph_hidden_dim)
            self.conv2 = WeightedSAGEConv(config.graph_hidden_dim, config.d_model)
        elif encoder == "gat":
            self.conv1 = GATConv(
                config.d_model,
                config.graph_hidden_dim,
                heads=config.graph_heads,
                dropout=config.dropout,
                edge_dim=1,
            )
            self.conv2 = GATConv(
                config.graph_hidden_dim * config.graph_heads,
                config.d_model,
                heads=1,
                concat=False,
                dropout=config.dropout,
                edge_dim=1,
            )
        else:
            raise ValueError("graph_encoder must be one of: gcn, graphsage, gat")
        self.encoder = encoder
        self.dropout = nn.Dropout(config.dropout)
        self.edge_type_bias = (
            nn.Embedding(config.graph_edge_types, 1)
            if config.graph_edge_types > 1
            else None
        )

    def forward(self, node_features: torch.Tensor, graph_data: Data) -> torch.Tensor:
        edge_index = graph_data.edge_index.to(node_features.device)
        edge_weight = getattr(graph_data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(node_features.device)
        edge_type = getattr(graph_data, "edge_type", None)
        if (
            self.edge_type_bias is not None
            and edge_weight is not None
            and edge_type is not None
        ):
            edge_type = edge_type.to(node_features.device).clamp(
                0,
                self.edge_type_bias.num_embeddings - 1,
            )
            edge_weight = edge_weight * torch.sigmoid(self.edge_type_bias(edge_type).squeeze(-1))

        if self.encoder == "gcn":
            x = self.conv1(node_features, edge_index, edge_weight)
        elif self.encoder == "gat":
            edge_attr = None if edge_weight is None else edge_weight.unsqueeze(-1)
            x = self.conv1(node_features, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv1(node_features, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        if self.encoder == "gcn":
            return self.conv2(x, edge_index, edge_weight)
        if self.encoder == "gat":
            edge_attr = None if edge_weight is None else edge_weight.unsqueeze(-1)
            return self.conv2(x, edge_index, edge_attr=edge_attr)
        return self.conv2(x, edge_index, edge_weight)


class GraphTokenFusion(nn.Module):
    """Combine normal token embeddings with graph embeddings."""

    def __init__(self, config: GraphLMConfig):
        super().__init__()
        self.fusion = config.fusion.lower()
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
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fusion in {"add", "sum"}:
            return token_embeddings + graph_embeddings
        if self.fusion in {"context_gated", "contextual"}:
            context = token_embeddings if context_embeddings is None else context_embeddings
            gate_input = torch.cat([token_embeddings, graph_embeddings, context], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            fused = gate * token_embeddings + (1.0 - gate) * graph_embeddings
            return self.norm(token_embeddings + fused)
        else:
            gate_input = torch.cat([token_embeddings, graph_embeddings], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        return gate * token_embeddings + (1.0 - gate) * graph_embeddings


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
        self.dropout = nn.Dropout(config.dropout)

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

    def _graph_embeddings_for_input(
        self,
        input_ids: torch.Tensor,
        graph_data: Data | None,
        token_node_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        token_table = self.token_embedding.weight
        output_shape = (*input_ids.shape, token_table.size(1))
        if self.graph_encoder is None or graph_data is None or token_node_ids is None:
            return token_table.new_zeros(output_shape)

        token_node_ids = token_node_ids.to(token_table.device)
        valid_tokens = token_node_ids >= 0
        if not valid_tokens.any():
            return token_table.new_zeros(output_shape)

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
        return flat_output.reshape(output_shape)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        graph_data: Data | None = None,
        token_node_ids: torch.Tensor | None = None,
        graph_embeddings: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
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
            self.graph_encoder is not None and graph_data is not None and token_node_ids is not None
        )
        if not has_graph:
            hidden = token_embeddings
        else:
            if graph_embeddings is None:
                graph_embeddings = self._graph_embeddings_for_input(
                    input_ids,
                    graph_data,
                    token_node_ids,
                )
            hidden = self.fusion(token_embeddings, graph_embeddings)
        hidden = self.dropout(hidden + self.position_embedding(positions))
        padding_mask = input_ids.eq(self.config.pad_token_id)
        mask = self._causal_mask(seq_len, input_ids.device)
        for layer in self.transformer_layers:
            if has_graph and self.config.fusion_layers == "all":
                hidden = self.fusion(hidden, graph_embeddings, context_embeddings=hidden)
            hidden = layer(hidden, src_mask=mask, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden)
        output = {"logits": logits}
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
            torch.save(payload, graph_artifact)
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
        model.load_state_dict(state)
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
        return data, state["token_node_ids"].long()


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))

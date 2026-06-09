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
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

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
    pad_token_id: int = 0


@dataclass
class GenerationConfig:
    max_new_tokens: int = 100
    min_new_tokens: int = 0
    temperature: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    eos_token_id: int = 3


class RakhshaiGraphEncoder(nn.Module):
    """GNN encoder that creates graph-aware token embeddings."""

    def __init__(self, config: GraphLMConfig):
        super().__init__()
        encoder = config.graph_encoder.lower()
        if encoder == "gcn":
            self.conv1 = GCNConv(config.d_model, config.graph_hidden_dim)
            self.conv2 = GCNConv(config.graph_hidden_dim, config.d_model)
        elif encoder == "graphsage":
            self.conv1 = SAGEConv(config.d_model, config.graph_hidden_dim)
            self.conv2 = SAGEConv(config.graph_hidden_dim, config.d_model)
        elif encoder == "gat":
            self.conv1 = GATConv(
                config.d_model,
                config.graph_hidden_dim,
                heads=config.graph_heads,
                dropout=config.dropout,
            )
            self.conv2 = GATConv(
                config.graph_hidden_dim * config.graph_heads,
                config.d_model,
                heads=1,
                concat=False,
                dropout=config.dropout,
            )
        else:
            raise ValueError("graph_encoder must be one of: gcn, graphsage, gat")
        self.encoder = encoder
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, node_features: torch.Tensor, graph_data: Data) -> torch.Tensor:
        edge_index = graph_data.edge_index.to(node_features.device)
        edge_weight = getattr(graph_data, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(node_features.device)

        if self.encoder == "gcn":
            x = self.conv1(node_features, edge_index, edge_weight)
        else:
            x = self.conv1(node_features, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        if self.encoder == "gcn":
            return self.conv2(x, edge_index, edge_weight)
        return self.conv2(x, edge_index)


class GraphTokenFusion(nn.Module):
    """Combine normal token embeddings with graph embeddings."""

    def __init__(self, config: GraphLMConfig):
        super().__init__()
        self.fusion = config.fusion.lower()
        if self.fusion == "gated":
            self.gate = nn.Linear(config.d_model * 2, config.d_model)
        elif self.fusion in {"add", "sum"}:
            self.gate = None
        else:
            raise ValueError("fusion must be one of: gated, add")

    def forward(
        self,
        token_embeddings: torch.Tensor,
        graph_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion in {"add", "sum"}:
            return token_embeddings + graph_embeddings
        gate = torch.sigmoid(self.gate(torch.cat([token_embeddings, graph_embeddings], dim=-1)))
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
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(config.dropout)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
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

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        graph_data: Data | None = None,
        token_node_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len :]
            if labels is not None:
                labels = labels[:, -self.config.max_seq_len :]
            seq_len = self.config.max_seq_len

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        token_embeddings = self.token_embedding(input_ids)
        graph_table = self._graph_embedding_table(graph_data, token_node_ids)
        graph_embeddings = F.embedding(input_ids, graph_table)
        if self.graph_encoder is None or graph_data is None or token_node_ids is None:
            hidden = token_embeddings
        else:
            hidden = self.fusion(token_embeddings, graph_embeddings)
        hidden = self.dropout(hidden + self.position_embedding(positions))
        padding_mask = input_ids.eq(self.config.pad_token_id)
        hidden = self.transformer(
            hidden,
            mask=self._causal_mask(seq_len, input_ids.device),
            src_key_padding_mask=padding_mask,
        )
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
            logits = self(
                context,
                graph_data=graph_data,
                token_node_ids=token_node_ids,
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


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))

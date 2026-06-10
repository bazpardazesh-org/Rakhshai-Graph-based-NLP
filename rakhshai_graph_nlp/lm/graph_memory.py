"""Prompt-aware graph memory for Graph-LM generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch_geometric.data import Data

from .graph_builder import GraphLMGraph, build_graph_lm_graph
from .tokenizer import PersianTokenizer


@dataclass
class GraphMemoryConfig:
    """Retrieval controls for graph memory at generation time."""

    enabled: bool = True
    top_k_nodes: int = 32
    depth: int = 1
    max_edges: int = 256
    min_score: float = 0.0
    relation_weights: dict[str, float] | None = None


@dataclass
class RetrievedGraphContext:
    """A retrieved PyG subgraph and lightweight retrieval diagnostics."""

    graph_data: Data | None
    token_node_ids: torch.Tensor | None
    report: dict[str, object]


class GraphMemoryArtifact:
    """Serializable graph memory built from a Graph-LM corpus graph."""

    def __init__(
        self,
        *,
        nodes: Sequence[str],
        token_to_node: dict[int, int],
        graph_config: dict[str, object],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,
        node_type_id: torch.Tensor | None = None,
    ) -> None:
        self.nodes = list(nodes)
        self.token_to_node = {int(k): int(v) for k, v in token_to_node.items()}
        self.graph_config = dict(graph_config)
        self.edge_index = edge_index.long().cpu()
        self.edge_weight = (
            edge_weight.float().cpu()
            if edge_weight is not None
            else torch.ones((self.edge_index.size(1),), dtype=torch.float32)
        )
        self.edge_type = edge_type.long().cpu() if edge_type is not None else None
        self.node_type_id = node_type_id.long().cpu() if node_type_id is not None else None
        self._relation_names = self._relation_names_from_config()

    @classmethod
    def from_graph(cls, graph: GraphLMGraph) -> "GraphMemoryArtifact":
        data = graph.to_pyg_data()
        return cls(
            nodes=graph.nodes,
            token_to_node=graph.token_to_node,
            graph_config=graph.graph_config,
            edge_index=data.edge_index,
            edge_weight=getattr(data, "edge_weight", None),
            edge_type=getattr(data, "edge_type", None),
            node_type_id=getattr(data, "node_type_id", None),
        )

    @classmethod
    def from_pyg_data(
        cls,
        graph_data: Data,
        token_node_ids: torch.Tensor,
        tokenizer: PersianTokenizer,
        graph_config: dict[str, object],
    ) -> "GraphMemoryArtifact":
        token_to_node = {
            int(token_id): int(node_id)
            for token_id, node_id in enumerate(token_node_ids.detach().cpu().tolist())
            if int(node_id) >= 0
        }
        node_names = [f"node:{idx}" for idx in range(int(graph_data.num_nodes))]
        for token_id, node_id in token_to_node.items():
            node_names[node_id] = tokenizer.id_to_token.get(token_id, node_names[node_id])
        return cls(
            nodes=node_names,
            token_to_node=token_to_node,
            graph_config=graph_config,
            edge_index=graph_data.edge_index.detach().cpu(),
            edge_weight=getattr(graph_data, "edge_weight", None),
            edge_type=getattr(graph_data, "edge_type", None),
            node_type_id=getattr(graph_data, "node_type_id", None),
        )

    @classmethod
    def from_corpus(
        cls,
        texts: Sequence[str],
        tokenizer: PersianTokenizer,
        graph_config: dict[str, object],
    ) -> "GraphMemoryArtifact":
        graph = build_graph_lm_graph(
            texts,
            tokenizer,
            window_size=int(graph_config.get("window_size", 4)),
            min_count=int(graph_config.get("min_count", 1)),
            weighting=str(graph_config.get("weighting", "distance")),
            min_edge_weight=float(graph_config.get("min_edge_weight", 0.0)),
            top_k=graph_config.get("top_k"),  # type: ignore[arg-type]
            directed=bool(graph_config.get("directed", False)),
            graph_scope=str(graph_config.get("graph_scope", "document")),
            context_node_type=str(graph_config.get("context_node_type", "none")),
            graph_relations=graph_config.get("enabled_relations"),  # type: ignore[arg-type]
            relation_weights=graph_config.get("relation_weights"),  # type: ignore[arg-type]
            semantic_similarity_threshold=float(
                graph_config.get("semantic_similarity_threshold", 0.6)
            ),
            semantic_top_k=graph_config.get("semantic_top_k"),  # type: ignore[arg-type]
            topic_top_k=int(graph_config.get("topic_top_k", 8)),
        )
        return cls.from_graph(graph)

    def _relation_names_from_config(self) -> dict[int, str]:
        edge_types = self.graph_config.get("edge_types", {})
        if not isinstance(edge_types, dict):
            return {}
        return {int(value): str(key) for key, value in edge_types.items()}

    def token_node_ids(self, vocab_size: int, *, device: torch.device | None = None) -> torch.Tensor:
        ids = [self.token_to_node.get(i, -1) for i in range(vocab_size)]
        tensor = torch.tensor(ids, dtype=torch.long)
        return tensor if device is None else tensor.to(device)

    def to_pyg_data(self, *, device: torch.device | None = None) -> Data:
        data = Data(edge_index=self.edge_index.clone(), num_nodes=len(self.nodes))
        data.edge_weight = self.edge_weight.clone()
        if self.edge_type is not None:
            data.edge_type = self.edge_type.clone()
        if self.node_type_id is not None:
            data.node_type_id = self.node_type_id.clone()
        return data if device is None else data.to(device)

    def save(self, model_dir: str | Path, config: GraphMemoryConfig | None = None) -> None:
        output_path = Path(model_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "version": 1,
            "nodes": self.nodes,
            "token_to_node": self.token_to_node,
            "graph_config": self.graph_config,
            "edge_index": self.edge_index,
            "edge_weight": self.edge_weight,
        }
        if self.edge_type is not None:
            payload["edge_type"] = self.edge_type
        if self.node_type_id is not None:
            payload["node_type_id"] = self.node_type_id
        torch.save(payload, output_path / "graph_memory.pt")
        cfg = config or GraphMemoryConfig()
        with (output_path / "graph_memory_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["GraphMemoryArtifact" | None, GraphMemoryConfig]:
        model_path = Path(model_dir)
        config_path = model_path / "graph_memory_config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as f:
                config = GraphMemoryConfig(**json.load(f))
        else:
            config = GraphMemoryConfig()
        memory_path = model_path / "graph_memory.pt"
        if not memory_path.exists():
            return None, config
        state = torch.load(memory_path, map_location=map_location)
        return (
            cls(
                nodes=state["nodes"],
                token_to_node={int(k): int(v) for k, v in state["token_to_node"].items()},
                graph_config=state.get("graph_config", {}),
                edge_index=state["edge_index"],
                edge_weight=state.get("edge_weight"),
                edge_type=state.get("edge_type"),
                node_type_id=state.get("node_type_id"),
            ),
            config,
        )

    def retrieve(
        self,
        prompt: str,
        tokenizer: PersianTokenizer,
        *,
        config: GraphMemoryConfig | None = None,
        device: torch.device | None = None,
    ) -> RetrievedGraphContext:
        cfg = config or GraphMemoryConfig()
        if not cfg.enabled:
            return RetrievedGraphContext(None, None, {"enabled": False})

        prompt_tokens = tokenizer.tokenize(prompt)
        seed_nodes = [
            self.token_to_node[tokenizer.token_to_id[token]]
            for token in prompt_tokens
            if token in tokenizer.token_to_id
            and tokenizer.token_to_id[token] in self.token_to_node
        ]
        seed_nodes = sorted(set(seed_nodes))
        if not seed_nodes:
            return RetrievedGraphContext(
                None,
                None,
                {
                    "enabled": True,
                    "fallback": "no_prompt_nodes",
                    "prompt_tokens": prompt_tokens,
                    "retrieved_nodes": 0,
                    "retrieved_edges": 0,
                },
            )

        adjacency: dict[int, list[tuple[int, int, float, int | None]]] = {}
        for edge_id, (src, dst) in enumerate(self.edge_index.t().tolist()):
            edge_type = int(self.edge_type[edge_id]) if self.edge_type is not None else None
            weight = float(self.edge_weight[edge_id])
            relation = self._relation_names.get(edge_type, "") if edge_type is not None else ""
            relation_weight = 1.0
            if cfg.relation_weights and relation:
                relation_weight = float(cfg.relation_weights.get(relation, 1.0))
            score = weight * relation_weight
            adjacency.setdefault(int(src), []).append((int(dst), edge_id, score, edge_type))

        scores = {node: 1.0 for node in seed_nodes}
        frontier = set(seed_nodes)
        depth = max(0, int(cfg.depth))
        for hop in range(depth):
            next_frontier: set[int] = set()
            decay = 1.0 / float(hop + 2)
            for src in frontier:
                for dst, _, edge_score, _ in adjacency.get(src, []):
                    score = scores[src] + edge_score * decay
                    if score > scores.get(dst, float("-inf")):
                        scores[dst] = score
                    next_frontier.add(dst)
            frontier = next_frontier

        selected = [
            node
            for node, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if score >= cfg.min_score
        ][: max(1, int(cfg.top_k_nodes))]
        selected_set = set(selected)
        if not selected:
            return RetrievedGraphContext(
                None,
                None,
                {"enabled": True, "fallback": "empty_after_scoring", "retrieved_nodes": 0},
            )

        candidate_edges: list[tuple[int, int, int, float]] = []
        for edge_id, (src, dst) in enumerate(self.edge_index.t().tolist()):
            src = int(src)
            dst = int(dst)
            if src not in selected_set or dst not in selected_set:
                continue
            edge_score = scores.get(src, 0.0) + scores.get(dst, 0.0) + float(self.edge_weight[edge_id])
            candidate_edges.append((edge_id, src, dst, edge_score))
        candidate_edges.sort(key=lambda item: item[3], reverse=True)
        candidate_edges = candidate_edges[: max(0, int(cfg.max_edges))]

        remap = {old: new for new, old in enumerate(selected)}
        if candidate_edges:
            local_edges = torch.tensor(
                [[remap[src], remap[dst]] for _, src, dst, _ in candidate_edges],
                dtype=torch.long,
            ).t()
            edge_ids = torch.tensor([edge_id for edge_id, _, _, _ in candidate_edges], dtype=torch.long)
            edge_weight = self.edge_weight[edge_ids].clone()
        else:
            local_edges = torch.empty((2, 0), dtype=torch.long)
            edge_ids = torch.empty((0,), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)

        data = Data(edge_index=local_edges, num_nodes=len(selected))
        data.edge_weight = edge_weight
        if self.edge_type is not None:
            data.edge_type = self.edge_type[edge_ids].clone()
        if self.node_type_id is not None:
            selected_ids = torch.tensor(selected, dtype=torch.long)
            data.node_type_id = self.node_type_id[selected_ids].clone()

        token_node_ids = torch.full((tokenizer.vocab_size,), -1, dtype=torch.long)
        for token_id, old_node in self.token_to_node.items():
            if old_node in remap and token_id < tokenizer.vocab_size:
                token_node_ids[token_id] = remap[old_node]

        relation_counts: dict[str, int] = {}
        if self.edge_type is not None and candidate_edges:
            for edge_id, _, _, _ in candidate_edges:
                relation = self._relation_names.get(int(self.edge_type[edge_id]), "unknown")
                relation_counts[relation] = relation_counts.get(relation, 0) + 1

        report = {
            "enabled": True,
            "prompt_tokens": prompt_tokens,
            "seed_nodes": len(seed_nodes),
            "retrieved_nodes": len(selected),
            "retrieved_edges": len(candidate_edges),
            "top_nodes": [self.nodes[node] for node in selected[:10]],
            "relation_counts": relation_counts,
            "coverage": sum(1 for node in seed_nodes if node in selected_set)
            / max(1, len(seed_nodes)),
        }
        if device is not None:
            data = data.to(device)
            token_node_ids = token_node_ids.to(device)
        return RetrievedGraphContext(data, token_node_ids, report)

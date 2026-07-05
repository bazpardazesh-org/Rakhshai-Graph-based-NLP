"""Graph scalability helpers and native LM ablations."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .graph_builder import build_graph_lm_graph
from .model import GraphLMConfig
from .tokenizer import PersianTokenizer
from .trainer import LMTrainingConfig, train_graph_lm


@dataclass
class LMGraphAblationConfig:
    output_dir: str
    training_config: LMTrainingConfig
    model_config: GraphLMConfig | None = None
    graph_encoders: Sequence[str] = field(default_factory=lambda: ["none", "gat"])
    graph_scopes: Sequence[str] = field(default_factory=lambda: ["document"])
    relation_groups: dict[str, Sequence[str]] | None = None


def write_graph_feature_store(
    texts: Sequence[str],
    tokenizer: PersianTokenizer,
    output_path: str | Path,
    *,
    graph_scope: str = "document",
    graph_relations: Sequence[str] | None = None,
    graph_window_size: int = 4,
) -> dict[str, object]:
    """Write a compact graph-feature summary for a corpus/tokenizer pair."""

    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        window_size=graph_window_size,
        graph_scope=graph_scope,
        graph_relations=graph_relations,
    )
    token_node_ids = graph.token_node_ids(tokenizer.vocab_size)
    token_coverage = int(token_node_ids.ge(0).sum().item())
    data = graph.to_pyg_data()
    report = {
        "num_nodes": graph.graph_config.get("num_nodes", len(graph.nodes)),
        "num_edges": graph.graph_config.get("num_edges", int(data.edge_index.shape[1])),
        "edge_types": graph.graph_config.get("edge_types", {}),
        "enabled_relations": graph.graph_config.get("enabled_relations", []),
        "graph_scope": graph_scope,
        "token_node_coverage": token_coverage,
        "vocab_size": tokenizer.vocab_size,
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_lm_graph_ablation(
    texts: Sequence[str],
    config: LMGraphAblationConfig,
) -> dict[str, object]:
    """Run native Graph-LM ablations under a shared training recipe."""

    output_dir = Path(config.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    relation_groups = config.relation_groups or {"default": config.training_config.graph_relations or []}
    variants: list[dict[str, object]] = []
    for encoder in config.graph_encoders:
        scopes = ["none"] if encoder == "none" else list(config.graph_scopes)
        for scope in scopes:
            for group_name, relations in relation_groups.items():
                if encoder == "none" and group_name != next(iter(relation_groups)):
                    continue
                variant_name = f"{encoder}-{scope}-{group_name}"
                variant_dir = output_dir / variant_name
                variant_config = LMTrainingConfig(**asdict(config.training_config))
                variant_config.output_dir = str(variant_dir)
                if scope != "none":
                    variant_config.graph_scope = scope
                variant_config.graph_relations = list(relations) if relations else None
                metrics = train_graph_lm(
                    texts,
                    training_config=variant_config,
                    model_config=config.model_config,
                    graph_encoder=encoder,
                    fusion="gated",
                )
                variants.append(
                    {
                        "variant": variant_name,
                        "graph_encoder": encoder,
                        "graph_scope": scope,
                        "relation_group": group_name,
                        "relations": list(relations),
                        "best_perplexity": metrics.get("best_perplexity"),
                        "best_next_token_loss": metrics.get("best_next_token_loss"),
                        "graph_scalability": metrics.get("graph_scalability"),
                        "fusion_stats": metrics.get("fusion_stats", {}),
                        "checkpoint_dir": metrics.get("checkpoint_dir"),
                    }
                )
    report = {
        "variants": variants,
        "native_independence": {
            "uses_external_pretrained_lm": False,
            "uses_pretrained_embeddings": False,
            "uses_distillation": False,
            "uses_llm_synthetic_data": False,
            "uses_external_llm_judge": False,
        },
    }
    (output_dir / "lm_ablation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report

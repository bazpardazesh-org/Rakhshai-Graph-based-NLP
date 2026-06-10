"""Training loop for Graph causal language models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from .dataset import LMDataset, LMLoaders, build_lm_dataloaders
from .graph_builder import build_graph_lm_graph, build_graph_lm_graph_from_token_ids
from .model import GenerationConfig, GraphCausalLM, GraphLMConfig, perplexity
from .tokenizer import PersianTokenizer


@dataclass
class LMTrainingConfig:
    output_dir: str = "runs/graph-lm"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    validation_ratio: float = 0.1
    block_size: int = 128
    stride: int | None = None
    min_freq: int = 1
    max_vocab_size: int | None = None
    graph_window_size: int = 4
    graph_min_count: int = 1
    graph_weighting: str = "distance"
    graph_min_edge_weight: float = 0.0
    graph_top_k: int | None = None
    graph_directed: bool = False
    graph_scope: str = "document"
    context_node_type: str = "none"
    graph_relations: Sequence[str] | None = None
    relation_weights: dict[str, float] | None = None
    semantic_similarity_threshold: float = 0.6
    semantic_top_k: int | None = 4
    topic_top_k: int = 8
    graph_relation_mode: str = "bias"
    graph_pooling: str = "none"
    graph_node_importance: bool = False
    fusion_levels: str = "token"
    graph_fusion_scale: float = 1.0
    graph_fusion_dropout: float = 0.0
    dynamic_graph: bool = False
    tokenizer_type: str = "word"
    tokenizer_half_space: str = "preserve"
    tokenizer_morph_splitting: bool = False
    tokenizer_compound_verb_mode: str = "none"
    tokenizer_bpe_merges: int = 200
    device: str = "cpu"
    seed: int = 0


class LMTrainer:
    """Dedicated trainer with validation loss, perplexity and checkpoints."""

    def __init__(
        self,
        model: GraphCausalLM,
        tokenizer: PersianTokenizer,
        graph_data,
        token_node_ids: torch.Tensor | None,
        *,
        config: LMTrainingConfig,
        graph_config: dict[str, object],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.graph_data = graph_data
        self.token_node_ids = token_node_ids
        self.config = config
        self.graph_config = graph_config
        self.device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        if self.graph_data is not None:
            self.graph_data = self.graph_data.to(self.device)
        if self.token_node_ids is not None:
            self.token_node_ids = self.token_node_ids.to(self.device)
        self._last_fusion_stats: dict[str, float] = {}

    def _causal_dynamic_graph_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        step_embeddings: list[torch.Tensor] = []
        token_sequences = input_ids.detach().cpu().tolist()
        for step in range(input_ids.size(1)):
            prefixes = [
                [
                    int(token_id)
                    for token_id in sequence[: step + 1]
                    if int(token_id) != self.tokenizer.pad_id
                ]
                for sequence in token_sequences
            ]
            local_graph = build_graph_lm_graph_from_token_ids(
                prefixes,
                self.tokenizer,
                window_size=self.config.graph_window_size,
                weighting=self.config.graph_weighting,
                min_edge_weight=self.config.graph_min_edge_weight,
                top_k=self.config.graph_top_k,
                directed=True,
            )
            graph_data = local_graph.to_pyg_data().to(self.device)
            token_node_ids = local_graph.token_node_ids(self.tokenizer.vocab_size).to(self.device)
            current_ids = input_ids[:, step : step + 1]
            current_embeddings = self.model._graph_embeddings_for_input(
                current_ids,
                graph_data,
                token_node_ids,
            )
            step_embeddings.append(current_embeddings.squeeze(1))
        return torch.stack(step_embeddings, dim=1)

    def _build_loaders(
        self,
        dataset: LMDataset,
        validation_dataset: LMDataset | None,
    ) -> LMLoaders:
        if validation_dataset is None:
            return build_lm_dataloaders(
                dataset,
                batch_size=self.config.batch_size,
                validation_ratio=self.config.validation_ratio,
                seed=self.config.seed,
            )
        generator = torch.Generator().manual_seed(self.config.seed)
        return LMLoaders(
            train=DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=generator,
            ),
            validation=DataLoader(validation_dataset, batch_size=self.config.batch_size),
        )

    def _run_epoch(self, loader, optimizer: torch.optim.Optimizer | None = None) -> float:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        total_batches = 0
        fusion_sums: dict[str, float] = {}
        fusion_count = 0
        for input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            graph_data = self.graph_data
            token_node_ids = self.token_node_ids
            graph_embeddings = None
            if self.config.dynamic_graph and self.model.config.graph_encoder != "none":
                graph_data = None
                token_node_ids = None
                graph_embeddings = self._causal_dynamic_graph_embeddings(input_ids)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = self.model(
                input_ids,
                graph_data=graph_data,
                token_node_ids=token_node_ids,
                graph_embeddings=graph_embeddings,
                labels=labels,
            )
            loss = output["loss"]
            fusion_stats = output.get("fusion_stats", {})
            if fusion_stats:
                fusion_count += 1
                for key, value in fusion_stats.items():
                    fusion_sums[key] = fusion_sums.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        self._last_fusion_stats = {
            key: value / max(1, fusion_count) for key, value in fusion_sums.items()
        }
        return total_loss / max(1, total_batches)

    def train(
        self,
        dataset: LMDataset,
        validation_dataset: LMDataset | None = None,
    ) -> dict[str, object]:
        loaders = self._build_loaders(dataset, validation_dataset)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        history: list[dict[str, float | int]] = []
        best_val = float("inf")
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(loaders.train, optimizer)
            train_fusion_stats = dict(self._last_fusion_stats)
            if loaders.validation is not None:
                with torch.no_grad():
                    val_loss = self._run_epoch(loaders.validation)
                validation_fusion_stats = dict(self._last_fusion_stats)
            else:
                val_loss = train_loss
                validation_fusion_stats = train_fusion_stats
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "perplexity": perplexity(val_loss),
            }
            if train_fusion_stats:
                row["train_fusion"] = train_fusion_stats
            if validation_fusion_stats:
                row["validation_fusion"] = validation_fusion_stats
            history.append(row)
            if val_loss <= best_val:
                best_val = val_loss
                self.model.save_pretrained(
                    output_dir,
                    tokenizer=self.tokenizer,
                    graph_config=self.graph_config,
                    graph_data=self.graph_data,
                    token_node_ids=self.token_node_ids,
                    generation_config=GenerationConfig(eos_token_id=self.tokenizer.eos_id),
                )

        metrics = {
            "training_config": asdict(self.config),
            "model_config": asdict(self.model.config),
            "history": history,
            "best_validation_loss": best_val,
            "best_perplexity": perplexity(best_val),
            "checkpoint_dir": str(output_dir),
        }
        if history:
            last_validation_fusion = history[-1].get("validation_fusion")
            if last_validation_fusion:
                metrics["fusion_stats"] = last_validation_fusion
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return metrics


def _split_corpus(
    corpus: Sequence[str],
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if len(corpus) <= 1 or validation_ratio == 0:
        return list(corpus), []
    val_size = int(round(len(corpus) * validation_ratio))
    val_size = max(1, min(val_size, len(corpus) - 1))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(corpus), generator=generator).tolist()
    val_indices = set(permutation[:val_size])
    train = [text for idx, text in enumerate(corpus) if idx not in val_indices]
    validation = [text for idx, text in enumerate(corpus) if idx in val_indices]
    return train, validation


def _tokenizer_stats(
    tokenizer: PersianTokenizer,
    train_corpus: Sequence[str],
    validation_corpus: Sequence[str],
) -> dict[str, object]:
    train_tokens = [token for text in train_corpus for token in tokenizer.tokenize(text)]
    validation_tokens = [
        token for text in validation_corpus for token in tokenizer.tokenize(text)
    ]
    unk_count = sum(1 for token in validation_tokens if token not in tokenizer.token_to_id)
    validation_text_count = max(1, len(validation_corpus))
    return {
        "tokenizer_type": tokenizer.tokenizer_type,
        "vocab_size": tokenizer.vocab_size,
        "keep_half_space": tokenizer.keep_half_space,
        "morph_splitting": tokenizer.morph_splitting,
        "compound_verb_mode": tokenizer.compound_verb_mode,
        "bpe_merges": len(tokenizer.bpe_merges),
        "train_token_count": len(train_tokens),
        "validation_token_count": len(validation_tokens),
        "validation_unk_count": unk_count,
        "validation_unk_rate": unk_count / max(1, len(validation_tokens)),
        "avg_validation_tokens_per_text": len(validation_tokens) / validation_text_count,
    }


def train_graph_lm(
    texts: Iterable[str],
    *,
    training_config: LMTrainingConfig,
    model_config: GraphLMConfig | None = None,
    graph_encoder: str = "gat",
    fusion: str = "gated",
) -> dict[str, object]:
    corpus = [text.strip() for text in texts if text.strip()]
    if not corpus:
        raise ValueError("corpus is empty")
    torch.manual_seed(training_config.seed)
    train_corpus, validation_corpus = _split_corpus(
        corpus,
        validation_ratio=training_config.validation_ratio,
        seed=training_config.seed,
    )
    tokenizer = PersianTokenizer(
        min_freq=training_config.min_freq,
        max_vocab_size=training_config.max_vocab_size,
        tokenizer_type=training_config.tokenizer_type,
        keep_half_space=training_config.tokenizer_half_space == "preserve",
        morph_splitting=training_config.tokenizer_morph_splitting,
        compound_verb_mode=training_config.tokenizer_compound_verb_mode,
        bpe_num_merges=training_config.tokenizer_bpe_merges,
    ).fit(train_corpus)
    dataset = LMDataset(
        train_corpus,
        tokenizer,
        block_size=training_config.block_size,
        stride=training_config.stride,
    )
    validation_dataset = (
        LMDataset(
            validation_corpus,
            tokenizer,
            block_size=training_config.block_size,
            stride=training_config.stride,
        )
        if validation_corpus
        else None
    )
    graph = None
    if graph_encoder != "none":
        graph = build_graph_lm_graph(
            train_corpus,
            tokenizer,
            window_size=training_config.graph_window_size,
            min_count=training_config.graph_min_count,
            weighting=training_config.graph_weighting,
            min_edge_weight=training_config.graph_min_edge_weight,
            top_k=training_config.graph_top_k,
            directed=training_config.graph_directed,
            graph_scope=training_config.graph_scope,
            context_node_type=training_config.context_node_type,
            graph_relations=training_config.graph_relations,
            relation_weights=training_config.relation_weights,
            semantic_similarity_threshold=training_config.semantic_similarity_threshold,
            semantic_top_k=training_config.semantic_top_k,
            topic_top_k=training_config.topic_top_k,
        )
    cfg = model_config or GraphLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=training_config.block_size,
        graph_encoder=graph_encoder,
        fusion=fusion,
        pad_token_id=tokenizer.pad_id,
    )
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = training_config.block_size
    cfg.graph_encoder = graph_encoder
    if training_config.graph_relation_mode == "rgcn" and graph_encoder != "none":
        cfg.graph_encoder = "rgcn"
    cfg.fusion = fusion
    cfg.pad_token_id = tokenizer.pad_id
    cfg.graph_relation_mode = training_config.graph_relation_mode
    cfg.graph_pooling = training_config.graph_pooling
    cfg.graph_node_importance = training_config.graph_node_importance
    cfg.fusion_levels = training_config.fusion_levels
    cfg.graph_fusion_scale = training_config.graph_fusion_scale
    cfg.graph_fusion_dropout = training_config.graph_fusion_dropout
    cfg.graph_edge_types = (
        len(graph.graph_config.get("edge_types", {"cooccurrence": 0}))
        if graph is not None
        else 1
    )
    model = GraphCausalLM(cfg)
    graph_config = (
        {
            "mode": "baseline",
            "graph_encoder": "none",
            "num_nodes": 0,
            "num_edges": 0,
            "dynamic_graph": training_config.dynamic_graph,
        }
        if graph is None
        else {
            **graph.graph_config,
            "dynamic_graph": training_config.dynamic_graph,
            "graph_source": "train",
        }
    )
    trainer = LMTrainer(
        model,
        tokenizer,
        None if graph is None else graph.to_pyg_data(),
        None if graph is None else graph.token_node_ids(tokenizer.vocab_size),
        config=training_config,
        graph_config=graph_config,
    )
    metrics = trainer.train(dataset, validation_dataset)
    metrics["tokenizer_stats"] = _tokenizer_stats(
        tokenizer,
        train_corpus,
        validation_corpus,
    )
    metrics_path = Path(training_config.output_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

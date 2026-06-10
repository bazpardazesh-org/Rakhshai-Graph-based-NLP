"""Training loop for Graph causal language models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augmentation import (
    TextAugmentationConfig,
    augment_corpus,
    augment_graph_data,
    mean_pool_hidden,
)
from .dataset import LMDataset, LMLoaders, build_lm_dataloaders
from .graph_builder import build_graph_lm_graph, build_graph_lm_graph_from_token_ids
from .graph_memory import GraphMemoryArtifact, GraphMemoryConfig
from .model import GenerationConfig, GraphCausalLM, GraphLMConfig, perplexity
from .multitask import DEFAULT_TASK_LOSSES, MultiTaskLossConfig, compute_multitask_losses
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
    text_augmentation: bool = True
    augmentation_ratio: float = 0.5
    token_dropout: float = 0.05
    punctuation_dropout: float = 0.5
    node_dropout: float = 0.05
    edge_dropout: float = 0.1
    subgraph_sampling_ratio: float = 0.9
    contrastive_weight: float = 0.05
    curriculum_learning: bool = True
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    max_grad_norm: float = 1.0
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
        graph_memory: GraphMemoryArtifact | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.graph_data = graph_data
        self.token_node_ids = token_node_ids
        self.config = config
        self.graph_config = graph_config
        self.graph_memory = graph_memory
        self.device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        if self.graph_data is not None:
            self.graph_data = self.graph_data.to(self.device)
        if self.token_node_ids is not None:
            self.token_node_ids = self.token_node_ids.to(self.device)
        self._last_fusion_stats: dict[str, float] = {}
        self._last_loss_stats: dict[str, float] = {}
        self._last_task_status: dict[str, str] = {}

    def _multitask_config(self) -> MultiTaskLossConfig:
        return MultiTaskLossConfig(
            task_losses=self.config.task_losses,
            next_token_weight=self.config.next_token_weight,
            masked_token_weight=self.config.masked_token_weight,
            edge_prediction_weight=self.config.edge_prediction_weight,
            neighbor_prediction_weight=self.config.neighbor_prediction_weight,
            node_relation_weight=self.config.node_relation_weight,
            graph_text_alignment_weight=self.config.graph_text_alignment_weight,
            sentence_graph_alignment_weight=self.config.sentence_graph_alignment_weight,
            mask_probability=self.config.mask_probability,
            negative_samples=self.config.negative_samples,
        )

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
        if self.config.curriculum_learning and hasattr(dataset, "examples"):
            dataset.examples.sort(
                key=lambda pair: int(pair[1].ne(-100).sum().item())
            )
        if validation_dataset is None:
            loaders = build_lm_dataloaders(
                dataset,
                batch_size=self.config.batch_size,
                validation_ratio=self.config.validation_ratio,
                seed=self.config.seed,
                shuffle_train=not self.config.curriculum_learning,
            )
            return loaders
        generator = torch.Generator().manual_seed(self.config.seed)
        return LMLoaders(
            train=DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=not self.config.curriculum_learning,
                generator=generator,
            ),
            validation=DataLoader(validation_dataset, batch_size=self.config.batch_size),
        )

    def _contrastive_loss(
        self,
        input_ids: torch.Tensor,
        output: dict[str, torch.Tensor],
        graph_data,
        token_node_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            self.config.contrastive_weight <= 0
            or self.model.config.graph_encoder == "none"
            or graph_data is None
            or token_node_ids is None
            or output.get("hidden") is None
        ):
            return None
        contrast_graph = augment_graph_data(
            self.graph_data,
            edge_dropout=max(self.config.edge_dropout, 0.15),
            node_dropout=max(self.config.node_dropout, 0.1),
            subgraph_ratio=min(self.config.subgraph_sampling_ratio, 0.85),
            training=True,
        )
        if contrast_graph is None:
            return None
        with torch.no_grad():
            contrast_output = self.model(
                input_ids,
                graph_data=contrast_graph,
                token_node_ids=token_node_ids,
                return_hidden=True,
            )
        anchor = mean_pool_hidden(
            output["hidden"],
            input_ids,
            self.model.config.pad_token_id,
        )
        positive = mean_pool_hidden(
            contrast_output["hidden"],
            input_ids,
            self.model.config.pad_token_id,
        )
        return 1.0 - F.cosine_similarity(anchor, positive.detach(), dim=-1).mean()

    def _run_epoch(self, loader, optimizer: torch.optim.Optimizer | None = None) -> float:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        total_batches = 0
        fusion_sums: dict[str, float] = {}
        fusion_count = 0
        loss_sums: dict[str, float] = {}
        task_status: dict[str, str] = {}
        multitask_config = self._multitask_config()
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
            elif training:
                graph_data = augment_graph_data(
                    graph_data,
                    edge_dropout=self.config.edge_dropout,
                    node_dropout=self.config.node_dropout,
                    subgraph_ratio=self.config.subgraph_sampling_ratio,
                    training=True,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = self.model(
                input_ids,
                graph_data=graph_data,
                token_node_ids=token_node_ids,
                graph_embeddings=graph_embeddings,
                labels=labels,
                return_hidden=True,
            )
            loss, task_losses, batch_status = compute_multitask_losses(
                self.model,
                input_ids,
                labels,
                output,
                graph_data=graph_data,
                token_node_ids=token_node_ids,
                config=multitask_config,
            )
            if training:
                contrastive_loss = self._contrastive_loss(
                    input_ids,
                    output,
                    graph_data,
                    token_node_ids,
                )
                if contrastive_loss is not None:
                    task_losses["contrastive"] = contrastive_loss
                    batch_status["contrastive"] = "active"
                    loss = loss + contrastive_loss * float(self.config.contrastive_weight)
                elif self.config.contrastive_weight > 0:
                    batch_status["contrastive"] = "skipped"
            for key, value in task_losses.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu())
            for key, value in batch_status.items():
                previous = task_status.get(key)
                if previous == "active" or value == "active":
                    task_status[key] = "active"
                elif previous == "skipped" or value == "skipped":
                    task_status[key] = "skipped"
                else:
                    task_status[key] = value
            fusion_stats = output.get("fusion_stats", {})
            if fusion_stats:
                fusion_count += 1
                for key, value in fusion_stats.items():
                    fusion_sums[key] = fusion_sums.get(key, 0.0) + float(
                        value.detach().cpu()
                    )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        self._last_fusion_stats = {
            key: value / max(1, fusion_count) for key, value in fusion_sums.items()
        }
        self._last_loss_stats = {
            key: value / max(1, total_batches) for key, value in loss_sums.items()
        }
        self._last_task_status = task_status
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
        best_epoch = 0
        epochs_without_improvement = 0
        stopped_early = False
        early_stopping_reason = ""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(loaders.train, optimizer)
            train_fusion_stats = dict(self._last_fusion_stats)
            train_task_losses = dict(self._last_loss_stats)
            task_status = dict(self._last_task_status)
            if loaders.validation is not None:
                with torch.no_grad():
                    val_loss = self._run_epoch(loaders.validation)
                validation_fusion_stats = dict(self._last_fusion_stats)
                validation_task_losses = dict(self._last_loss_stats)
                task_status.update(self._last_task_status)
            else:
                val_loss = train_loss
                validation_fusion_stats = train_fusion_stats
                validation_task_losses = train_task_losses
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "generalization_gap": val_loss - train_loss,
                "perplexity": perplexity(val_loss),
            }
            if train_fusion_stats:
                row["train_fusion"] = train_fusion_stats
            if train_task_losses:
                row["train_task_losses"] = train_task_losses
            if task_status:
                row["task_status"] = task_status
            if validation_fusion_stats:
                row["validation_fusion"] = validation_fusion_stats
            if validation_task_losses:
                row["validation_task_losses"] = validation_task_losses
            history.append(row)
            improved = val_loss < (best_val - self.config.early_stopping_min_delta)
            if improved or val_loss <= best_val:
                best_val = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self.model.save_pretrained(
                    output_dir,
                    tokenizer=self.tokenizer,
                    graph_config=self.graph_config,
                    graph_data=self.graph_data,
                    token_node_ids=self.token_node_ids,
                    graph_memory=self.graph_memory,
                    graph_memory_config=GraphMemoryConfig(enabled=True),
                    generation_config=GenerationConfig(eos_token_id=self.tokenizer.eos_id),
                )
            else:
                epochs_without_improvement += 1
            if (
                self.config.early_stopping_patience > 0
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                stopped_early = True
                early_stopping_reason = (
                    "validation_loss did not improve for "
                    f"{self.config.early_stopping_patience} epoch(s)"
                )
                break

        metrics = {
            "training_config": asdict(self.config),
            "model_config": asdict(self.model.config),
            "history": history,
            "best_validation_loss": best_val,
            "best_perplexity": perplexity(best_val),
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "stopped_early": stopped_early,
            "early_stopping_reason": early_stopping_reason,
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
    augmented_train_corpus = augment_corpus(
        train_corpus,
        TextAugmentationConfig(
            enabled=training_config.text_augmentation,
            ratio=training_config.augmentation_ratio,
            token_dropout=training_config.token_dropout,
            punctuation_dropout=training_config.punctuation_dropout,
        ),
        seed=training_config.seed,
    )
    dataset = LMDataset(
        augmented_train_corpus,
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
            augmented_train_corpus,
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
            "low_data_training": {
                "text_augmentation": training_config.text_augmentation,
                "augmentation_ratio": training_config.augmentation_ratio,
                "augmented_train_examples": len(augmented_train_corpus),
                "original_train_examples": len(train_corpus),
                "node_dropout": training_config.node_dropout,
                "edge_dropout": training_config.edge_dropout,
                "subgraph_sampling_ratio": training_config.subgraph_sampling_ratio,
                "contrastive_weight": training_config.contrastive_weight,
                "curriculum_learning": training_config.curriculum_learning,
                "early_stopping_patience": training_config.early_stopping_patience,
            },
        }
    )
    trainer = LMTrainer(
        model,
        tokenizer,
        None if graph is None else graph.to_pyg_data(),
        None if graph is None else graph.token_node_ids(tokenizer.vocab_size),
        config=training_config,
        graph_config=graph_config,
        graph_memory=None if graph is None else GraphMemoryArtifact.from_graph(graph),
    )
    metrics = trainer.train(dataset, validation_dataset)
    metrics["tokenizer_stats"] = _tokenizer_stats(
        tokenizer,
        train_corpus,
        validation_corpus,
    )
    metrics["low_data_training"] = {
        "text_augmentation": training_config.text_augmentation,
        "augmentation_ratio": training_config.augmentation_ratio,
        "augmented_train_examples": len(augmented_train_corpus),
        "original_train_examples": len(train_corpus),
        "node_dropout": training_config.node_dropout,
        "edge_dropout": training_config.edge_dropout,
        "subgraph_sampling_ratio": training_config.subgraph_sampling_ratio,
        "contrastive_weight": training_config.contrastive_weight,
        "curriculum_learning": training_config.curriculum_learning,
        "early_stopping_patience": training_config.early_stopping_patience,
    }
    metrics_path = Path(training_config.output_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

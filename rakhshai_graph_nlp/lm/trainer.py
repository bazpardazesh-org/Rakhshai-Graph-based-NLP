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
    dynamic_graph: bool = False
    tokenizer_type: str = "word"
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
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
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
            if loaders.validation is not None:
                with torch.no_grad():
                    val_loss = self._run_epoch(loaders.validation)
            else:
                val_loss = train_loss
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "perplexity": perplexity(val_loss),
            }
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
    cfg.fusion = fusion
    cfg.pad_token_id = tokenizer.pad_id
    cfg.graph_edge_types = 2 if training_config.context_node_type != "none" else 1
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
    return trainer.train(dataset, validation_dataset)

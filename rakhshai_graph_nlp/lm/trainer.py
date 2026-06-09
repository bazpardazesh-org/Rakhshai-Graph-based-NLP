"""Training loop for Graph causal language models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from .dataset import LMDataset, build_lm_dataloaders
from .graph_builder import build_graph_lm_graph
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

    def _run_epoch(self, loader, optimizer: torch.optim.Optimizer | None = None) -> float:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        total_batches = 0
        for input_ids, labels in loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = self.model(
                input_ids,
                graph_data=self.graph_data,
                token_node_ids=self.token_node_ids,
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

    def train(self, dataset: LMDataset) -> dict[str, object]:
        loaders = build_lm_dataloaders(
            dataset,
            batch_size=self.config.batch_size,
            validation_ratio=self.config.validation_ratio,
            seed=self.config.seed,
        )
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
    tokenizer = PersianTokenizer(
        min_freq=training_config.min_freq,
        max_vocab_size=training_config.max_vocab_size,
    ).fit(corpus)
    dataset = LMDataset(
        corpus,
        tokenizer,
        block_size=training_config.block_size,
        stride=training_config.stride,
    )
    graph = None
    if graph_encoder != "none":
        graph = build_graph_lm_graph(
            corpus,
            tokenizer,
            window_size=training_config.graph_window_size,
            min_count=training_config.graph_min_count,
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
    model = GraphCausalLM(cfg)
    graph_config = (
        {
            "mode": "baseline",
            "graph_encoder": "none",
            "num_nodes": 0,
            "num_edges": 0,
        }
        if graph is None
        else graph.graph_config
    )
    trainer = LMTrainer(
        model,
        tokenizer,
        None if graph is None else graph.to_pyg_data(),
        None if graph is None else graph.token_node_ids(tokenizer.vocab_size),
        config=training_config,
        graph_config=graph_config,
    )
    return trainer.train(dataset)

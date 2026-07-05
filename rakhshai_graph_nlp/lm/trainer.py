"""Training loop for Graph causal language models."""

from __future__ import annotations

import json
import hashlib
import math
import time
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
from .distributed import get_distributed_info, maybe_wrap_distributed
from .graph_builder import (
    GraphLMGraph,
    build_graph_lm_graph,
    build_graph_lm_graph_from_token_ids,
)
from .graph_memory import GraphMemoryArtifact, GraphMemoryConfig
from .model import GenerationConfig, GraphCausalLM, GraphLMConfig, perplexity
from .multitask import DEFAULT_TASK_LOSSES, MultiTaskLossConfig, compute_multitask_losses
from .shards import TokenShardDataset
from .tokenizer import PersianTokenizer


@dataclass
class LMTrainingConfig:
    output_dir: str = "runs/graph-lm"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 3e-4
    # "cosine"/"linear" decay the LR per optimizer step after a linear warmup
    # over warmup_ratio of the total steps; "none" keeps a constant LR.
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int | None = None
    min_lr_ratio: float = 0.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
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
    semantic_method: str = "distributional"
    linguistic_backend: str = "auto"
    topic_top_k: int = 8
    # "embedding" learns a vector per relation and best exploits the multi-
    # relational (parallel-edge) graph; "bias"/"rgcn" remain available.
    graph_relation_mode: str = "embedding"
    graph_pooling: str = "none"
    graph_node_importance: bool = False
    graph_node_type_embedding: bool = True
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
    # Which validation signal drives best-checkpoint selection and early stopping.
    # "next_token" tracks the pure language-modelling loss (perplexity) and is the
    # right default for text generation; "total" tracks the full multi-task loss.
    checkpoint_metric: str = "next_token"
    max_grad_norm: float = 1.0
    dynamic_graph: bool = False
    graph_build_batch_size: int | None = None
    graph_cache_dir: str | None = None
    reuse_graph_cache: bool = True
    token_shard_manifest: str | None = None
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    amp: bool = False
    precision: str = "auto"
    gradient_accumulation_steps: int = 1
    compile_model: bool = False
    sharded_checkpoint: bool = False
    distributed_backend: str = "none"
    resume_from: str | None = None
    tokenizer_type: str = "unigram"
    tokenizer_half_space: str = "preserve"
    tokenizer_morph_splitting: bool = False
    tokenizer_compound_verb_mode: str = "none"
    tokenizer_bpe_merges: int = 200
    tokenizer_unigram_num_pieces: int = 8000
    tokenizer_byte_fallback: bool = False
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
        self.raw_model = model
        self.model.to(self.device)
        self.distributed_info = get_distributed_info(config.distributed_backend)
        if config.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        self.model = maybe_wrap_distributed(self.model, config.distributed_backend)
        if self.graph_data is not None:
            self.graph_data = self.graph_data.to(self.device)
        if self.token_node_ids is not None:
            self.token_node_ids = self.token_node_ids.to(self.device)
        self._last_fusion_stats: dict[str, float] = {}
        self._last_loss_stats: dict[str, float] = {}
        self._last_task_status: dict[str, str] = {}
        self._last_token_count = 0
        self._last_optimizer_steps = 0

    def _model_for_io(self) -> GraphCausalLM:
        model = self.model
        if hasattr(model, "module"):
            model = model.module  # type: ignore[assignment]
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod  # type: ignore[assignment]
        return model  # type: ignore[return-value]

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
        *,
        shuffle_train: bool | None = None,
    ) -> LMLoaders:
        # Curriculum ordering only applies to the first epoch; the training
        # loop rebuilds a shuffled loader for later epochs so batches vary.
        if shuffle_train is None:
            shuffle_train = not self.config.curriculum_learning
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
                shuffle_train=shuffle_train,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory and self.device.type == "cuda",
            )
            return loaders
        effective_pin_memory = self.config.dataloader_pin_memory and self.device.type == "cuda"
        generator = torch.Generator().manual_seed(self.config.seed)
        loader_kwargs = {
            "num_workers": self.config.dataloader_num_workers,
            "pin_memory": effective_pin_memory,
            "persistent_workers": self.config.dataloader_num_workers > 0,
        }
        return LMLoaders(
            train=DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle_train,
                generator=generator,
                **loader_kwargs,
            ),
            validation=DataLoader(
                validation_dataset,
                batch_size=self.config.batch_size,
                **loader_kwargs,
            ),
        )

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
    ) -> torch.optim.lr_scheduler.LambdaLR | None:
        name = self.config.lr_scheduler.lower()
        if name not in {"none", "cosine", "linear"}:
            raise ValueError("lr_scheduler must be one of: none, cosine, linear")
        if name == "none":
            return None
        warmup_steps = (
            int(self.config.warmup_steps)
            if self.config.warmup_steps is not None
            else int(total_steps * max(0.0, self.config.warmup_ratio))
        )
        warmup_steps = min(total_steps, max(0, warmup_steps))
        min_lr_ratio = min(1.0, max(0.0, float(self.config.min_lr_ratio)))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(1.0, progress)
            if name == "linear":
                decay = 1.0 - progress
            else:
                decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _contrastive_loss(
        self,
        input_ids: torch.Tensor,
        output: dict[str, torch.Tensor],
        graph_data,
        token_node_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if (
            self.config.contrastive_weight <= 0
            or self._model_for_io().config.graph_encoder == "none"
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
            self._model_for_io().config.pad_token_id,
        )
        positive = mean_pool_hidden(
            contrast_output["hidden"],
            input_ids,
            self._model_for_io().config.pad_token_id,
        )
        return 1.0 - F.cosine_similarity(anchor, positive.detach(), dim=-1).mean()

    def _autocast_dtype(self) -> torch.dtype | None:
        precision = self.config.precision.lower()
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: auto, fp32, fp16, bf16")
        if self.device.type != "cuda":
            return None
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16" or (precision == "auto" and self.config.amp):
            return torch.float16
        return None

    def _amp_enabled(self) -> bool:
        return self._autocast_dtype() is not None

    def _scaler_enabled(self) -> bool:
        return self._autocast_dtype() == torch.float16

    def _save_training_state(
        self,
        output_dir: Path,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        *,
        epoch: int,
        best_val: float,
        best_epoch: int,
        epochs_without_improvement: int,
        history: list[dict[str, object]],
    ) -> None:
        state: dict[str, object] = {
            "version": 1,
            "epoch": epoch,
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                None if scheduler is None else scheduler.state_dict()
            ),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(state, output_dir / "training_state.pt")

    def _load_resume_state(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> tuple[int, float, int, int, list[dict[str, object]]]:
        if not self.config.resume_from:
            return 1, float("inf"), 0, 0, []

        resume_dir = Path(self.config.resume_from)
        model_path = resume_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self._model_for_io().load_state_dict(state_dict, strict=False)

        state_path = resume_dir / "training_state.pt"
        if not state_path.exists():
            return 1, float("inf"), 0, 0, []

        state = torch.load(state_path, map_location=self.device)
        optimizer_state = state.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = state.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        torch_rng_state = state.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.detach().cpu())
        cuda_rng_state = state.get("cuda_rng_state_all")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)
        start_epoch = int(state.get("epoch", 0)) + 1
        best_val = float(state.get("best_validation_loss", float("inf")))
        best_epoch = int(state.get("best_epoch", 0))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        history = list(state.get("history", []))
        return start_epoch, best_val, best_epoch, epochs_without_improvement, history

    def _run_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> float:
        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        total_batches = 0
        fusion_sums: dict[str, float] = {}
        fusion_count = 0
        loss_sums: dict[str, float] = {}
        task_status: dict[str, str] = {}
        token_count = 0
        optimizer_steps = 0
        multitask_config = self._multitask_config()
        total_loader_batches = len(loader)
        accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
        if training:
            optimizer.zero_grad(set_to_none=True)
        for batch_index, (input_ids, labels) in enumerate(loader, start=1):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            token_count += int(labels.ne(-100).sum().detach().cpu())
            graph_data = self.graph_data
            token_node_ids = self.token_node_ids
            graph_embeddings = None
            if (
                self.config.dynamic_graph
                and self._model_for_io().config.graph_encoder != "none"
            ):
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
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self._amp_enabled(),
                dtype=self._autocast_dtype(),
            ):
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
                scaled_loss = loss / accumulation_steps
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                should_step = (
                    batch_index % accumulation_steps == 0
                    or batch_index == total_loader_batches
                )
                if should_step:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if scheduler is not None:
                        scheduler.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        self._last_fusion_stats = {
            key: value / max(1, fusion_count) for key, value in fusion_sums.items()
        }
        self._last_loss_stats = {
            key: value / max(1, total_batches) for key, value in loss_sums.items()
        }
        self._last_task_status = task_status
        self._last_token_count = token_count
        self._last_optimizer_steps = optimizer_steps
        return total_loss / max(1, total_batches)

    def train(
        self,
        dataset: LMDataset,
        validation_dataset: LMDataset | None = None,
    ) -> dict[str, object]:
        loaders = self._build_loaders(dataset, validation_dataset)
        validation_available = loaders.validation is not None
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )
        accumulation_steps = max(1, int(self.config.gradient_accumulation_steps))
        total_steps = (
            max(1, math.ceil(len(loaders.train) / accumulation_steps))
            * self.config.epochs
        )
        scheduler = self._build_scheduler(optimizer, total_steps)
        (
            start_epoch,
            best_val,
            best_epoch,
            epochs_without_improvement,
            history,
        ) = self._load_resume_state(optimizer, scheduler)
        scaler = torch.amp.GradScaler("cuda", enabled=self._scaler_enabled())
        stopped_early = False
        early_stopping_reason = ""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        shuffled_train_loader = None
        for epoch in range(start_epoch, self.config.epochs + 1):
            train_loader = loaders.train
            if self.config.curriculum_learning and epoch > 1:
                # Curriculum ordering only seeds the first epoch; afterwards
                # standard shuffling keeps batch composition varying.
                if shuffled_train_loader is None:
                    shuffled_train_loader = self._build_loaders(
                        dataset,
                        validation_dataset,
                        shuffle_train=True,
                    ).train
                train_loader = shuffled_train_loader
            train_loss = self._run_epoch(train_loader, optimizer, scaler, scheduler)
            train_fusion_stats = dict(self._last_fusion_stats)
            train_task_losses = dict(self._last_loss_stats)
            train_token_count = int(self._last_token_count)
            train_optimizer_steps = int(self._last_optimizer_steps)
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
            # Perplexity must come from the next-token cross-entropy alone;
            # the total validation loss also contains the other weighted
            # multi-task terms and is not a language-modelling loss.
            val_next_token = float(
                validation_task_losses.get("next_token", val_loss)
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_next_token_loss": val_next_token,
                "validation_available": validation_available,
                "generalization_gap": val_loss - train_loss,
                "perplexity": perplexity(val_next_token),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_tokens": train_token_count,
                "optimizer_steps": train_optimizer_steps,
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
            # Best-checkpoint / early-stopping signal. Default tracks the pure
            # next-token loss so the selected model is the best language model,
            # not the lowest weighted sum of auxiliary multi-task terms.
            monitor = (
                val_loss
                if self.config.checkpoint_metric == "total"
                else val_next_token
            )
            improved_beyond_delta = monitor < (
                best_val - self.config.early_stopping_min_delta
            )
            if monitor < best_val:
                best_val = monitor
                best_epoch = epoch
                self._model_for_io().save_pretrained(
                    output_dir,
                    tokenizer=self.tokenizer,
                    graph_config=self.graph_config,
                    graph_data=self.graph_data,
                    token_node_ids=self.token_node_ids,
                    graph_memory=self.graph_memory,
                    graph_memory_config=GraphMemoryConfig(enabled=True),
                    generation_config=GenerationConfig(eos_token_id=self.tokenizer.eos_id),
                )
                checkpoint_manifest = {
                    "format": "single_file_with_manifest",
                    "sharded_checkpoint": self.config.sharded_checkpoint,
                    "model_files": ["model.pt"],
                    "optimizer_state_file": "training_state.pt",
                    "tokenizer_file": "tokenizer.json",
                    "graph_files": [
                        name
                        for name in ["graph.pt", "graph_config.json", "graph_memory.pt"]
                        if (output_dir / name).exists()
                    ],
                    "distributed": self.distributed_info.to_dict(),
                }
                (output_dir / "checkpoint_manifest.json").write_text(
                    json.dumps(checkpoint_manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            # min_delta gates patience only: marginal gains still refresh the
            # best checkpoint but do not postpone early stopping.
            if improved_beyond_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            self._save_training_state(
                output_dir,
                optimizer,
                scheduler,
                epoch=epoch,
                best_val=best_val,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )
            if (
                self.config.early_stopping_patience > 0
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                stopped_early = True
                monitored = (
                    "validation_loss"
                    if self.config.checkpoint_metric == "total"
                    else "validation_next_token_loss"
                )
                if not validation_available:
                    monitored = monitored.replace("validation", "train")
                early_stopping_reason = (
                    f"{monitored} did not improve for "
                    f"{self.config.early_stopping_patience} epoch(s)"
                )
                break

        best_row = next(
            (row for row in history if row.get("epoch") == best_epoch),
            None,
        )
        best_next_token = (
            float(best_row.get("validation_next_token_loss", best_row["validation_loss"]))
            if best_row is not None
            else best_val
        )
        metrics = {
            "training_config": asdict(self.config),
            "model_config": asdict(self._model_for_io().config),
            "history": history,
            # Without a validation set the "validation" figures below fall
            # back to training-loss values.
            "validation_available": validation_available,
            "best_validation_loss": best_val,
            "best_next_token_loss": best_next_token,
            "best_perplexity": perplexity(best_next_token),
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "resumed_from": self.config.resume_from,
            "stopped_early": stopped_early,
            "early_stopping_reason": early_stopping_reason,
            "checkpoint_dir": str(output_dir),
            "trainer_scaling": {
                "gradient_accumulation_steps": accumulation_steps,
                "effective_tokens_per_optimizer_step": (
                    self.config.batch_size * self.config.block_size * accumulation_steps
                ),
                "precision": self.config.precision,
                "amp_legacy_flag": self.config.amp,
                "compile_model": self.config.compile_model,
                "sharded_checkpoint": self.config.sharded_checkpoint,
                "distributed": self.distributed_info.to_dict(),
                "total_optimizer_steps_planned": total_steps,
            },
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
        "byte_fallback": tokenizer.byte_fallback,
        "bpe_merges": len(tokenizer.bpe_merges),
        "train_token_count": len(train_tokens),
        "validation_token_count": len(validation_tokens),
        "validation_unk_count": unk_count,
        "validation_unk_rate": unk_count / max(1, len(validation_tokens)),
        "avg_validation_tokens_per_text": len(validation_tokens) / validation_text_count,
    }


def _graph_cache_key(
    corpus: Sequence[str],
    tokenizer: PersianTokenizer,
    graph_params: dict[str, object],
) -> str:
    payload = {
        "corpus": list(corpus),
        "tokenizer": {
            "tokenizer_type": tokenizer.tokenizer_type,
            "vocab": tokenizer.token_to_id,
            "keep_half_space": tokenizer.keep_half_space,
            "morph_splitting": tokenizer.morph_splitting,
            "compound_verb_mode": tokenizer.compound_verb_mode,
            "bpe_merges": tokenizer.bpe_merges,
        },
        "graph": graph_params,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_or_build_graph(
    corpus: Sequence[str],
    tokenizer: PersianTokenizer,
    training_config: LMTrainingConfig,
) -> tuple[GraphLMGraph, dict[str, object]]:
    graph_params: dict[str, object] = {
        "window_size": training_config.graph_window_size,
        "min_count": training_config.graph_min_count,
        "weighting": training_config.graph_weighting,
        "min_edge_weight": training_config.graph_min_edge_weight,
        "top_k": training_config.graph_top_k,
        "directed": training_config.graph_directed,
        "graph_scope": training_config.graph_scope,
        "context_node_type": training_config.context_node_type,
        "graph_relations": list(training_config.graph_relations or []),
        "relation_weights": training_config.relation_weights or {},
        "semantic_similarity_threshold": training_config.semantic_similarity_threshold,
        "semantic_top_k": training_config.semantic_top_k,
        "semantic_method": training_config.semantic_method,
        "linguistic_backend": training_config.linguistic_backend,
        "topic_top_k": training_config.topic_top_k,
        "build_batch_size": training_config.graph_build_batch_size,
        "token_shard_manifest": training_config.token_shard_manifest,
    }
    cache_key = _graph_cache_key(corpus, tokenizer, graph_params)
    cache_path = None
    cache_hit = False
    started = time.perf_counter()

    if training_config.graph_cache_dir:
        cache_dir = Path(training_config.graph_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.graph.pt"
        if training_config.reuse_graph_cache and cache_path.exists():
            graph = GraphLMGraph.load_artifact(cache_path)
            cache_hit = True
        else:
            graph = build_graph_lm_graph(
                corpus,
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
                semantic_method=training_config.semantic_method,
                linguistic_backend=training_config.linguistic_backend,
                topic_top_k=training_config.topic_top_k,
                build_batch_size=training_config.graph_build_batch_size,
            )
            graph.save_artifact(cache_path)
    else:
        graph = build_graph_lm_graph(
            corpus,
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
            build_batch_size=training_config.graph_build_batch_size,
        )

    elapsed = time.perf_counter() - started
    graph.graph_config["cache_key"] = cache_key
    graph.graph_config["cache_hit"] = cache_hit
    if cache_path is not None:
        graph.graph_config["cache_path"] = str(cache_path)
    graph.graph_config["build_seconds"] = elapsed
    report = {
        "cache_enabled": training_config.graph_cache_dir is not None,
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "cache_path": None if cache_path is None else str(cache_path),
        "build_seconds": elapsed,
        "build_batch_size": training_config.graph_build_batch_size,
        "graph_build_batches": graph.graph_config.get("graph_build_batches", 1),
        "num_nodes": graph.graph_config.get("num_nodes", len(graph.nodes)),
        "num_edges": graph.graph_config.get("num_edges", int(graph.edge_index.shape[1])),
    }
    if cache_path is not None:
        manifest_path = cache_path.with_suffix(".graph.json")
        manifest_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return graph, report


def train_graph_lm(
    texts: Iterable[str],
    *,
    training_config: LMTrainingConfig,
    model_config: GraphLMConfig | None = None,
    graph_encoder: str = "gat",
    fusion: str = "gated",
    validation_texts: Iterable[str] | None = None,
) -> dict[str, object]:
    corpus = [text.strip() for text in texts if text.strip()]
    if not corpus:
        raise ValueError("corpus is empty")
    torch.manual_seed(training_config.seed)
    if validation_texts is None:
        train_corpus, validation_corpus = _split_corpus(
            corpus,
            validation_ratio=training_config.validation_ratio,
            seed=training_config.seed,
        )
        validation_source = "validation_ratio"
    else:
        train_corpus = corpus
        validation_corpus = [text.strip() for text in validation_texts if text.strip()]
        validation_source = "explicit"
    tokenizer = PersianTokenizer(
        min_freq=training_config.min_freq,
        max_vocab_size=training_config.max_vocab_size,
        tokenizer_type=training_config.tokenizer_type,
        keep_half_space=training_config.tokenizer_half_space == "preserve",
        morph_splitting=training_config.tokenizer_morph_splitting,
        compound_verb_mode=training_config.tokenizer_compound_verb_mode,
        bpe_num_merges=training_config.tokenizer_bpe_merges,
        unigram_num_pieces=training_config.tokenizer_unigram_num_pieces,
        byte_fallback=training_config.tokenizer_byte_fallback,
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
    graph_scalability: dict[str, object] = {
        "cache_enabled": False,
        "cache_hit": False,
        "build_seconds": 0.0,
        "build_batch_size": training_config.graph_build_batch_size,
        "graph_build_batches": 0,
        "num_nodes": 0,
        "num_edges": 0,
    }
    if graph_encoder != "none":
        # Graph statistics (co-occurrence, PMI, ...) come from the clean
        # training corpus; augmentation only feeds the text stream, otherwise
        # noisy duplicates would distort the edge weights.
        graph, graph_scalability = _load_or_build_graph(
            train_corpus,
            tokenizer,
            training_config,
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
    cfg.mask_token_id = tokenizer.mask_id
    cfg.graph_relation_mode = training_config.graph_relation_mode
    cfg.graph_pooling = training_config.graph_pooling
    cfg.graph_node_importance = training_config.graph_node_importance
    cfg.graph_node_type_embedding = training_config.graph_node_type_embedding
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
            "scalability": graph_scalability,
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
    metrics["corpus_split"] = {
        "train_examples": len(train_corpus),
        "validation_examples": len(validation_corpus),
        "validation_source": validation_source,
    }
    metrics["graph_scalability"] = graph_scalability
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


def train_graph_lm_from_token_shards(
    manifest_path: str,
    *,
    training_config: LMTrainingConfig,
    model_config: GraphLMConfig | None = None,
    fusion: str = "gated",
) -> dict[str, object]:
    """Train a text-only native LM from memory-mapped token shards."""

    if training_config.graph_relations or training_config.dynamic_graph:
        raise ValueError("token-shard pretraining currently supports text-only training")
    training_config.token_shard_manifest = manifest_path
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    tokenizer = PersianTokenizer.load(manifest_file.parent / str(manifest["tokenizer"]))
    dataset = TokenShardDataset(
        manifest_file,
        split="train",
        block_size=training_config.block_size,
        stride=training_config.stride,
        pad_token_id=tokenizer.pad_id,
    )
    try:
        validation_dataset = TokenShardDataset(
            manifest_file,
            split="validation",
            block_size=training_config.block_size,
            stride=training_config.stride,
            pad_token_id=tokenizer.pad_id,
        )
    except ValueError:
        validation_dataset = None
    cfg = model_config or GraphLMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=training_config.block_size,
        graph_encoder="none",
        fusion=fusion,
        pad_token_id=tokenizer.pad_id,
    )
    cfg.vocab_size = tokenizer.vocab_size
    cfg.max_seq_len = training_config.block_size
    cfg.graph_encoder = "none"
    cfg.fusion = fusion
    cfg.pad_token_id = tokenizer.pad_id
    cfg.mask_token_id = tokenizer.mask_id
    model = GraphCausalLM(cfg)
    graph_config = {
        "mode": "baseline",
        "graph_encoder": "none",
        "num_nodes": 0,
        "num_edges": 0,
        "token_shard_manifest": manifest_path,
    }
    trainer = LMTrainer(
        model,
        tokenizer,
        None,
        None,
        config=training_config,
        graph_config=graph_config,
        graph_memory=None,
    )
    metrics = trainer.train(dataset, validation_dataset)
    metrics["token_shards"] = {
        "manifest": manifest_path,
        "train_windows": len(dataset),
        "validation_windows": 0 if validation_dataset is None else len(validation_dataset),
    }
    metrics_path = Path(training_config.output_dir) / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

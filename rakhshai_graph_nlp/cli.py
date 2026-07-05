"""Command line interface for Rakhshai Graph NLP."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .features.pyg_data import graph_to_data
from .features.tokenizer import tokenize
from .graphs.graph import Graph
from .graphs.text_graph import build_text_graph
from .llm.article.cli import (
    ARTICLE_COMMANDS,
    add_article_subcommands,
    run_article_command,
)
from .lm.corpus import CorpusBuildConfig, build_lm_corpus
from .lm.eval import NativeEvalConfig, evaluate_lm_checkpoint
from .lm.graph_builder import build_graph_lm_graph
from .lm.graph_memory import GraphMemoryArtifact, GraphMemoryConfig
from .lm.graph_scaling import LMGraphAblationConfig, run_lm_graph_ablation
from .lm.model import GraphCausalLM, GraphLMConfig
from .lm.profiles import available_model_profiles, build_graph_lm_config_from_profile
from .lm.registry import RunRegistryConfig, build_run_report, write_run_registry
from .lm.sft import SFTConfig, train_sft
from .lm.shards import TokenShardConfig, write_token_shards
from .lm.trainer import LMTrainingConfig, train_graph_lm, train_graph_lm_from_token_shards
from .metrics import accuracy, macro_f1
from .tasks.classification import train_node_classifier
from .utils.logging import setup_logger
from .utils.random import set_seed


def _synthetic_graph() -> tuple[Graph, np.ndarray, np.ndarray]:
    nodes = [0, 1, 2]
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    X = np.eye(3, dtype=float)
    labels = np.array([0, 1, 0])
    g = Graph(nodes=nodes, adjacency=A)
    return g, X, labels


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _load_text_dataset(
    path: str,
    *,
    text_column: str = "text",
    label_column: str = "label",
    dataset_format: str = "auto",
) -> tuple[list[str], list[str]]:
    """Load a labelled text classification dataset from CSV, TSV or JSONL."""

    dataset_path = Path(path)
    fmt = dataset_format
    if fmt == "auto":
        suffix = dataset_path.suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".tsv":
            fmt = "tsv"
        else:
            fmt = "csv"

    texts: list[str] = []
    labels: list[str] = []
    if fmt in {"csv", "tsv"}:
        delimiter = "\t" if fmt == "tsv" else ","
        with dataset_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError("dataset file must include a header row")
            missing = {text_column, label_column} - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"dataset is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                text = (row.get(text_column) or "").strip()
                label = (row.get(label_column) or "").strip()
                if text and label:
                    texts.append(text)
                    labels.append(label)
    elif fmt == "jsonl":
        with dataset_path.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    text = str(row[text_column]).strip()
                    label = str(row[label_column]).strip()
                except KeyError as exc:
                    raise ValueError(
                        f"dataset line {line_number} is missing field {exc.args[0]!r}"
                    ) from exc
                if text and label:
                    texts.append(text)
                    labels.append(label)
    else:
        raise ValueError("dataset_format must be one of: auto, csv, tsv, jsonl")

    if not texts:
        raise ValueError("dataset did not contain any labelled texts")
    return texts, labels


def _encode_labels(labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    return np.array([label_to_id[label] for label in labels], dtype=int), label_to_id


def _split_indices(
    n_items: int,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("at least one split ratio must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_items)
    if n_items == 1:
        return indices, np.array([], dtype=int), np.array([], dtype=int)

    total = train_ratio + val_ratio + test_ratio
    val_count = int(round(n_items * (val_ratio / total))) if val_ratio > 0 else 0
    test_count = int(round(n_items * (test_ratio / total))) if test_ratio > 0 else 0
    if val_ratio > 0 and n_items >= 3:
        val_count = max(1, val_count)
    if test_ratio > 0 and n_items >= 2:
        test_count = max(1, test_count)

    while val_count + test_count >= n_items:
        if val_count >= test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1
        else:
            break

    train_count = n_items - val_count - test_count
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return train_idx, val_idx, test_idx


def _make_node_mask(n_nodes: int, node_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_nodes, dtype=bool)
    mask[node_indices] = True
    return mask


def _evaluate_split(
    model: torch.nn.Module,
    data,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "accuracy": 0.0, "macro_f1": 0.0}
    preds = model.predict(data).cpu().numpy()
    y_true = labels[mask]
    y_pred = preds[mask]
    return {
        "count": count,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def _load_corpus(path: str) -> list[str]:
    corpus_path = Path(path)
    with corpus_path.open(encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    if not texts:
        raise ValueError("corpus file is empty")
    return texts


def _parse_relation_weights(raw: str | None) -> dict[str, float] | None:
    if not raw:
        return None
    weights: dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError("--relation-weights entries must use relation=value")
        relation, value = item.split("=", 1)
        weights[relation.strip()] = float(value)
    return weights


def _run_lm_build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    return build_lm_corpus(
        CorpusBuildConfig(
            input_paths=args.input,
            output_dir=args.output_dir,
            input_format=args.input_format,
            text_fields=args.text_fields,
            source_id=args.source_id,
            min_chars=args.min_chars,
            min_persian_ratio=args.min_persian_ratio,
            near_duplicate_threshold=args.near_duplicate_threshold,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            eval_paths=args.eval_paths or [],
        )
    )


def _run_lm_tokenize(args: argparse.Namespace) -> dict[str, Any]:
    return write_token_shards(
        TokenShardConfig(
            output_dir=args.output_dir,
            input_paths=args.input,
            corpus_dir=args.corpus_dir,
            tokenizer_path=args.tokenizer,
            tokenizer_type=args.tokenizer_type,
            tokenizer_half_space=args.tokenizer_half_space,
            tokenizer_morph_splitting=args.tokenizer_morph_splitting,
            tokenizer_compound_verb_mode=args.tokenizer_compound_verb_mode,
            tokenizer_bpe_merges=args.tokenizer_bpe_merges,
            tokenizer_unigram_num_pieces=args.unigram_num_pieces,
            tokenizer_max_vocab_size=args.max_vocab_size,
            byte_fallback=args.tokenizer_byte_fallback,
            block_size=args.block_size,
            stride=args.stride,
            tokens_per_shard=args.tokens_per_shard,
            seed=args.seed,
        )
    )


def _run_lm_ablation(args: argparse.Namespace) -> dict[str, Any]:
    training_config = LMTrainingConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        validation_ratio=args.validation_ratio,
        task_losses="next_token",
        tokenizer_type=args.tokenizer_type,
        tokenizer_unigram_num_pieces=args.unigram_num_pieces,
        graph_cache_dir=args.graph_cache_dir,
        device="cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu",
        seed=args.seed,
    )
    relation_groups = None
    if args.relation_groups:
        relation_groups = {}
        for item in args.relation_groups.split(";"):
            if not item.strip():
                continue
            name, raw_relations = item.split("=", 1)
            relation_groups[name.strip()] = [
                part.strip() for part in raw_relations.split(",") if part.strip()
            ]
    return run_lm_graph_ablation(
        _load_corpus(args.corpus),
        LMGraphAblationConfig(
            output_dir=args.output_dir,
            training_config=training_config,
            graph_encoders=args.graph_encoders,
            graph_scopes=args.graph_scopes,
            relation_groups=relation_groups,
        ),
    )


def _run_lm_eval(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_lm_checkpoint(
        NativeEvalConfig(
            model_dir=args.model,
            eval_path=args.eval_file,
            output_path=args.output_path,
            text_field=args.text_field,
            prompt_field=args.prompt_field,
            choices_field=args.choices_field,
            answer_field=args.answer_field,
            prediction_field=args.prediction_field,
            block_size=args.block_size,
            device=args.device,
            generation_prompts=args.generation_prompts or [],
            max_new_tokens=args.max_new_tokens,
        )
    )


def _run_lm_sft(args: argparse.Namespace) -> dict[str, Any]:
    return train_sft(
        SFTConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            prompt_field=args.prompt_field,
            completion_field=args.completion_field,
            messages_field=args.messages_field,
            min_completion_chars=args.min_completion_chars,
            epochs=args.epochs,
            batch_size=args.batch_size,
            block_size=args.block_size,
            validation_ratio=args.validation_ratio,
            tokenizer_type=args.tokenizer_type,
            tokenizer_unigram_num_pieces=args.unigram_num_pieces,
            device=args.device,
            seed=args.seed,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=args.block_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            graph_encoder="none",
        ),
    )


def _run_lm_pretrain(args: argparse.Namespace) -> dict[str, Any]:
    if not args.corpus and not args.shard_manifest:
        raise ValueError("lm-pretrain requires --corpus or --shard-manifest")
    if args.model_profile:
        model_config = build_graph_lm_config_from_profile(
            args.model_profile,
            vocab_size=1,
            overrides={
                "max_seq_len": args.context_size,
                "graph_encoder": "none",
                "attention_backend": args.attention_backend,
                "activation_checkpointing": args.activation_checkpointing,
            },
        )
        block_size = model_config.max_seq_len
    else:
        block_size = args.block_size
        model_config = GraphLMConfig(
            vocab_size=1,
            max_seq_len=block_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            graph_encoder="none",
            attention_backend=args.attention_backend,
            activation_checkpointing=args.activation_checkpointing,
        )
    training_config = LMTrainingConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=block_size,
        validation_ratio=args.validation_ratio,
        task_losses="next_token",
        tokenizer_type=args.tokenizer_type,
        tokenizer_unigram_num_pieces=args.unigram_num_pieces,
        precision=args.precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
        device="cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu",
        seed=args.seed,
    )
    if args.shard_manifest:
        metrics = train_graph_lm_from_token_shards(
            args.shard_manifest,
            training_config=training_config,
            model_config=model_config,
        )
        data_paths = [args.shard_manifest]
    else:
        metrics = train_graph_lm(
            _load_corpus(args.corpus),
            training_config=training_config,
            model_config=model_config,
            graph_encoder="none",
        )
        data_paths = [args.corpus]
    write_run_registry(
        RunRegistryConfig(
            run_dir=args.output_dir,
            command=["lm-pretrain"],
            data_paths=data_paths,
            checkpoint_dir=args.output_dir,
        )
    )
    return metrics


def _run_lm_run_report(args: argparse.Namespace) -> dict[str, Any]:
    return build_run_report(args.run_dir, args.output_path)


def _run_lm_train(args: argparse.Namespace) -> dict[str, Any]:
    if args.d_model % args.n_heads != 0:
        raise ValueError("--d-model must be divisible by --n-heads")
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if args.model_profile:
        model_config = build_graph_lm_config_from_profile(
            args.model_profile,
            vocab_size=1,
            overrides={
                "max_seq_len": args.context_size,
                "graph_encoder": args.graph_encoder,
                "fusion": args.fusion,
                "fusion_layers": args.fusion_layers,
                "fusion_levels": args.fusion_levels,
                "graph_relation_mode": args.graph_relation_mode,
                "graph_pooling": args.graph_pooling,
                "graph_node_importance": args.graph_node_importance,
                "graph_node_type_embedding": args.graph_node_type_embedding,
                "graph_fusion_scale": args.graph_fusion_scale,
                "graph_fusion_dropout": args.graph_fusion_dropout,
                "attention_backend": args.attention_backend,
                "activation_checkpointing": args.activation_checkpointing,
            },
        )
        effective_block_size = model_config.max_seq_len
    else:
        effective_block_size = args.block_size
        model_config = GraphLMConfig(
            vocab_size=1,
            max_seq_len=effective_block_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            attention_backend=args.attention_backend,
            activation_checkpointing=args.activation_checkpointing,
            graph_encoder=args.graph_encoder,
            graph_hidden_dim=args.graph_hidden_dim,
            graph_heads=args.graph_heads,
            graph_relation_mode=args.graph_relation_mode,
            graph_pooling=args.graph_pooling,
            graph_node_importance=args.graph_node_importance,
            graph_node_type_embedding=args.graph_node_type_embedding,
            fusion=args.fusion,
            fusion_layers=args.fusion_layers,
            fusion_levels=args.fusion_levels,
            graph_fusion_scale=args.graph_fusion_scale,
            graph_fusion_dropout=args.graph_fusion_dropout,
        )
    training_config = LMTrainingConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        validation_ratio=args.validation_ratio,
        block_size=effective_block_size,
        stride=args.stride,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        graph_window_size=args.graph_window_size,
        graph_min_count=args.graph_min_count,
        graph_weighting=args.graph_weighting,
        graph_min_edge_weight=args.graph_min_edge_weight,
        graph_top_k=args.graph_top_k,
        graph_directed=args.graph_directed,
        graph_scope=args.graph_scope,
        context_node_type=args.context_node_type,
        graph_relations=args.graph_relations,
        relation_weights=_parse_relation_weights(args.relation_weights),
        semantic_similarity_threshold=args.semantic_similarity_threshold,
        semantic_top_k=args.semantic_top_k,
        semantic_method=args.semantic_method,
        linguistic_backend=args.linguistic_backend,
        topic_top_k=args.topic_top_k,
        graph_relation_mode=args.graph_relation_mode,
        graph_pooling=args.graph_pooling,
        graph_node_importance=args.graph_node_importance,
        graph_node_type_embedding=args.graph_node_type_embedding,
        fusion_levels=args.fusion_levels,
        graph_fusion_scale=args.graph_fusion_scale,
        graph_fusion_dropout=args.graph_fusion_dropout,
        task_losses=args.task_losses,
        next_token_weight=args.next_token_weight,
        masked_token_weight=args.masked_token_weight,
        edge_prediction_weight=args.edge_prediction_weight,
        neighbor_prediction_weight=args.neighbor_prediction_weight,
        node_relation_weight=args.node_relation_weight,
        graph_text_alignment_weight=args.graph_text_alignment_weight,
        sentence_graph_alignment_weight=args.sentence_graph_alignment_weight,
        mask_probability=args.mask_probability,
        negative_samples=args.negative_samples,
        text_augmentation=not args.no_text_augmentation,
        augmentation_ratio=args.augmentation_ratio,
        token_dropout=args.token_dropout,
        punctuation_dropout=args.punctuation_dropout,
        node_dropout=args.node_dropout,
        edge_dropout=args.edge_dropout,
        subgraph_sampling_ratio=args.subgraph_sampling_ratio,
        contrastive_weight=args.contrastive_weight,
        curriculum_learning=not args.no_curriculum,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        checkpoint_metric=args.checkpoint_metric,
        max_grad_norm=args.max_grad_norm,
        dynamic_graph=args.dynamic_graph,
        graph_build_batch_size=args.graph_build_batch_size,
        graph_cache_dir=args.graph_cache_dir,
        reuse_graph_cache=not args.no_reuse_graph_cache,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        amp=args.amp,
        precision=args.precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compile_model=args.compile_model,
        sharded_checkpoint=args.sharded_checkpoint,
        distributed_backend=args.distributed_backend,
        resume_from=args.resume_from,
        tokenizer_type=args.tokenizer_type,
        tokenizer_half_space=args.tokenizer_half_space,
        tokenizer_morph_splitting=args.tokenizer_morph_splitting,
        tokenizer_compound_verb_mode=args.tokenizer_compound_verb_mode,
        tokenizer_bpe_merges=args.tokenizer_bpe_merges,
        tokenizer_unigram_num_pieces=args.unigram_num_pieces,
        tokenizer_byte_fallback=args.tokenizer_byte_fallback,
        device=device,
        seed=args.seed,
    )
    return train_graph_lm(
        _load_corpus(args.corpus),
        training_config=training_config,
        model_config=model_config,
        graph_encoder=args.graph_encoder,
        fusion=args.fusion,
    )


def _run_generate(args: argparse.Namespace) -> str:
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model, tokenizer, generation_config, graph_config = GraphCausalLM.from_pretrained(
        args.model,
        map_location=device,
    )
    if args.min_new_tokens is not None:
        generation_config.min_new_tokens = args.min_new_tokens
    if args.temperature is not None:
        generation_config.temperature = args.temperature
    if args.top_k is not None:
        generation_config.top_k = args.top_k
    if args.repetition_penalty is not None:
        generation_config.repetition_penalty = args.repetition_penalty
    model.to(device)
    graph_data, token_node_ids = GraphCausalLM.load_graph_artifacts(
        args.model,
        map_location=device,
    )
    graph_memory = None
    graph_memory_config = GraphMemoryConfig(
        enabled=args.graph_memory == "on",
        top_k_nodes=args.graph_memory_top_k,
        depth=args.graph_memory_depth,
        max_edges=args.graph_memory_max_edges,
        min_score=args.graph_memory_min_score,
        relation_weights=_parse_relation_weights(args.graph_memory_relation_weights),
    )
    dynamic_graph_config = None
    corpus_path = Path(args.model) / "corpus.txt"
    if model.config.graph_encoder != "none" and args.graph_memory == "on":
        graph_memory, saved_memory_config = GraphMemoryArtifact.load(
            args.model,
            map_location=device,
        )
        if graph_memory is None and corpus_path.exists():
            graph_memory = GraphMemoryArtifact.from_corpus(
                _load_corpus(str(corpus_path)),
                tokenizer,
                graph_config,
            )
        elif (
            graph_memory is None
            and graph_data is not None
            and token_node_ids is not None
        ):
            graph_memory = GraphMemoryArtifact.from_pyg_data(
                graph_data,
                token_node_ids,
                tokenizer,
                graph_config,
            )
        graph_memory_config.enabled = (
            saved_memory_config.enabled and graph_memory_config.enabled
        )
    if (
        model.config.graph_encoder != "none"
        and bool(graph_config.get("dynamic_graph", False))
        and graph_memory is None
    ):
        graph_data = None
        token_node_ids = None
        dynamic_graph_config = graph_config
    elif graph_data is not None and token_node_ids is not None:
        graph_data = graph_data.to(device)
        token_node_ids = token_node_ids.to(device)
    elif model.config.graph_encoder != "none" and corpus_path.exists():
        graph = build_graph_lm_graph(
            _load_corpus(str(corpus_path)),
            tokenizer,
            window_size=int(graph_config.get("window_size", 4)),
            min_count=int(graph_config.get("min_count", 1)),
            weighting=str(graph_config.get("weighting", "distance")),
            min_edge_weight=float(graph_config.get("min_edge_weight", 0.0)),
            top_k=graph_config.get("top_k"),  # type: ignore[arg-type]
            directed=bool(graph_config.get("directed", False)),
            graph_scope=str(graph_config.get("graph_scope", "document")),
            context_node_type=str(graph_config.get("context_node_type", "none")),
            graph_relations=graph_config.get(  # type: ignore[arg-type]
                "enabled_relations"
            ),
            relation_weights=graph_config.get(  # type: ignore[arg-type]
                "relation_weights"
            ),
            semantic_similarity_threshold=float(
                graph_config.get("semantic_similarity_threshold", 0.6)
            ),
            semantic_top_k=graph_config.get("semantic_top_k"),  # type: ignore[arg-type]
            semantic_method=str(graph_config.get("semantic_method", "distributional")),
            linguistic_backend=str(graph_config.get("linguistic_backend", "auto")),
            topic_top_k=int(graph_config.get("topic_top_k", 8)),
        )
        graph_data = graph.to_pyg_data().to(device)
        token_node_ids = graph.token_node_ids(tokenizer.vocab_size).to(device)
    else:
        graph_data = None
        token_node_ids = None
    if graph_memory is not None and args.graph_memory_report_path:
        report = graph_memory.retrieve(
            args.prompt,
            tokenizer,
            config=graph_memory_config,
        ).report
        report_path = Path(args.graph_memory_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return model.generate(
        args.prompt,
        tokenizer,
        graph_data=graph_data,
        token_node_ids=token_node_ids,
        graph_memory=graph_memory,
        graph_memory_config=graph_memory_config,
        generation_config=generation_config,
        dynamic_graph_config=dynamic_graph_config,
        max_new_tokens=args.max_new_tokens,
    )


def _run_dataset_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    texts, raw_labels = _load_text_dataset(
        args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        dataset_format=args.dataset_format,
    )
    encoded_doc_labels, label_to_id = _encode_labels(raw_labels)
    tokenised = [tokenize(text) for text in texts]
    graph = build_text_graph(
        tokenised,
        window_size=args.window_size,
        min_count=args.min_count,
    )

    n_word_nodes = len(graph.nodes) - len(texts)
    doc_node_indices = np.arange(n_word_nodes, len(graph.nodes))
    train_docs, val_docs, test_docs = _split_indices(
        len(texts),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_nodes = doc_node_indices[train_docs]
    val_nodes = doc_node_indices[val_docs]
    test_nodes = doc_node_indices[test_docs]

    node_labels = np.zeros(len(graph.nodes), dtype=int)
    node_labels[doc_node_indices] = encoded_doc_labels
    train_mask = _make_node_mask(len(graph.nodes), train_nodes)
    features = np.eye(len(graph.nodes), dtype=float)

    model, losses = train_node_classifier(
        graph,
        node_labels,
        X=features,
        mask=train_mask,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        device=device,
        gat_heads=args.gat_heads,
    )
    data = graph_to_data(graph, features=features, labels=node_labels).to(device)

    val_mask = _make_node_mask(len(graph.nodes), val_nodes)
    test_mask = _make_node_mask(len(graph.nodes), test_nodes)
    report: dict[str, Any] = {
        "dataset": str(args.dataset),
        "model": args.model,
        "device": device,
        "num_documents": len(texts),
        "num_nodes": len(graph.nodes),
        "num_classes": len(label_to_id),
        "label_to_id": label_to_id,
        "splits": {
            "train": _evaluate_split(model, data, node_labels, train_mask),
            "validation": _evaluate_split(model, data, node_labels, val_mask),
            "test": _evaluate_split(model, data, node_labels, test_mask),
        },
        "final_loss": losses[-1] if losses else None,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.report_path) if args.report_path else output_dir / "metrics.json"
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    report["report_path"] = str(report_path)

    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": args.model,
                "input_dim": features.shape[1],
                "hidden_dim": args.hidden_dim,
                "num_classes": len(label_to_id),
                "label_to_id": label_to_id,
                "metrics": report,
            },
            model_path,
        )
        report["model_path"] = str(model_path)

    return report


def _build_lm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Graph-LM models and generate Persian text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_corpus = subparsers.add_parser(
        "lm-build-corpus",
        help="Clean and split local Persian text for independent LM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_corpus.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input TXT, JSONL, JSON, CSV or TSV files",
    )
    build_corpus.add_argument("--output-dir", required=True)
    build_corpus.add_argument(
        "--input-format",
        choices=["auto", "txt", "json", "jsonl", "csv", "tsv"],
        default="auto",
    )
    build_corpus.add_argument(
        "--text-fields",
        nargs="+",
        default=["text", "body"],
        help="Fields concatenated when reading structured files",
    )
    build_corpus.add_argument("--source-id", default=None)
    build_corpus.add_argument("--min-chars", type=int, default=20)
    build_corpus.add_argument("--min-persian-ratio", type=float, default=0.35)
    build_corpus.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    build_corpus.add_argument("--validation-ratio", type=float, default=0.1)
    build_corpus.add_argument("--test-ratio", type=float, default=0.05)
    build_corpus.add_argument("--eval-paths", nargs="*", default=[])
    build_corpus.add_argument("--seed", type=int, default=0)
    build_corpus.add_argument("--log-level", default="INFO")

    tokenize_cmd = subparsers.add_parser(
        "lm-tokenize",
        help="Train/load a native tokenizer and write memory-mapped token shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tokenize_cmd.add_argument("--input", nargs="*", default=None)
    tokenize_cmd.add_argument("--corpus-dir", default=None)
    tokenize_cmd.add_argument("--output-dir", required=True)
    tokenize_cmd.add_argument("--tokenizer", default=None, help="Existing tokenizer.json to reuse")
    tokenize_cmd.add_argument(
        "--tokenizer-type",
        choices=["word", "subword", "char_chunk", "bpe", "unigram"],
        default="unigram",
    )
    tokenize_cmd.add_argument(
        "--tokenizer-half-space",
        choices=["preserve", "split"],
        default="preserve",
    )
    tokenize_cmd.add_argument("--tokenizer-morph-splitting", action="store_true")
    tokenize_cmd.add_argument(
        "--tokenizer-compound-verb-mode",
        choices=["none", "join"],
        default="none",
    )
    tokenize_cmd.add_argument("--tokenizer-bpe-merges", type=int, default=200)
    tokenize_cmd.add_argument("--unigram-num-pieces", type=int, default=8000)
    tokenize_cmd.add_argument("--max-vocab-size", type=int, default=None)
    tokenize_cmd.add_argument("--tokenizer-byte-fallback", action="store_true")
    tokenize_cmd.add_argument("--block-size", type=int, default=128)
    tokenize_cmd.add_argument("--stride", type=int, default=None)
    tokenize_cmd.add_argument("--tokens-per-shard", type=int, default=2_000_000)
    tokenize_cmd.add_argument("--seed", type=int, default=0)
    tokenize_cmd.add_argument("--log-level", default="INFO")

    ablation = subparsers.add_parser(
        "lm-ablation",
        help="Run native text-only/graph Graph-LM ablation variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ablation.add_argument("--corpus", required=True)
    ablation.add_argument("--output-dir", required=True)
    ablation.add_argument(
        "--graph-encoders",
        nargs="+",
        choices=["none", "gcn", "graphsage", "gat", "rgcn"],
        default=["none", "gat"],
    )
    ablation.add_argument(
        "--graph-scopes",
        nargs="+",
        choices=["corpus", "document", "sentence"],
        default=["document"],
    )
    ablation.add_argument("--relation-groups", default=None)
    ablation.add_argument("--epochs", type=int, default=1)
    ablation.add_argument("--batch-size", type=int, default=1)
    ablation.add_argument("--block-size", type=int, default=64)
    ablation.add_argument("--validation-ratio", type=float, default=0.1)
    ablation.add_argument(
        "--tokenizer-type",
        choices=["word", "subword", "char_chunk", "bpe", "unigram"],
        default="unigram",
    )
    ablation.add_argument("--unigram-num-pieces", type=int, default=8000)
    ablation.add_argument("--graph-cache-dir", default=None)
    ablation.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ablation.add_argument("--seed", type=int, default=0)
    ablation.add_argument("--log-level", default="INFO")

    eval_cmd = subparsers.add_parser(
        "lm-eval",
        help="Run local-only evaluation for a saved Graph-LM checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_cmd.add_argument("--model", required=True)
    eval_cmd.add_argument("--eval-file", required=True)
    eval_cmd.add_argument("--output-path", default=None)
    eval_cmd.add_argument("--text-field", default="text")
    eval_cmd.add_argument("--prompt-field", default="prompt")
    eval_cmd.add_argument("--choices-field", default="choices")
    eval_cmd.add_argument("--answer-field", default="answer")
    eval_cmd.add_argument("--prediction-field", default="prediction")
    eval_cmd.add_argument("--block-size", type=int, default=128)
    eval_cmd.add_argument("--generation-prompts", nargs="*", default=[])
    eval_cmd.add_argument("--max-new-tokens", type=int, default=32)
    eval_cmd.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    eval_cmd.add_argument("--seed", type=int, default=0)
    eval_cmd.add_argument("--log-level", default="INFO")

    sft_cmd = subparsers.add_parser(
        "lm-sft",
        help="Train on human-authored local instruction data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sft_cmd.add_argument("--input", required=True)
    sft_cmd.add_argument("--output-dir", required=True)
    sft_cmd.add_argument("--prompt-field", default="prompt")
    sft_cmd.add_argument("--completion-field", default="completion")
    sft_cmd.add_argument("--messages-field", default="messages")
    sft_cmd.add_argument("--min-completion-chars", type=int, default=1)
    sft_cmd.add_argument("--epochs", type=int, default=1)
    sft_cmd.add_argument("--batch-size", type=int, default=1)
    sft_cmd.add_argument("--block-size", type=int, default=128)
    sft_cmd.add_argument("--validation-ratio", type=float, default=0.1)
    sft_cmd.add_argument(
        "--tokenizer-type",
        choices=["word", "subword", "char_chunk", "bpe", "unigram"],
        default="unigram",
    )
    sft_cmd.add_argument("--unigram-num-pieces", type=int, default=8000)
    sft_cmd.add_argument("--d-model", type=int, default=64)
    sft_cmd.add_argument("--n-heads", type=int, default=4)
    sft_cmd.add_argument("--n-layers", type=int, default=1)
    sft_cmd.add_argument("--dim-feedforward", type=int, default=128)
    sft_cmd.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    sft_cmd.add_argument("--seed", type=int, default=0)
    sft_cmd.add_argument("--log-level", default="INFO")

    pretrain = subparsers.add_parser(
        "lm-pretrain",
        help="Native text-only pretraining from corpus text or token shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pretrain.add_argument("--corpus", default=None)
    pretrain.add_argument("--shard-manifest", default=None)
    pretrain.add_argument("--output-dir", required=True)
    pretrain.add_argument("--model-profile", choices=available_model_profiles(), default=None)
    pretrain.add_argument("--context-size", type=int, choices=[512, 1024, 2048, 4096], default=None)
    pretrain.add_argument("--d-model", type=int, default=64)
    pretrain.add_argument("--n-heads", type=int, default=4)
    pretrain.add_argument("--n-layers", type=int, default=1)
    pretrain.add_argument("--dim-feedforward", type=int, default=128)
    pretrain.add_argument("--attention-backend", choices=["auto", "math", "flash", "memory_efficient"], default="auto")
    pretrain.add_argument("--activation-checkpointing", action="store_true")
    pretrain.add_argument("--epochs", type=int, default=1)
    pretrain.add_argument("--batch-size", type=int, default=1)
    pretrain.add_argument("--block-size", type=int, default=128)
    pretrain.add_argument("--validation-ratio", type=float, default=0.1)
    pretrain.add_argument("--tokenizer-type", choices=["word", "subword", "char_chunk", "bpe", "unigram"], default="unigram")
    pretrain.add_argument("--unigram-num-pieces", type=int, default=8000)
    pretrain.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    pretrain.add_argument("--gradient-accumulation-steps", type=int, default=1)
    pretrain.add_argument("--early-stopping-patience", type=int, default=0)
    pretrain.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    pretrain.add_argument("--seed", type=int, default=0)
    pretrain.add_argument("--log-level", default="INFO")

    run_report = subparsers.add_parser(
        "lm-run-report",
        help="Build a consolidated report for a native LM run directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_report.add_argument("--run-dir", required=True)
    run_report.add_argument("--output-path", default=None)
    run_report.add_argument("--seed", type=int, default=0)
    run_report.add_argument("--log-level", default="INFO")

    train = subparsers.add_parser(
        "lm-train",
        help="Train a Persian Graph causal language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train.add_argument(
        "--corpus",
        required=True,
        help="Plain text corpus, one sample per line",
    )
    train.add_argument("--output-dir", default="runs/graph-lm")
    train.add_argument(
        "--graph-encoder",
        choices=["none", "gcn", "graphsage", "gat", "rgcn"],
        default="gat",
    )
    train.add_argument(
        "--fusion",
        choices=["gated", "context_gated", "add"],
        default="gated",
    )
    train.add_argument("--fusion-layers", choices=["input", "all"], default="input")
    train.add_argument(
        "--fusion-levels",
        default="token",
        help=(
            "Comma-separated adaptive fusion levels: token, sentence, subgraph, "
            "or all. Example: token,sentence,subgraph"
        ),
    )
    train.add_argument(
        "--graph-fusion-scale",
        type=float,
        default=1.0,
        help="Multiplier for graph embeddings before fusion",
    )
    train.add_argument(
        "--graph-fusion-dropout",
        type=float,
        default=0.0,
        help="Dropout applied to graph embeddings inside fusion",
    )
    train.add_argument(
        "--task-losses",
        default=(
            "next_token,masked_token,edge,neighbor,node_relation,"
            "graph_text,sentence_graph"
        ),
        help=(
            "Comma-separated multi-task losses. Use all, or any of next_token, "
            "masked_token, edge, neighbor, node_relation, graph_text, sentence_graph"
        ),
    )
    train.add_argument("--next-token-weight", type=float, default=1.0)
    train.add_argument("--masked-token-weight", type=float, default=0.25)
    train.add_argument("--edge-prediction-weight", type=float, default=0.1)
    train.add_argument("--neighbor-prediction-weight", type=float, default=0.1)
    train.add_argument("--node-relation-weight", type=float, default=0.1)
    train.add_argument("--graph-text-alignment-weight", type=float, default=0.1)
    train.add_argument("--sentence-graph-alignment-weight", type=float, default=0.1)
    train.add_argument(
        "--mask-probability",
        type=float,
        default=0.15,
        help="Probability of selecting non-padding tokens for masked-token loss",
    )
    train.add_argument(
        "--negative-samples",
        type=int,
        default=1,
        help="Number of sampled negative graph pairs per positive edge",
    )
    train.add_argument(
        "--no-text-augmentation",
        action="store_true",
        help="Disable Phase 7 text augmentation for low-data training",
    )
    train.add_argument(
        "--augmentation-ratio",
        type=float,
        default=0.5,
        help="Additional augmented training examples as a fraction of train corpus",
    )
    train.add_argument(
        "--token-dropout",
        type=float,
        default=0.05,
        help="Probability of dropping a token in augmented text examples",
    )
    train.add_argument(
        "--punctuation-dropout",
        type=float,
        default=0.5,
        help="Probability of removing punctuation in augmented text examples",
    )
    train.add_argument(
        "--node-dropout",
        type=float,
        default=0.05,
        help="Probability of masking graph nodes through incident-edge removal",
    )
    train.add_argument(
        "--edge-dropout",
        type=float,
        default=0.1,
        help="Probability of dropping graph edges during training",
    )
    train.add_argument(
        "--subgraph-sampling-ratio",
        type=float,
        default=0.9,
        help="Fraction of graph edges retained for each sampled training view",
    )
    train.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.05,
        help="Weight for Phase 7 graph-view contrastive consistency loss",
    )
    train.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable Phase 7 curriculum ordering of LM windows",
    )
    train.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop after this many epochs without validation improvement; 0 disables",
    )
    train.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement counted by early stopping",
    )
    train.add_argument(
        "--checkpoint-metric",
        choices=["next_token", "total"],
        default="next_token",
        help=(
            "Validation signal for best-checkpoint selection and early stopping: "
            "'next_token' (perplexity, recommended) or 'total' multi-task loss"
        ),
    )
    train.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm used by the LM trainer",
    )
    train.add_argument(
        "--model-profile",
        choices=available_model_profiles(),
        default=None,
        help="Named native model-size profile; omitted keeps manual sizing flags",
    )
    train.add_argument(
        "--context-size",
        type=int,
        choices=[512, 1024, 2048, 4096],
        default=None,
        help="Context-size override used with --model-profile",
    )
    train.add_argument("--d-model", type=int, default=128)
    train.add_argument("--n-heads", type=int, default=4)
    train.add_argument("--n-layers", type=int, default=2)
    train.add_argument("--dim-feedforward", type=int, default=512)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument(
        "--attention-backend",
        choices=["auto", "math", "flash", "memory_efficient"],
        default="auto",
        help="PyTorch scaled-dot-product attention backend preference",
    )
    train.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Checkpoint decoder layer activations during training",
    )
    train.add_argument("--graph-hidden-dim", type=int, default=128)
    train.add_argument("--graph-heads", type=int, default=4)
    train.add_argument(
        "--graph-relation-mode",
        choices=["bias", "embedding", "rgcn"],
        default="embedding",
        help=(
            "How the graph encoder consumes edge_type relation ids; 'embedding' "
            "learns a per-relation vector and best uses the multi-relation graph"
        ),
    )
    train.add_argument(
        "--graph-pooling",
        choices=["none", "mean", "attention"],
        default="none",
        help="Optional subgraph/global pooling added to graph node embeddings",
    )
    train.add_argument(
        "--graph-node-importance",
        action="store_true",
        help="Enable node-importance scoring inside the graph encoder",
    )
    train.add_argument(
        "--no-graph-node-type-embedding",
        dest="graph_node_type_embedding",
        action="store_false",
        help="Disable learned per-node-type embeddings for non-token graph nodes",
    )
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batches before each optimizer step",
    )
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "linear"],
        default="cosine",
    )
    train.add_argument("--warmup-ratio", type=float, default=0.05)
    train.add_argument("--warmup-steps", type=int, default=None)
    train.add_argument("--min-lr-ratio", type=float, default=0.0)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--adam-beta1", type=float, default=0.9)
    train.add_argument("--adam-beta2", type=float, default=0.999)
    train.add_argument("--adam-eps", type=float, default=1e-8)
    train.add_argument("--validation-ratio", type=float, default=0.1)
    train.add_argument("--block-size", type=int, default=128)
    train.add_argument("--stride", type=int, default=None)
    train.add_argument("--min-freq", type=int, default=1)
    train.add_argument("--max-vocab-size", type=int, default=None)
    train.add_argument("--graph-window-size", type=int, default=4)
    train.add_argument("--graph-min-count", type=int, default=1)
    train.add_argument(
        "--graph-weighting",
        choices=["distance", "count", "raw", "pmi", "ppmi"],
        default="distance",
    )
    train.add_argument("--graph-min-edge-weight", type=float, default=0.0)
    train.add_argument("--graph-top-k", type=int, default=None)
    train.add_argument("--graph-directed", action="store_true")
    train.add_argument(
        "--graph-scope",
        choices=["corpus", "document", "sentence"],
        default="document",
    )
    train.add_argument(
        "--context-node-type",
        choices=["none", "document", "sentence"],
        default="none",
    )
    train.add_argument(
        "--graph-relations",
        nargs="+",
        default=None,
        choices=[
            "cooccurrence",
            "pmi",
            "ppmi",
            "dependency",
            "stem",
            "subword",
            "semantic_similarity",
            "semantic",
            "word_document",
            "topic_document",
        ],
        help="Relation types to include in the Phase 3 multi-relation graph",
    )
    train.add_argument(
        "--relation-weights",
        default=None,
        help="Comma-separated relation weights, for example cooccurrence=1,pmi=0.8",
    )
    train.add_argument("--semantic-similarity-threshold", type=float, default=0.6)
    train.add_argument("--semantic-top-k", type=int, default=4)
    train.add_argument(
        "--semantic-method",
        choices=["distributional", "orthographic"],
        default="distributional",
        help=(
            "semantic_similarity relation: 'distributional' (PPMI-cosine, real "
            "semantics) or 'orthographic' (character-overlap heuristic)"
        ),
    )
    train.add_argument(
        "--linguistic-backend",
        choices=["auto", "stanza", "heuristic"],
        default="auto",
        help=(
            "Backend for dependency/lemma relations: 'auto' uses Stanza when "
            "installed and falls back to heuristics, 'stanza' forces it, "
            "'heuristic' disables it"
        ),
    )
    train.add_argument("--topic-top-k", type=int, default=8)
    train.add_argument("--dynamic-graph", action="store_true")
    train.add_argument(
        "--graph-build-batch-size",
        type=int,
        default=None,
        help="Build co-occurrence graph statistics in batches of this many text units",
    )
    train.add_argument(
        "--graph-cache-dir",
        default=None,
        help="Directory for reusable Phase 9 graph cache artifacts",
    )
    train.add_argument(
        "--no-reuse-graph-cache",
        action="store_true",
        help="Rebuild the graph even when a matching graph cache artifact exists",
    )
    train.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="Number of PyTorch DataLoader worker processes for LM training",
    )
    train.add_argument(
        "--dataloader-pin-memory",
        action="store_true",
        help="Pin DataLoader memory, useful when training on CUDA",
    )
    train.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision for LM training",
    )
    train.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="Autocast precision mode; CPU falls back safely to fp32",
    )
    train.add_argument("--compile-model", action="store_true")
    train.add_argument("--sharded-checkpoint", action="store_true")
    train.add_argument(
        "--distributed-backend",
        choices=["none", "ddp", "fsdp"],
        default="none",
        help="Wrap with PyTorch DDP/FSDP only when torch.distributed is initialized",
    )
    train.add_argument(
        "--resume-from",
        default=None,
        help="Resume LM model and optimizer state from a previous checkpoint directory",
    )
    train.add_argument(
        "--tokenizer-type",
        choices=["word", "subword", "char_chunk", "bpe", "unigram"],
        default="unigram",
        help="Subword tokenizer; 'unigram' is recommended for Persian (low OOV)",
    )
    train.add_argument(
        "--tokenizer-half-space",
        choices=["preserve", "split"],
        default="preserve",
    )
    train.add_argument("--tokenizer-morph-splitting", action="store_true")
    train.add_argument(
        "--tokenizer-compound-verb-mode",
        choices=["none", "join"],
        default="none",
    )
    train.add_argument("--tokenizer-bpe-merges", type=int, default=200)
    train.add_argument(
        "--unigram-num-pieces",
        type=int,
        default=8000,
        help="Target subword vocabulary size for the unigram tokenizer",
    )
    train.add_argument(
        "--tokenizer-byte-fallback",
        action="store_true",
        help="Encode out-of-vocabulary tokens as native UTF-8 byte tokens",
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    train.add_argument("--log-level", default="INFO")

    generate = subparsers.add_parser(
        "generate",
        help="Generate text from a saved Graph-LM checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate.add_argument(
        "--model",
        required=True,
        help="Directory containing Graph-LM files",
    )
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=None)
    generate.add_argument("--min-new-tokens", type=int, default=None)
    generate.add_argument("--temperature", type=float, default=None)
    generate.add_argument("--top-k", type=int, default=None)
    generate.add_argument("--repetition-penalty", type=float, default=None)
    generate.add_argument(
        "--graph-memory",
        choices=["on", "off"],
        default="on",
        help="Use prompt-aware Graph Memory retrieval during generation",
    )
    generate.add_argument(
        "--graph-memory-top-k",
        type=int,
        default=32,
        help="Maximum graph memory nodes retrieved for the prompt",
    )
    generate.add_argument(
        "--graph-memory-depth",
        type=int,
        default=1,
        help="Neighbour expansion depth for graph memory retrieval",
    )
    generate.add_argument(
        "--graph-memory-max-edges",
        type=int,
        default=256,
        help="Maximum graph memory edges kept in the retrieved subgraph",
    )
    generate.add_argument(
        "--graph-memory-min-score",
        type=float,
        default=0.0,
        help="Minimum retrieval score for graph memory nodes",
    )
    generate.add_argument(
        "--graph-memory-relation-weights",
        default=None,
        help="Optional relation=value weights for graph memory retrieval",
    )
    generate.add_argument(
        "--graph-memory-report-path",
        default=None,
        help="Optional JSON path for graph memory retrieval diagnostics",
    )
    generate.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    generate.add_argument("--log-level", default="INFO")
    add_article_subcommands(subparsers)
    return parser


def _build_classification_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate graph-based Persian text classifiers, or run a "
            "small built-in smoke experiment when no dataset is provided."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON config file whose keys match CLI option names",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        help="CSV, TSV or JSONL labelled text dataset for training/evaluation",
        default=None,
    )
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "csv", "tsv", "jsonl"],
        default="auto",
        help="Dataset file format; auto detects from the file extension",
    )
    parser.add_argument("--text-column", default="text", help="Name of the text field")
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the class label field",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Relative share of labelled documents used for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Relative share of labelled documents used for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Relative share of labelled documents used for final testing",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/rgnn",
        help="Directory where metrics.json is written unless --report-path is set",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Explicit path for the JSON metrics report",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Optional path for saving the trained PyTorch model checkpoint",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Token co-occurrence window used when building the text graph",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum token frequency required to keep a word node",
    )
    parser.add_argument(
        "--model",
        choices=["gcn", "graphsage", "gat"],
        default="gcn",
        help="Graph neural network architecture to train",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splits")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, such as DEBUG, INFO or WARNING",
    )
    parser.add_argument(
        "--log-to",
        choices=["wandb", "mlflow"],
        default=None,
        help="Optional experiment tracker used by the built-in smoke experiment",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=8,
        help="Hidden representation size in the GNN model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Optimizer L2 regularization strength",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability used by the GNN model",
    )
    parser.add_argument(
        "--gat-heads",
        type=int,
        default=4,
        help="Number of attention heads when --model gat is selected",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device; cuda falls back to cpu when unavailable",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    lm_commands = {
        "lm-build-corpus",
        "lm-tokenize",
        "lm-ablation",
        "lm-eval",
        "lm-sft",
        "lm-pretrain",
        "lm-run-report",
        "lm-train",
        "generate",
        *ARTICLE_COMMANDS,
    }
    if argv and argv[0] in lm_commands:
        parser = _build_lm_parser()
        args = parser.parse_args(argv)
        logger = setup_logger(getattr(args, "log_level", "INFO"))
        set_seed(getattr(args, "seed", 0))
        if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            args.device = "cpu"
        if args.command == "lm-build-corpus":
            manifest = _run_lm_build_corpus(args)
            logger.info(
                "lm_corpus output=%s accepted=%s rejected=%s",
                args.output_dir,
                manifest["quality_report"]["records_accepted"],
                manifest["quality_report"]["records_rejected"],
            )
            return 0
        if args.command == "lm-tokenize":
            manifest = _run_lm_tokenize(args)
            logger.info(
                "lm_tokenize output=%s shards=%s vocab=%s",
                args.output_dir,
                len(manifest["shards"]),
                manifest["audit"].get("train", {}).get("vocab_size"),
            )
            return 0
        if args.command == "lm-ablation":
            report = _run_lm_ablation(args)
            logger.info("lm_ablation output=%s variants=%s", args.output_dir, len(report["variants"]))
            return 0
        if args.command == "lm-eval":
            report = _run_lm_eval(args)
            logger.info("lm_eval rows=%s output=%s", report["row_count"], args.output_path)
            return 0
        if args.command == "lm-sft":
            metrics = _run_lm_sft(args)
            write_run_registry(
                RunRegistryConfig(
                    run_dir=args.output_dir,
                    command=["lm-sft"],
                    data_paths=[args.input],
                    checkpoint_dir=args.output_dir,
                )
            )
            logger.info(
                "lm_sft checkpoint=%s records=%s",
                metrics["checkpoint_dir"],
                metrics["sft_manifest"]["records_used"],
            )
            return 0
        if args.command == "lm-pretrain":
            metrics = _run_lm_pretrain(args)
            logger.info(
                "lm_pretrain checkpoint=%s perplexity=%.3f",
                metrics["checkpoint_dir"],
                metrics["best_perplexity"],
            )
            return 0
        if args.command == "lm-run-report":
            report = _run_lm_run_report(args)
            logger.info("lm_run_report run_dir=%s files=%s", args.run_dir, len(report["files"]))
            return 0
        if args.command == "lm-train":
            metrics = _run_lm_train(args)
            write_run_registry(
                RunRegistryConfig(
                    run_dir=args.output_dir,
                    command=["lm-train"],
                    data_paths=[args.corpus],
                    checkpoint_dir=args.output_dir,
                )
            )
            logger.info(
                "graph_lm checkpoint=%s validation_loss=%.3f perplexity=%.3f",
                metrics["checkpoint_dir"],
                metrics["best_validation_loss"],
                metrics["best_perplexity"],
            )
            corpus_copy = Path(args.output_dir) / "corpus.txt"
            corpus_copy.write_text(
                "\n".join(_load_corpus(args.corpus)) + "\n",
                encoding="utf-8",
            )
            return 0
        if args.command == "generate":
            generated = _run_generate(args)
            print(generated)
            return 0
        if args.command in ARTICLE_COMMANDS:
            return run_article_command(args, logger)

    parser = _build_classification_parser()
    config_args, _ = parser.parse_known_args(argv)
    parser.set_defaults(**_load_config(config_args.config))
    args = parser.parse_args(argv)

    logger = setup_logger(args.log_level)
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    if args.dataset:
        report = _run_dataset_pipeline(args)
        val_metrics = report["splits"]["validation"]
        test_metrics = report["splits"]["test"]
        logger.info(
            "model=%s docs=%d val_accuracy=%.3f test_accuracy=%.3f report=%s",
            report["model"],
            report["num_documents"],
            val_metrics["accuracy"],
            test_metrics["accuracy"],
            report["report_path"],
        )
        return 0

    g, X, y = _synthetic_graph()
    model, _ = train_node_classifier(
        g,
        y,
        X=X,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        num_epochs=args.epochs,
        device=args.device,
    )
    data = graph_to_data(g, features=X, labels=y).to(args.device)
    preds = model.predict(data).cpu().numpy()
    acc = accuracy(y, preds)
    logger.info("model=%s accuracy=%.3f", args.model, acc)

    if args.log_to == "wandb":  # pragma: no cover - optional
        try:
            import wandb

            wandb.init(project="rgnn")
            wandb.log({"accuracy": acc})
        except Exception:  # pragma: no cover
            logger.warning("wandb not available")
    elif args.log_to == "mlflow":  # pragma: no cover
        try:
            import mlflow

            mlflow.log_metric("accuracy", acc)
        except Exception:  # pragma: no cover
            logger.warning("mlflow not available")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI wiring for the native Persian article LLM workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .core import (
    DEFAULT_ARTICLE_GENERATION_AUDIENCE,
    DEFAULT_ARTICLE_GENERATION_GRAPH_MEMORY,
    DEFAULT_ARTICLE_GENERATION_MAX_NEW_TOKENS,
    DEFAULT_ARTICLE_GENERATION_MIN_NEW_TOKENS,
    DEFAULT_ARTICLE_GENERATION_REPETITION_PENALTY,
    DEFAULT_ARTICLE_GENERATION_SECTIONS,
    DEFAULT_ARTICLE_GENERATION_TEMPERATURE,
    DEFAULT_ARTICLE_GENERATION_TONE,
    DEFAULT_ARTICLE_GENERATION_TOP_K,
    DEFAULT_ARTICLE_GRAPH_RELATIONS,
)
from . import (
    ArticleAblationConfig,
    ArticleAuditConfig,
    ArticleCorpusConfig,
    ArticleGenerationConfig,
    ArticleTrainingConfig,
    audit_article_corpus,
    generate_persian_article,
    prepare_article_corpus,
    run_article_ablation,
    train_article_llm,
)

ARTICLE_COMMANDS = frozenset(
    {
        "article-prepare",
        "article-audit",
        "article-train",
        "article-ablation",
        "article-generate",
    }
)


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


def _parse_relation_groups(raw: str | None) -> dict[str, list[str]] | None:
    if not raw:
        return None
    groups: dict[str, list[str]] = {}
    for group in raw.split(";"):
        if not group.strip():
            continue
        if "=" not in group:
            raise ValueError("--relation-groups entries must use name=rel,rel")
        name, values = group.split("=", 1)
        relations = [item.strip() for item in values.split(",") if item.strip()]
        if relations:
            groups[name.strip()] = relations
    return groups or None


def _run_article_prepare(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_article_corpus(
        ArticleCorpusConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            input_format=args.input_format,
            training_format=args.training_format,
            title_field=args.title_field,
            body_field=args.body_field,
            summary_field=args.summary_field,
            keywords_field=args.keywords_field,
            metadata_field=args.metadata_field,
            prompt_audience=args.prompt_audience,
            prompt_tone=args.prompt_tone,
            prompt_sections=args.prompt_sections,
            min_body_chars=args.min_body_chars,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
    )


def _run_article_audit(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    return audit_article_corpus(
        ArticleAuditConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            input_format=args.input_format,
            training_format=args.training_format,
            title_field=args.title_field,
            body_field=args.body_field,
            summary_field=args.summary_field,
            keywords_field=args.keywords_field,
            metadata_field=args.metadata_field,
            min_body_chars=args.min_body_chars,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            tokenizer_types=args.tokenizer_types,
            tokenizer_half_space=args.tokenizer_half_space,
            tokenizer_morph_splitting=args.tokenizer_morph_splitting,
            tokenizer_compound_verb_mode=args.tokenizer_compound_verb_mode,
            tokenizer_bpe_merges=args.tokenizer_bpe_merges,
            tokenizer_unigram_num_pieces=args.unigram_num_pieces,
            tokenizer_max_vocab_size=args.tokenizer_max_vocab_size,
            tokenizer_probe_epochs=args.tokenizer_probe_epochs,
            tokenizer_probe_block_size=args.tokenizer_probe_block_size,
            tokenizer_probe_d_model=args.tokenizer_probe_d_model,
            tokenizer_probe_n_heads=args.tokenizer_probe_n_heads,
            tokenizer_probe_n_layers=args.tokenizer_probe_n_layers,
            tokenizer_probe_batch_size=args.tokenizer_probe_batch_size,
            near_duplicate_ngram_size=args.near_duplicate_ngram_size,
            near_duplicate_threshold=args.near_duplicate_threshold,
            near_duplicate_max_pairs=args.near_duplicate_max_pairs,
            max_examples=args.max_examples,
            device=device,
        )
    )


def _run_article_train(args: argparse.Namespace) -> dict[str, Any]:
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    return train_article_llm(
        ArticleTrainingConfig(
            corpus_path=args.corpus,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            validation_ratio=args.validation_ratio,
            block_size=args.block_size,
            stride=args.stride,
            min_freq=args.min_freq,
            max_vocab_size=args.max_vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            graph_encoder=args.graph_encoder,
            graph_hidden_dim=args.graph_hidden_dim,
            graph_heads=args.graph_heads,
            fusion=args.fusion,
            fusion_layers=args.fusion_layers,
            fusion_levels=args.fusion_levels,
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
            resume_from=args.resume_from,
            tokenizer_type=args.tokenizer_type,
            tokenizer_half_space=args.tokenizer_half_space,
            tokenizer_morph_splitting=args.tokenizer_morph_splitting,
            tokenizer_compound_verb_mode=args.tokenizer_compound_verb_mode,
            tokenizer_bpe_merges=args.tokenizer_bpe_merges,
            tokenizer_unigram_num_pieces=args.unigram_num_pieces,
            device=device,
            seed=args.seed,
        )
    )


def _training_config_from_ablation_args(args: argparse.Namespace) -> ArticleTrainingConfig:
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    return ArticleTrainingConfig(
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        graph_hidden_dim=args.graph_hidden_dim,
        graph_heads=args.graph_heads,
        fusion=args.fusion,
        fusion_layers=args.fusion_layers,
        fusion_levels=args.fusion_levels,
        graph_window_size=args.graph_window_size,
        graph_min_count=args.graph_min_count,
        graph_weighting=args.graph_weighting,
        graph_scope=args.graph_scopes[0] if args.graph_scopes else "document",
        graph_relations=args.graph_relations,
        graph_relation_mode=args.graph_relation_mode,
        task_losses=args.task_losses,
        text_augmentation=not args.no_text_augmentation,
        contrastive_weight=args.contrastive_weight,
        edge_dropout=args.edge_dropout,
        node_dropout=args.node_dropout,
        subgraph_sampling_ratio=args.subgraph_sampling_ratio,
        tokenizer_type=args.tokenizer_type,
        tokenizer_bpe_merges=args.tokenizer_bpe_merges,
        tokenizer_unigram_num_pieces=args.unigram_num_pieces,
        device=device,
        seed=args.seed,
    )


def _run_article_ablation(args: argparse.Namespace) -> dict[str, Any]:
    base_output_dir = Path(args.output_dir) / "template"
    training_config = _training_config_from_ablation_args(args)
    training_config = ArticleTrainingConfig(
        **{**training_config.__dict__, "output_dir": str(base_output_dir)}
    )
    return run_article_ablation(
        ArticleAblationConfig(
            training_config=training_config,
            output_dir=args.output_dir,
            graph_encoders=args.graph_encoders,
            graph_scopes=args.graph_scopes,
            relation_groups=_parse_relation_groups(args.relation_groups),
            probe_topic=args.probe_topic,
            probe_sections=args.probe_sections,
            probe_max_new_tokens=args.probe_max_new_tokens,
        )
    )


def _run_article_generate(args: argparse.Namespace) -> str:
    article = generate_persian_article(
        ArticleGenerationConfig(
            model_dir=args.model,
            topic=args.topic,
            audience=args.audience,
            tone=args.tone,
            sections=args.sections,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            graph_memory=args.graph_memory == "on",
            graph_memory_top_k=args.graph_memory_top_k,
            graph_memory_depth=args.graph_memory_depth,
            graph_memory_max_edges=args.graph_memory_max_edges,
            graph_memory_min_score=args.graph_memory_min_score,
            graph_memory_relation_weights=_parse_relation_weights(
                args.graph_memory_relation_weights
            ),
            device=args.device,
        )
    )
    rendered = (
        article.to_json()
        if args.output_format == "json"
        else article.full_markdown
    )
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    return rendered


def add_article_subcommands(subparsers: Any) -> None:
    article_prepare = subparsers.add_parser(
        "article-prepare",
        help=(
            "Prepare Persian article datasets for native Article Graph-LM "
            "training"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    article_prepare.add_argument(
        "--input",
        required=True,
        help="TXT, JSONL, CSV or TSV dataset",
    )
    article_prepare.add_argument("--output-dir", required=True)
    article_prepare.add_argument(
        "--input-format",
        choices=["auto", "txt", "jsonl", "csv", "tsv"],
        default="auto",
    )
    article_prepare.add_argument("--title-field", default="title")
    article_prepare.add_argument("--body-field", default="body")
    article_prepare.add_argument("--summary-field", default="summary")
    article_prepare.add_argument("--keywords-field", default="keywords")
    article_prepare.add_argument("--metadata-field", default="metadata")
    article_prepare.add_argument(
        "--training-format",
        choices=["article_fields", "prompt_completion", "wikipedia_prompt"],
        default="article_fields",
        help=(
            "How each article is converted into LM text. wikipedia_prompt "
            "uses title/text-style records as prompt-completion examples."
        ),
    )
    article_prepare.add_argument("--prompt-audience", default="عمومی")
    article_prepare.add_argument("--prompt-tone", default="دانشنامه‌ای")
    article_prepare.add_argument("--prompt-sections", type=int, default=3)
    article_prepare.add_argument("--min-body-chars", type=int, default=80)
    article_prepare.add_argument("--validation-ratio", type=float, default=0.1)
    article_prepare.add_argument("--seed", type=int, default=0)
    article_prepare.add_argument("--log-level", default="INFO")

    article_audit = subparsers.add_parser(
        "article-audit",
        help=(
            "Audit Persian article data and compare native tokenizers before "
            "Article Graph-LM training"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    article_audit.add_argument("--input", required=True)
    article_audit.add_argument("--output-dir", required=True)
    article_audit.add_argument(
        "--input-format",
        choices=["auto", "txt", "jsonl", "csv", "tsv"],
        default="auto",
    )
    article_audit.add_argument("--title-field", default="title")
    article_audit.add_argument("--body-field", default="body")
    article_audit.add_argument("--summary-field", default="summary")
    article_audit.add_argument("--keywords-field", default="keywords")
    article_audit.add_argument("--metadata-field", default="metadata")
    article_audit.add_argument(
        "--training-format",
        choices=["article_fields", "prompt_completion", "wikipedia_prompt"],
        default="article_fields",
    )
    article_audit.add_argument("--min-body-chars", type=int, default=80)
    article_audit.add_argument("--validation-ratio", type=float, default=0.1)
    article_audit.add_argument(
        "--tokenizer-types",
        nargs="+",
        choices=["word", "char_chunk", "bpe", "unigram"],
        default=["word", "bpe", "unigram"],
    )
    article_audit.add_argument(
        "--tokenizer-half-space",
        choices=["preserve", "split"],
        default="preserve",
    )
    article_audit.add_argument("--tokenizer-morph-splitting", action="store_true")
    article_audit.add_argument(
        "--tokenizer-compound-verb-mode",
        choices=["none", "join"],
        default="none",
    )
    article_audit.add_argument("--tokenizer-bpe-merges", type=int, default=200)
    article_audit.add_argument("--unigram-num-pieces", type=int, default=8000)
    article_audit.add_argument("--tokenizer-max-vocab-size", type=int, default=None)
    article_audit.add_argument("--tokenizer-probe-epochs", type=int, default=0)
    article_audit.add_argument("--tokenizer-probe-block-size", type=int, default=64)
    article_audit.add_argument("--tokenizer-probe-d-model", type=int, default=64)
    article_audit.add_argument("--tokenizer-probe-n-heads", type=int, default=4)
    article_audit.add_argument("--tokenizer-probe-n-layers", type=int, default=1)
    article_audit.add_argument("--tokenizer-probe-batch-size", type=int, default=4)
    article_audit.add_argument("--near-duplicate-ngram-size", type=int, default=13)
    article_audit.add_argument("--near-duplicate-threshold", type=float, default=0.82)
    article_audit.add_argument("--near-duplicate-max-pairs", type=int, default=20000)
    article_audit.add_argument("--max-examples", type=int, default=None)
    article_audit.add_argument("--seed", type=int, default=0)
    article_audit.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    article_audit.add_argument("--log-level", default="INFO")

    article_train = subparsers.add_parser(
        "article-train",
        help="Train a native Persian Article Graph-LM checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    article_train.add_argument(
        "--corpus",
        required=True,
        help="Prepared article corpus.txt",
    )
    article_train.add_argument("--output-dir", default="runs/article-llm")
    article_train.add_argument(
        "--graph-encoder",
        choices=["none", "gcn", "graphsage", "gat", "rgcn"],
        default="gat",
    )
    article_train.add_argument(
        "--fusion",
        choices=["gated", "context_gated", "add"],
        default="context_gated",
    )
    article_train.add_argument(
        "--fusion-layers",
        choices=["input", "all"],
        default="input",
    )
    article_train.add_argument("--fusion-levels", default="token")
    article_train.add_argument("--graph-fusion-scale", type=float, default=1.0)
    article_train.add_argument("--graph-fusion-dropout", type=float, default=0.0)
    article_train.add_argument(
        "--task-losses",
        default=(
            "next_token,masked_token,edge,neighbor,node_relation,"
            "graph_text,sentence_graph"
        ),
    )
    article_train.add_argument("--next-token-weight", type=float, default=1.0)
    article_train.add_argument("--masked-token-weight", type=float, default=0.25)
    article_train.add_argument("--edge-prediction-weight", type=float, default=0.1)
    article_train.add_argument("--neighbor-prediction-weight", type=float, default=0.1)
    article_train.add_argument("--node-relation-weight", type=float, default=0.1)
    article_train.add_argument("--graph-text-alignment-weight", type=float, default=0.1)
    article_train.add_argument(
        "--sentence-graph-alignment-weight",
        type=float,
        default=0.1,
    )
    article_train.add_argument("--mask-probability", type=float, default=0.15)
    article_train.add_argument("--negative-samples", type=int, default=1)
    article_train.add_argument("--no-text-augmentation", action="store_true")
    article_train.add_argument("--augmentation-ratio", type=float, default=0.5)
    article_train.add_argument("--token-dropout", type=float, default=0.05)
    article_train.add_argument("--punctuation-dropout", type=float, default=0.5)
    article_train.add_argument("--node-dropout", type=float, default=0.05)
    article_train.add_argument("--edge-dropout", type=float, default=0.1)
    article_train.add_argument("--subgraph-sampling-ratio", type=float, default=0.9)
    article_train.add_argument("--contrastive-weight", type=float, default=0.05)
    article_train.add_argument("--no-curriculum", action="store_true")
    article_train.add_argument("--early-stopping-patience", type=int, default=3)
    article_train.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    article_train.add_argument(
        "--checkpoint-metric",
        choices=["next_token", "total"],
        default="next_token",
    )
    article_train.add_argument("--max-grad-norm", type=float, default=1.0)
    article_train.add_argument("--d-model", type=int, default=128)
    article_train.add_argument("--n-heads", type=int, default=4)
    article_train.add_argument("--n-layers", type=int, default=2)
    article_train.add_argument("--dim-feedforward", type=int, default=512)
    article_train.add_argument("--dropout", type=float, default=0.1)
    article_train.add_argument("--graph-hidden-dim", type=int, default=128)
    article_train.add_argument("--graph-heads", type=int, default=4)
    article_train.add_argument(
        "--graph-relation-mode",
        choices=["bias", "embedding", "rgcn"],
        default="embedding",
    )
    article_train.add_argument(
        "--graph-pooling",
        choices=["none", "mean", "attention"],
        default="none",
    )
    article_train.add_argument("--graph-node-importance", action="store_true")
    article_train.add_argument(
        "--no-graph-node-type-embedding",
        dest="graph_node_type_embedding",
        action="store_false",
    )
    article_train.add_argument("--epochs", type=int, default=3)
    article_train.add_argument("--batch-size", type=int, default=8)
    article_train.add_argument("--learning-rate", type=float, default=3e-4)
    article_train.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "linear"],
        default="cosine",
    )
    article_train.add_argument("--warmup-ratio", type=float, default=0.05)
    article_train.add_argument("--weight-decay", type=float, default=0.01)
    article_train.add_argument("--validation-ratio", type=float, default=0.1)
    article_train.add_argument("--block-size", type=int, default=128)
    article_train.add_argument("--stride", type=int, default=None)
    article_train.add_argument("--min-freq", type=int, default=1)
    article_train.add_argument("--max-vocab-size", type=int, default=None)
    article_train.add_argument("--graph-window-size", type=int, default=4)
    article_train.add_argument("--graph-min-count", type=int, default=1)
    article_train.add_argument(
        "--graph-weighting",
        choices=["distance", "count", "raw", "pmi", "ppmi"],
        default="distance",
    )
    article_train.add_argument("--graph-min-edge-weight", type=float, default=0.0)
    article_train.add_argument("--graph-top-k", type=int, default=None)
    article_train.add_argument("--graph-directed", action="store_true")
    article_train.add_argument(
        "--graph-scope",
        choices=["corpus", "document", "sentence"],
        default="document",
    )
    article_train.add_argument(
        "--context-node-type",
        choices=["none", "document", "sentence"],
        default="none",
    )
    article_train.add_argument(
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
    )
    article_train.add_argument("--relation-weights", default=None)
    article_train.add_argument(
        "--semantic-similarity-threshold",
        type=float,
        default=0.6,
    )
    article_train.add_argument("--semantic-top-k", type=int, default=4)
    article_train.add_argument(
        "--semantic-method",
        choices=["distributional", "orthographic"],
        default="distributional",
    )
    article_train.add_argument(
        "--linguistic-backend",
        choices=["auto", "stanza", "heuristic"],
        default="auto",
    )
    article_train.add_argument("--topic-top-k", type=int, default=8)
    article_train.add_argument("--dynamic-graph", action="store_true")
    article_train.add_argument("--graph-build-batch-size", type=int, default=None)
    article_train.add_argument("--graph-cache-dir", default=None)
    article_train.add_argument("--no-reuse-graph-cache", action="store_true")
    article_train.add_argument("--dataloader-num-workers", type=int, default=0)
    article_train.add_argument("--dataloader-pin-memory", action="store_true")
    article_train.add_argument("--amp", action="store_true")
    article_train.add_argument("--resume-from", default=None)
    article_train.add_argument(
        "--tokenizer-type",
        choices=["word", "subword", "char_chunk", "bpe", "unigram"],
        default="unigram",
    )
    article_train.add_argument(
        "--tokenizer-half-space",
        choices=["preserve", "split"],
        default="preserve",
    )
    article_train.add_argument("--tokenizer-morph-splitting", action="store_true")
    article_train.add_argument(
        "--tokenizer-compound-verb-mode",
        choices=["none", "join"],
        default="none",
    )
    article_train.add_argument("--tokenizer-bpe-merges", type=int, default=200)
    article_train.add_argument("--unigram-num-pieces", type=int, default=8000)
    article_train.add_argument("--seed", type=int, default=0)
    article_train.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    article_train.add_argument("--log-level", default="INFO")

    article_ablation = subparsers.add_parser(
        "article-ablation",
        help="Run native graph ablations for Article Graph-LM training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    article_ablation.add_argument("--corpus", required=True)
    article_ablation.add_argument("--output-dir", default="runs/article-ablation")
    article_ablation.add_argument(
        "--graph-encoders",
        nargs="+",
        choices=["none", "gcn", "graphsage", "gat", "rgcn"],
        default=["none", "gat", "rgcn"],
    )
    article_ablation.add_argument(
        "--graph-scopes",
        nargs="+",
        choices=["document", "sentence", "corpus"],
        default=["document", "sentence"],
    )
    article_ablation.add_argument(
        "--relation-groups",
        default=None,
        help="Semicolon-separated groups, for example name=cooccurrence,pmi",
    )
    article_ablation.add_argument(
        "--graph-relations",
        nargs="+",
        default=list(DEFAULT_ARTICLE_GRAPH_RELATIONS),
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
    )
    article_ablation.add_argument(
        "--fusion",
        choices=["gated", "context_gated", "add"],
        default="context_gated",
    )
    article_ablation.add_argument(
        "--fusion-layers",
        choices=["input", "all"],
        default="input",
    )
    article_ablation.add_argument("--fusion-levels", default="token")
    article_ablation.add_argument(
        "--task-losses",
        default="next_token,masked_token,edge,neighbor,node_relation,graph_text",
    )
    article_ablation.add_argument("--no-text-augmentation", action="store_true")
    article_ablation.add_argument("--contrastive-weight", type=float, default=0.05)
    article_ablation.add_argument("--edge-dropout", type=float, default=0.1)
    article_ablation.add_argument("--node-dropout", type=float, default=0.05)
    article_ablation.add_argument("--subgraph-sampling-ratio", type=float, default=0.9)
    article_ablation.add_argument("--epochs", type=int, default=3)
    article_ablation.add_argument("--batch-size", type=int, default=8)
    article_ablation.add_argument("--learning-rate", type=float, default=3e-4)
    article_ablation.add_argument("--weight-decay", type=float, default=0.01)
    article_ablation.add_argument("--validation-ratio", type=float, default=0.1)
    article_ablation.add_argument("--block-size", type=int, default=128)
    article_ablation.add_argument("--d-model", type=int, default=128)
    article_ablation.add_argument("--n-heads", type=int, default=4)
    article_ablation.add_argument("--n-layers", type=int, default=2)
    article_ablation.add_argument("--dim-feedforward", type=int, default=512)
    article_ablation.add_argument("--dropout", type=float, default=0.1)
    article_ablation.add_argument("--graph-hidden-dim", type=int, default=128)
    article_ablation.add_argument("--graph-heads", type=int, default=4)
    article_ablation.add_argument("--graph-window-size", type=int, default=4)
    article_ablation.add_argument("--graph-min-count", type=int, default=1)
    article_ablation.add_argument(
        "--graph-weighting",
        choices=["distance", "count", "raw", "pmi", "ppmi"],
        default="distance",
    )
    article_ablation.add_argument(
        "--graph-relation-mode",
        choices=["bias", "embedding", "rgcn"],
        default="embedding",
    )
    article_ablation.add_argument(
        "--tokenizer-type",
        choices=["word", "subword", "char_chunk", "bpe", "unigram"],
        default="unigram",
    )
    article_ablation.add_argument("--tokenizer-bpe-merges", type=int, default=200)
    article_ablation.add_argument("--unigram-num-pieces", type=int, default=8000)
    article_ablation.add_argument("--probe-topic", default=None)
    article_ablation.add_argument("--probe-sections", type=int, default=2)
    article_ablation.add_argument("--probe-max-new-tokens", type=int, default=24)
    article_ablation.add_argument("--seed", type=int, default=0)
    article_ablation.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    article_ablation.add_argument("--log-level", default="INFO")

    article_generate = subparsers.add_parser(
        "article-generate",
        help="Generate a structured Persian article from an Article Graph-LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    article_generate.add_argument("--model", required=True)
    article_generate.add_argument("--topic", required=True)
    article_generate.add_argument(
        "--audience", default=DEFAULT_ARTICLE_GENERATION_AUDIENCE
    )
    article_generate.add_argument("--tone", default=DEFAULT_ARTICLE_GENERATION_TONE)
    article_generate.add_argument(
        "--sections", type=int, default=DEFAULT_ARTICLE_GENERATION_SECTIONS
    )
    article_generate.add_argument(
        "--min-new-tokens",
        type=int,
        default=DEFAULT_ARTICLE_GENERATION_MIN_NEW_TOKENS,
    )
    article_generate.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_ARTICLE_GENERATION_MAX_NEW_TOKENS,
    )
    article_generate.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_ARTICLE_GENERATION_TEMPERATURE,
    )
    article_generate.add_argument(
        "--top-k", type=int, default=DEFAULT_ARTICLE_GENERATION_TOP_K
    )
    article_generate.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_ARTICLE_GENERATION_REPETITION_PENALTY,
    )
    article_generate.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
    )
    article_generate.add_argument("--output-path", default=None)
    article_generate.add_argument(
        "--graph-memory",
        choices=["on", "off"],
        default="on" if DEFAULT_ARTICLE_GENERATION_GRAPH_MEMORY else "off",
    )
    article_generate.add_argument("--graph-memory-top-k", type=int, default=32)
    article_generate.add_argument("--graph-memory-depth", type=int, default=1)
    article_generate.add_argument("--graph-memory-max-edges", type=int, default=256)
    article_generate.add_argument("--graph-memory-min-score", type=float, default=0.0)
    article_generate.add_argument("--graph-memory-relation-weights", default=None)
    article_generate.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    article_generate.add_argument("--log-level", default="INFO")


def run_article_command(args: argparse.Namespace, logger: Any) -> int:
    if args.command == "article-prepare":
        manifest = _run_article_prepare(args)
        logger.info(
            "article_corpus records=%d rejected=%d manifest=%s",
            manifest["accepted_records"],
            manifest["rejected_records"],
            manifest["manifest_path"],
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "article-audit":
        report = _run_article_audit(args)
        logger.info(
            "article_audit records=%d report=%s",
            report["accepted_records"],
            report["report_path"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "article-train":
        metrics = _run_article_train(args)
        logger.info(
            "article_llm checkpoint=%s perplexity=%.3f config=%s",
            metrics["checkpoint_dir"],
            metrics["best_perplexity"],
            metrics["article_llm_config_path"],
        )
        return 0
    if args.command == "article-ablation":
        report = _run_article_ablation(args)
        logger.info(
            "article_ablation variants=%d report=%s",
            len(report["variants"]),
            report["report_path"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "article-generate":
        print(_run_article_generate(args))
        return 0
    raise ValueError(f"unsupported article command: {args.command}")

"""Stable public API for Rakhshai Graph-based NLP.

This module is the supported import surface for application code.  It collects
the project's graph builders, Persian text utilities, task pipelines, neural
models and Graph-LM components behind a lazy facade so importing the package
does not eagerly load every optional path.

The compatibility promise is intentionally simple: names exported here are part
of the stable API for the current major release. New names may be added in minor
releases, while removals or breaking signature changes should wait for the next
major release.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

API_STATUS = "stable"
API_VERSION = "2.1"
STABLE_API_VERSION = API_VERSION
__api_version__ = API_VERSION

_STABLE_EXPORTS: dict[str, tuple[str, str]] = {
    # Graph data structures and graph builders
    "Graph": ("rakhshai_graph_nlp.graphs.graph", "Graph"),
    "build_cooccurrence_graph": (
        "rakhshai_graph_nlp.graphs.co_occurrence",
        "build_cooccurrence_graph",
    ),
    "build_document_graph": (
        "rakhshai_graph_nlp.graphs.document",
        "build_document_graph",
    ),
    "build_dependency_graph": (
        "rakhshai_graph_nlp.graphs.dependency",
        "build_dependency_graph",
    ),
    "build_semantic_graph": (
        "rakhshai_graph_nlp.graphs.semantic",
        "build_semantic_graph",
    ),
    "build_semantic_graph_from_farsnet": (
        "rakhshai_graph_nlp.graphs.semantic",
        "build_semantic_graph_from_farsnet",
    ),
    "build_text_graph": (
        "rakhshai_graph_nlp.graphs.text_graph",
        "build_text_graph",
    ),
    "load_farsnet_relations": (
        "rakhshai_graph_nlp.graphs.semantic",
        "load_farsnet_relations",
    ),
    "to_undirected_coo": (
        "rakhshai_graph_nlp.graphs.sparse",
        "to_undirected_coo",
    ),
    "add_self_loops": (
        "rakhshai_graph_nlp.graphs.sparse",
        "add_self_loops",
    ),
    "row_normalize_csr": (
        "rakhshai_graph_nlp.graphs.sparse",
        "row_normalize_csr",
    ),
    # Text preprocessing and feature extraction
    "tokenize": ("rakhshai_graph_nlp.features.tokenize", "tokenize"),
    "tokenize_persian": ("rakhshai_graph_nlp.features.tokenizer", "tokenize"),
    "split_sentences": (
        "rakhshai_graph_nlp.features.tokenize",
        "split_sentences",
    ),
    "PersianNormalizer": (
        "rakhshai_graph_nlp.features.preprocessing",
        "PersianNormalizer",
    ),
    "PersianNormalizerConfig": (
        "rakhshai_graph_nlp.features.preprocessing",
        "PersianNormalizerConfig",
    ),
    "normalize_persian_text": (
        "rakhshai_graph_nlp.features.preprocessing",
        "normalize_persian_text",
    ),
    "normalise_characters": (
        "rakhshai_graph_nlp.features.preprocessing",
        "normalise_characters",
    ),
    "remove_diacritics": (
        "rakhshai_graph_nlp.features.preprocessing",
        "remove_diacritics",
    ),
    "normalise_whitespace": (
        "rakhshai_graph_nlp.features.preprocessing",
        "normalise_whitespace",
    ),
    "preprocess": ("rakhshai_graph_nlp.features.preprocessing", "preprocess"),
    "preprocess_persian_corpus": (
        "rakhshai_graph_nlp.features.pyg_data",
        "preprocess_persian_corpus",
    ),
    "build_feature_matrix": (
        "rakhshai_graph_nlp.features.pyg_data",
        "build_feature_matrix",
    ),
    "graph_to_data": ("rakhshai_graph_nlp.features.pyg_data", "graph_to_data"),
    "cooccurrence_matrix": (
        "rakhshai_graph_nlp.features.text_graph",
        "cooccurrence_matrix",
    ),
    "load_dummy_classification_dataset": (
        "rakhshai_graph_nlp.features.datasets",
        "load_dummy_classification_dataset",
    ),
    # Graph neural models
    "GCNClassifier": ("rakhshai_graph_nlp.models.gcn", "GCNClassifier"),
    "GATClassifier": ("rakhshai_graph_nlp.models.gat", "GATClassifier"),
    "GATLayer": ("rakhshai_graph_nlp.models.gat", "GATLayer"),
    "GraphSAGE": ("rakhshai_graph_nlp.models.graphsage", "GraphSAGE"),
    "GraphSAGEClassifier": (
        "rakhshai_graph_nlp.models.graphsage",
        "GraphSAGEClassifier",
    ),
    "GraphSAGELayer": (
        "rakhshai_graph_nlp.models.graphsage",
        "GraphSAGELayer",
    ),
    # Task-level APIs
    "TextGraphClassifier": (
        "rakhshai_graph_nlp.tasks.classification",
        "TextGraphClassifier",
    ),
    "TextGraphClassifierConfig": (
        "rakhshai_graph_nlp.tasks.classification",
        "TextGraphClassifierConfig",
    ),
    "train_node_classifier": (
        "rakhshai_graph_nlp.tasks.classification",
        "train_node_classifier",
    ),
    "train_gcn_classifier": (
        "rakhshai_graph_nlp.tasks.classification",
        "train_gcn_classifier",
    ),
    "textrank_summarise": (
        "rakhshai_graph_nlp.tasks.summarization",
        "textrank_summarise",
    ),
    "gat_summarise": (
        "rakhshai_graph_nlp.tasks.summarization",
        "gat_summarise",
    ),
    "GATSummarizer": (
        "rakhshai_graph_nlp.tasks.summarization",
        "GATSummarizer",
    ),
    "recommend_similar": (
        "rakhshai_graph_nlp.tasks.recommendation",
        "recommend_similar",
    ),
    "contains_hate_speech": (
        "rakhshai_graph_nlp.tasks.hate_speech",
        "contains_hate_speech",
    ),
    "HateSpeechDetector": (
        "rakhshai_graph_nlp.tasks.hate_speech",
        "HateSpeechDetector",
    ),
    "compute_social_embeddings": (
        "rakhshai_graph_nlp.tasks.social_analysis",
        "compute_social_embeddings",
    ),
    # Metrics
    "accuracy": ("rakhshai_graph_nlp.metrics", "accuracy"),
    "macro_f1": ("rakhshai_graph_nlp.metrics", "macro_f1"),
    "confusion_matrix": ("rakhshai_graph_nlp.metrics", "confusion_matrix"),
    # Graph-LM, training and generation
    "PersianTokenizer": ("rakhshai_graph_nlp.lm.tokenizer", "PersianTokenizer"),
    "LMDataset": ("rakhshai_graph_nlp.lm.dataset", "LMDataset"),
    "LMLoaders": ("rakhshai_graph_nlp.lm.dataset", "LMLoaders"),
    "build_lm_dataloaders": (
        "rakhshai_graph_nlp.lm.dataset",
        "build_lm_dataloaders",
    ),
    "GraphLMGraph": ("rakhshai_graph_nlp.lm.graph_builder", "GraphLMGraph"),
    "build_graph_lm_graph": (
        "rakhshai_graph_nlp.lm.graph_builder",
        "build_graph_lm_graph",
    ),
    "build_graph_lm_graph_from_token_ids": (
        "rakhshai_graph_nlp.lm.graph_builder",
        "build_graph_lm_graph_from_token_ids",
    ),
    "GraphMemoryArtifact": (
        "rakhshai_graph_nlp.lm.graph_memory",
        "GraphMemoryArtifact",
    ),
    "GraphMemoryConfig": (
        "rakhshai_graph_nlp.lm.graph_memory",
        "GraphMemoryConfig",
    ),
    "RetrievedGraphContext": (
        "rakhshai_graph_nlp.lm.graph_memory",
        "RetrievedGraphContext",
    ),
    "GraphLMConfig": ("rakhshai_graph_nlp.lm.model", "GraphLMConfig"),
    "GenerationConfig": ("rakhshai_graph_nlp.lm.model", "GenerationConfig"),
    "GraphCausalLM": ("rakhshai_graph_nlp.lm.model", "GraphCausalLM"),
    "RakhshaiGraphEncoder": (
        "rakhshai_graph_nlp.lm.model",
        "RakhshaiGraphEncoder",
    ),
    "GraphTokenFusion": ("rakhshai_graph_nlp.lm.model", "GraphTokenFusion"),
    "perplexity": ("rakhshai_graph_nlp.lm.model", "perplexity"),
    "MultiTaskLossConfig": (
        "rakhshai_graph_nlp.lm.multitask",
        "MultiTaskLossConfig",
    ),
    "parse_task_losses": (
        "rakhshai_graph_nlp.lm.multitask",
        "parse_task_losses",
    ),
    "LMTrainer": ("rakhshai_graph_nlp.lm.trainer", "LMTrainer"),
    "LMTrainingConfig": ("rakhshai_graph_nlp.lm.trainer", "LMTrainingConfig"),
    "train_graph_lm": ("rakhshai_graph_nlp.lm.trainer", "train_graph_lm"),
    "TextAugmentationConfig": (
        "rakhshai_graph_nlp.lm.augmentation",
        "TextAugmentationConfig",
    ),
    "augment_text": ("rakhshai_graph_nlp.lm.augmentation", "augment_text"),
    "augment_corpus": ("rakhshai_graph_nlp.lm.augmentation", "augment_corpus"),
    "augment_graph_data": (
        "rakhshai_graph_nlp.lm.augmentation",
        "augment_graph_data",
    ),
    "mean_pool_hidden": (
        "rakhshai_graph_nlp.lm.augmentation",
        "mean_pool_hidden",
    ),
    "PoemRecommender": (
        "rakhshai_graph_nlp.lm.poem_recommender",
        "PoemRecommender",
    ),
    "build_poem_index": (
        "rakhshai_graph_nlp.lm.poem_recommender",
        "build_poem_index",
    ),
    "embed_texts": ("rakhshai_graph_nlp.lm.poem_recommender", "embed_texts"),
    "load_for_embedding": (
        "rakhshai_graph_nlp.lm.poem_recommender",
        "load_for_embedding",
    ),
    "list_poem_recommenders": (
        "rakhshai_graph_nlp.lm.poem_recommender",
        "list_poem_recommenders",
    ),
}

__all__ = [
    "API_STATUS",
    "API_VERSION",
    "STABLE_API_VERSION",
    "__api_version__",
    "stable_api",
    *_STABLE_EXPORTS,
]


def stable_api() -> tuple[str, ...]:
    """Return the names covered by the stable API compatibility contract."""

    return tuple(_STABLE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _STABLE_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _STABLE_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

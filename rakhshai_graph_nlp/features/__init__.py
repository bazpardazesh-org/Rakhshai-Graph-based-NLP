"""Text preprocessing and feature extraction utilities."""

from .datasets import load_dummy_classification_dataset
from .preprocessing import (
    PersianNormalizer,
    PersianNormalizerConfig,
    normalize_persian_text,
    normalise_characters,
    normalise_whitespace,
    preprocess,
    remove_diacritics,
)
from .pyg_data import build_feature_matrix, graph_to_data, preprocess_persian_corpus
from .text_graph import cooccurrence_matrix
from .tokenize import split_sentences, tokenize
from .tokenizer import tokenize as tokenize_persian

__all__ = [
    "tokenize",
    "tokenize_persian",
    "split_sentences",
    "PersianNormalizer",
    "PersianNormalizerConfig",
    "normalize_persian_text",
    "normalise_characters",
    "remove_diacritics",
    "normalise_whitespace",
    "preprocess",
    "preprocess_persian_corpus",
    "build_feature_matrix",
    "graph_to_data",
    "cooccurrence_matrix",
    "load_dummy_classification_dataset",
]

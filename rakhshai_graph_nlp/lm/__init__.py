"""Graph language modelling components for Persian text."""

from .augmentation import (
    TextAugmentationConfig,
    augment_corpus,
    augment_graph_data,
    augment_text,
    mean_pool_hidden,
)
from .dataset import LMDataset, LMLoaders, build_lm_dataloaders
from .graph_builder import (
    GraphLMGraph,
    build_graph_lm_graph,
    build_graph_lm_graph_from_token_ids,
)
from .graph_memory import GraphMemoryArtifact, GraphMemoryConfig, RetrievedGraphContext
from .model import (
    GenerationConfig,
    GraphCausalLM,
    GraphLMConfig,
    GraphTokenFusion,
    RakhshaiGraphEncoder,
    perplexity,
)
from .multitask import MultiTaskLossConfig, parse_task_losses
from .poem_recommender import (
    PoemRecommender,
    build_poem_index,
    embed_texts,
    list_poem_recommenders,
    load_for_embedding,
)
from .tokenizer import PersianTokenizer
from .trainer import LMTrainer, LMTrainingConfig, train_graph_lm

__all__ = [
    "PersianTokenizer",
    "LMDataset",
    "LMLoaders",
    "build_lm_dataloaders",
    "GraphLMGraph",
    "build_graph_lm_graph",
    "build_graph_lm_graph_from_token_ids",
    "GraphMemoryArtifact",
    "GraphMemoryConfig",
    "RetrievedGraphContext",
    "GraphCausalLM",
    "GraphLMConfig",
    "GenerationConfig",
    "RakhshaiGraphEncoder",
    "GraphTokenFusion",
    "perplexity",
    "MultiTaskLossConfig",
    "parse_task_losses",
    "LMTrainer",
    "LMTrainingConfig",
    "train_graph_lm",
    "TextAugmentationConfig",
    "augment_text",
    "augment_corpus",
    "augment_graph_data",
    "mean_pool_hidden",
    "PoemRecommender",
    "build_poem_index",
    "embed_texts",
    "load_for_embedding",
    "list_poem_recommenders",
]

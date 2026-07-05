"""Graph language modelling components for Persian text."""

from .augmentation import (
    TextAugmentationConfig,
    augment_corpus,
    augment_graph_data,
    augment_text,
    mean_pool_hidden,
)
from .corpus import CorpusBuildConfig, build_lm_corpus, persian_ratio
from .dataset import LMDataset, LMLoaders, build_lm_dataloaders
from .distributed import (
    DistributedTrainingInfo,
    get_distributed_info,
    maybe_wrap_distributed,
)
from .eval import (
    NativeEvalConfig,
    evaluate_lm_checkpoint,
    export_human_review,
    score_prompt_completion,
    score_texts,
)
from .graph_builder import (
    GraphLMGraph,
    build_graph_lm_graph,
    build_graph_lm_graph_from_token_ids,
)
from .graph_memory import GraphMemoryArtifact, GraphMemoryConfig, RetrievedGraphContext
from .graph_scaling import (
    LMGraphAblationConfig,
    run_lm_graph_ablation,
    write_graph_feature_store,
)
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
from .profiles import (
    CONTEXT_PRESETS,
    MODEL_PROFILES,
    ModelProfile,
    available_model_profiles,
    build_graph_lm_config_from_profile,
)
from .registry import (
    RunRegistryConfig,
    build_run_report,
    hash_file,
    hash_json,
    write_run_registry,
)
from .sft import SFTConfig, format_sft_record, load_sft_texts, train_sft
from .shards import TokenShardConfig, TokenShardDataset, tokenizer_audit, write_token_shards
from .tokenizer import PersianTokenizer
from .trainer import (
    LMTrainer,
    LMTrainingConfig,
    train_graph_lm,
    train_graph_lm_from_token_shards,
)

__all__ = [
    "PersianTokenizer",
    "CorpusBuildConfig",
    "build_lm_corpus",
    "persian_ratio",
    "LMDataset",
    "LMLoaders",
    "build_lm_dataloaders",
    "DistributedTrainingInfo",
    "get_distributed_info",
    "maybe_wrap_distributed",
    "NativeEvalConfig",
    "evaluate_lm_checkpoint",
    "export_human_review",
    "score_prompt_completion",
    "score_texts",
    "GraphLMGraph",
    "build_graph_lm_graph",
    "build_graph_lm_graph_from_token_ids",
    "GraphMemoryArtifact",
    "GraphMemoryConfig",
    "RetrievedGraphContext",
    "LMGraphAblationConfig",
    "run_lm_graph_ablation",
    "write_graph_feature_store",
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
    "train_graph_lm_from_token_shards",
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
    "ModelProfile",
    "MODEL_PROFILES",
    "CONTEXT_PRESETS",
    "available_model_profiles",
    "build_graph_lm_config_from_profile",
    "RunRegistryConfig",
    "build_run_report",
    "hash_file",
    "hash_json",
    "write_run_registry",
    "SFTConfig",
    "format_sft_record",
    "load_sft_texts",
    "train_sft",
    "TokenShardConfig",
    "TokenShardDataset",
    "tokenizer_audit",
    "write_token_shards",
]

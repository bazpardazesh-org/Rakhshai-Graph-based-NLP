# Stable Public API

Rakhshai Graph-based NLP exposes a stable Python API starting with package
version `2.2.0` and API version `2.2`.

For step-by-step examples, see the [Python API Usage Guide](api_usage.md).

Prefer importing supported names from either of these two surfaces:

```python
from rakhshai_graph_nlp import TextGraphClassifier, build_text_graph
from rakhshai_graph_nlp.api import GraphCausalLM, PersianTokenizer
```

Both surfaces point to the same compatibility contract. The package exposes:

```python
import rakhshai_graph_nlp as rgnn

print(rgnn.API_STATUS)       # stable
print(rgnn.__api_version__)  # 2.2
print(rgnn.stable_api())     # names covered by the compatibility contract
```

## Compatibility Contract

- Names exported by `rakhshai_graph_nlp.api.stable_api()` are stable for the
  current major release.
- Minor releases may add new names without breaking old imports.
- Breaking removals or incompatible signature changes should be reserved for a
  future major release.
- Optional integrations still require their optional dependencies at call time:
  `scikit-learn` for TF-IDF document graphs and recommendation, `stanza` for
  dependency parsing, and `scipy` for sparse helpers.

## Core Graph API

| Name | Purpose |
| --- | --- |
| `Graph` | Dense graph container with adjacency, node metadata and helper methods. |
| `build_text_graph` | TextGCN-style word-document graph with PMI and TF-IDF edges. |
| `build_cooccurrence_graph` | Sliding-window word co-occurrence graph. |
| `build_document_graph` | Document similarity graph from TF-IDF or embeddings. |
| `build_dependency_graph` | Persian dependency graph through Stanza. |
| `build_semantic_graph` | Semantic graph from explicit relations or embeddings. |
| `build_semantic_graph_from_farsnet` | Semantic graph from FarsNet-style JSON/CSV/TSV exports. |
| `load_farsnet_relations` | Load FarsNet-style lexical relations. |
| `to_undirected_coo`, `add_self_loops`, `row_normalize_csr` | Sparse graph helpers. |

## Text And Feature API

| Name | Purpose |
| --- | --- |
| `tokenize`, `split_sentences` | Lightweight Unicode-aware tokenization and sentence splitting. |
| `tokenize_persian` | Persian-normalizing feature tokenizer used by graph pipelines. |
| `PersianNormalizer`, `PersianNormalizerConfig` | Reusable Persian normalization with half-space, digit, hamza and ezafe controls. |
| `normalise_characters`, `remove_diacritics`, `normalise_whitespace` | Lower-level Persian text normalization helpers. |
| `normalize_persian_text`, `preprocess` | Convenience text cleanup helpers. |
| `preprocess_persian_corpus` | Normalize, tokenize and optionally lemmatize a corpus. |
| `build_feature_matrix` | Build bag-of-words or embedding-averaged node features. |
| `graph_to_data` | Convert `Graph` to a PyTorch Geometric `Data` object. |
| `cooccurrence_matrix` | Sparse sliding-window co-occurrence matrix. |
| `load_dummy_classification_dataset` | Small built-in classification dataset for examples and smoke tests. |

## Task API

| Name | Purpose |
| --- | --- |
| `TextGraphClassifier`, `TextGraphClassifierConfig` | End-to-end fit/evaluate/predict/save/load text classification pipeline. |
| `train_node_classifier`, `train_gcn_classifier` | Lower-level GNN node classifier training helpers. |
| `textrank_summarise`, `gat_summarise`, `GATSummarizer` | Extractive summarization APIs. |
| `recommend_similar` | Content recommendation over document similarity graphs. |
| `contains_hate_speech`, `HateSpeechDetector` | Rule-based and trainable hate-speech detection helpers. |
| `compute_social_embeddings` | GraphSAGE embeddings for social/network analysis. |

## Model API

| Name | Purpose |
| --- | --- |
| `GCNClassifier` | PyTorch Geometric GCN node classifier. |
| `GraphSAGEClassifier` | PyTorch Geometric GraphSAGE node classifier. |
| `GATClassifier` | PyTorch Geometric GAT node classifier. |
| `GraphSAGE`, `GraphSAGELayer`, `GATLayer` | Lightweight NumPy-oriented graph layers used by analysis utilities. |

## Graph-LM API

| Name | Purpose |
| --- | --- |
| `PersianTokenizer` | Numeric tokenizer for Persian Graph-LM training and generation. |
| `CorpusBuildConfig`, `build_lm_corpus`, `persian_ratio` | Native corpus cleaning, filtering, split creation and Persian-character quality checks. |
| `LMDataset`, `LMLoaders`, `build_lm_dataloaders` | Next-token language-model datasets and loaders. |
| `TokenShardConfig`, `TokenShardDataset`, `write_token_shards`, `tokenizer_audit` | Memory-mapped token-shard preparation and lazy shard datasets for larger LM runs. |
| `GraphLMGraph`, `build_graph_lm_graph`, `build_graph_lm_graph_from_token_ids` | Multi-relation Graph-LM graph construction. |
| `GraphCausalLM`, `GraphLMConfig`, `GenerationConfig` | Graph-fused Persian causal LM and generation configuration. |
| `RakhshaiGraphEncoder`, `GraphTokenFusion` | Graph reasoning and graph-token fusion components. |
| `LMTrainer`, `LMTrainingConfig`, `train_graph_lm`, `train_graph_lm_from_token_shards` | Graph-LM training loop, config and high-level train functions for text corpora or token shards. |
| `DistributedTrainingInfo`, `get_distributed_info`, `maybe_wrap_distributed` | PyTorch-native distributed training inspection and optional DDP/FSDP wrapping. |
| `NativeEvalConfig`, `evaluate_lm_checkpoint`, `score_texts`, `score_prompt_completion`, `export_human_review` | Local-only checkpoint evaluation, prompt scoring and human-review export helpers. |
| `GraphMemoryArtifact`, `GraphMemoryConfig`, `RetrievedGraphContext` | Prompt-aware graph memory for generation. |
| `LMGraphAblationConfig`, `run_lm_graph_ablation`, `write_graph_feature_store` | Engine-level graph ablation and scalable graph-feature artifact helpers. |
| `ModelProfile`, `MODEL_PROFILES`, `CONTEXT_PRESETS`, `available_model_profiles`, `build_graph_lm_config_from_profile` | Named native model-size profiles and context presets for repeatable LM training configs. |
| `RunRegistryConfig`, `write_run_registry`, `build_run_report`, `hash_file`, `hash_json` | Run provenance, data/checkpoint hashing and consolidated run reports. |
| `SFTConfig`, `format_sft_record`, `load_sft_texts`, `train_sft` | Supervised fine-tuning helpers for local human-authored prompt/completion data. |
| `MultiTaskLossConfig`, `parse_task_losses` | Multi-task Graph-LM objective configuration. |
| `TextAugmentationConfig`, `augment_text`, `augment_corpus`, `augment_graph_data`, `mean_pool_hidden` | Low-data text and graph regularization utilities plus pooled text embeddings. |
| `PoemRecommender`, `build_poem_index`, `embed_texts`, `load_for_embedding`, `list_poem_recommenders` | Graph-LM powered poem search and recommendation. |
| `perplexity` | Convert mean next-token cross-entropy loss to perplexity. |

## LLM Workflow API

Workflow APIs are higher-level pipelines built on top of the Graph-LM engine.
Use `rakhshai_graph_nlp.llm.article` for article-specific code. The same public
names are re-exported from `rakhshai_graph_nlp` for stable-API compatibility.

| Name | Purpose |
| --- | --- |
| `ArticleCorpusConfig`, `prepare_article_corpus` | Convert raw TXT, JSONL, CSV or TSV Persian article data into prepared corpus, split and manifest files. |
| `ArticleAuditConfig`, `audit_article_corpus` | Audit native Persian article data quality, duplicate risk, Persian surface statistics and tokenizer behavior without external models. |
| `ArticleTrainingConfig`, `train_article_llm` | Train an article-focused Graph-LM checkpoint while preserving normal Graph-LM checkpoint artifacts. |
| `ArticleAblationConfig`, `run_article_ablation` | Run native no-graph/graph/scope/relation ablations around article training and collect zero-gate and optional graph-memory probe reports. |
| `ArticleGenerationConfig`, `generate_persian_article` | Load an article checkpoint and generate a structured Persian article. |
| `PersianArticle` | Structured output container with title, introduction, sections, conclusion, Markdown and JSON helpers. |

## Metrics

| Name | Purpose |
| --- | --- |
| `accuracy` | Classification accuracy. |
| `macro_f1` | Macro-averaged F1 score. |
| `confusion_matrix` | Confusion matrix for integer labels. |

## Reference

::: rakhshai_graph_nlp.api

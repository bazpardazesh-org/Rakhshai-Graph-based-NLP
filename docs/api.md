# Stable Public API

Rakhshai Graph-based NLP exposes a stable Python API starting with package
version `2.1.0` and API version `2.1`.

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
print(rgnn.__api_version__)  # 2.1
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
| `normalize_persian_text`, `preprocess` | Convenience text cleanup helpers. |
| `preprocess_persian_corpus` | Normalize, tokenize and optionally lemmatize a corpus. |
| `build_feature_matrix` | Build bag-of-words or embedding-averaged node features. |
| `graph_to_data` | Convert `Graph` to a PyTorch Geometric `Data` object. |
| `cooccurrence_matrix` | Sparse sliding-window co-occurrence matrix. |

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
| `LMDataset`, `build_lm_dataloaders` | Next-token language-model datasets and loaders. |
| `GraphLMGraph`, `build_graph_lm_graph`, `build_graph_lm_graph_from_token_ids` | Multi-relation Graph-LM graph construction. |
| `GraphCausalLM`, `GraphLMConfig`, `GenerationConfig` | Graph-fused Persian causal LM and generation configuration. |
| `RakhshaiGraphEncoder`, `GraphTokenFusion` | Graph reasoning and graph-token fusion components. |
| `LMTrainer`, `LMTrainingConfig`, `train_graph_lm` | Graph-LM training loop, config and high-level train function. |
| `GraphMemoryArtifact`, `GraphMemoryConfig`, `RetrievedGraphContext` | Prompt-aware graph memory for generation. |
| `MultiTaskLossConfig`, `parse_task_losses` | Multi-task Graph-LM objective configuration. |
| `TextAugmentationConfig`, `augment_text`, `augment_corpus`, `augment_graph_data` | Low-data text and graph regularization utilities. |
| `PoemRecommender`, `build_poem_index`, `embed_texts`, `load_for_embedding`, `list_poem_recommenders` | Graph-LM powered poem search and recommendation. |
| `perplexity` | Convert mean next-token cross-entropy loss to perplexity. |

## Metrics

| Name | Purpose |
| --- | --- |
| `accuracy` | Classification accuracy. |
| `macro_f1` | Macro-averaged F1 score. |
| `confusion_matrix` | Confusion matrix for integer labels. |

## Reference

::: rakhshai_graph_nlp.api

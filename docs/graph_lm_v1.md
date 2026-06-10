# Rakhshai Graph-LM v1 Architecture

This document defines the stable baseline architecture for the current
Graph-LM path. It is the reference contract for Phase 1: keep the present core
measurable, reproducible and comparable before adding larger tokenizer, graph
or fusion changes.

## Scope

Rakhshai Graph-LM v1 is a causal Persian language model that can run in two
comparable modes:

- `baseline-none`: a plain Transformer causal LM with no graph encoder.
- `simple-graph`: the same Transformer causal LM augmented with a token
  co-occurrence graph, a GNN graph encoder and token-level fusion.

Both modes use the same tokenizer, dataset construction, language-model loss,
trainer, checkpoint format and generation interface. The graph mode changes
only the graph-related components.

## Component Boundaries

The stable v1 pipeline is:

```text
Persian corpus
-> PersianTokenizer
-> LMDataset
-> GraphLMGraph
-> RakhshaiGraphEncoder
-> GraphTokenFusion
-> GraphCausalLM
-> LMTrainer
-> checkpoint + metrics
```

### Tokenizer

Implementation: `rakhshai_graph_nlp/lm/tokenizer.py`

`PersianTokenizer` normalizes Persian text, builds the vocabulary on the train
split, encodes text to token ids and saves/loads `tokenizer.json`.

Stable v1 expectations:

- Special ids are stable: `<pad>`, `<unk>`, `<bos>`, `<eos>`.
- The tokenizer is fitted on the training corpus only.
- `word` and the current compact `subword` mode are supported.

### Dataset

Implementation: `rakhshai_graph_nlp/lm/dataset.py`

`LMDataset` converts encoded corpus text into fixed-length next-token examples:

```text
input_ids[t] -> target_ids[t + 1]
```

Padding targets are ignored with `-100`, matching PyTorch cross-entropy
conventions.

### Graph Builder

Implementation: `rakhshai_graph_nlp/lm/graph_builder.py`

`build_graph_lm_graph` builds the token graph from the training split. The
current v1 graph is a weighted token co-occurrence graph with optional PMI/PPMI,
directionality, top-k pruning and optional document/sentence context nodes.

Stable v1 simple graph baseline:

```text
graph_encoder = gcn
fusion = gated
graph_weighting = distance
graph_scope = document
context_node_type = none
dynamic_graph = false
```

### Graph Encoder

Implementation: `rakhshai_graph_nlp/lm/model.py`

`RakhshaiGraphEncoder` converts graph node features into graph-aware token
representations. Current supported encoders are:

- `gcn`
- `graphsage`
- `gat`

`graph_encoder = none` disables this component and defines the no-graph
baseline.

### Fusion Layer

Implementation: `rakhshai_graph_nlp/lm/model.py`

`GraphTokenFusion` combines normal token embeddings with graph embeddings.
Current supported fusion modes are:

- `gated`
- `context_gated`
- `add`

For v1 comparisons, `gated` is the stable simple-graph default.

### Language Model

Implementation: `rakhshai_graph_nlp/lm/model.py`

`GraphCausalLM` is a causal Transformer language model. It uses token and
position embeddings, applies optional graph fusion, runs Transformer encoder
layers with a causal mask and predicts the next token with a tied LM head.

### Trainer

Implementation: `rakhshai_graph_nlp/lm/trainer.py`

`LMTrainer` owns the training loop, validation loop, best-checkpoint saving,
loss history and perplexity reporting.

Stable v1 metrics:

- `train_loss`
- `validation_loss`
- `perplexity`
- `best_validation_loss`
- `best_perplexity`

### Generation

Implementation: `GraphCausalLM.generate` and `rgnn-cli generate`

Generation reloads the saved model, tokenizer, generation config and graph
artifact when present. The no-graph baseline must generate without `graph.pt`.

## Checkpoint Contract

Each completed Graph-LM v1 run should write:

```text
model.pt
config.json
tokenizer.json
graph_config.json
generation_config.json
metrics.json
```

Graph-enabled static runs also write:

```text
graph.pt
```

The no-graph baseline should not require `graph.pt`.

## Baselines

### Baseline Without Graph

```bash
rgnn-cli lm-train \
  --corpus data/mini_persian_lm.txt \
  --graph-encoder none \
  --output-dir runs/v1-baseline-none \
  --device cpu \
  --seed 0
```

This is the control run for measuring whether graph components help.

### Simple Graph Baseline

```bash
rgnn-cli lm-train \
  --corpus data/mini_persian_lm.txt \
  --graph-encoder gcn \
  --fusion gated \
  --graph-weighting distance \
  --graph-scope document \
  --context-node-type none \
  --output-dir runs/v1-simple-graph \
  --device cpu \
  --seed 0
```

This is the first graph-enabled comparison point for v1.

## Phase 1 Non-Goals

The following changes belong to later phases and should not be mixed into v1
stabilization unless explicitly scoped:

- Replacing the tokenizer with a full Persian subword tokenizer.
- Adding multi-relation Persian graph layers.
- Redesigning the graph encoder.
- Redesigning graph-text fusion.
- Adding multi-task training losses.
- Adding graph memory during generation.

## Definition of Done

The architecture part of Phase 1 is complete when:

- The current component boundaries are documented.
- The two baseline modes are documented.
- The checkpoint contract is documented.
- Existing Graph-LM and CLI tests pass.

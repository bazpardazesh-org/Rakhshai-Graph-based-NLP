# Rakhshai Graph-LM V2 Architecture

This document defines the stable architecture contract for the Rakhshai V2
Graph-LM path. V2 keeps the measurable baseline from v1, then promotes the
multi-relation graph, relation-aware graph reasoning, adaptive graph-text
fusion, low-data training engine and graph memory into the documented release
surface.

## Scope

Rakhshai Graph-LM V2 is a causal Persian language model that can run in three
comparable modes:

- `baseline-none`: a plain Transformer causal LM with no graph encoder.
- `simple-graph`: the v1-compatible Transformer causal LM augmented with a
  token co-occurrence graph, a GNN graph encoder and token-level fusion.
- `multi-relation-graph`: the V2 default with multiple graph relations,
  relation-aware encoding, adaptive fusion and graph-memory-aware generation.

All modes use the same tokenizer, dataset construction, language-model loss,
trainer, checkpoint format and generation interface. Graph-enabled modes change
only the graph-related components.

## Component Boundaries

The stable V2 pipeline is:

```text
Persian corpus
-> PersianTokenizer
-> LMDataset
-> GraphLMGraph
-> RakhshaiGraphEncoder
-> GraphTokenFusion
-> GraphCausalLM
-> LMTrainer
-> GraphMemory
-> checkpoint + metrics
```

### Tokenizer

Implementation: `rakhshai_graph_nlp/lm/tokenizer.py`

`PersianTokenizer` normalizes Persian text, builds the vocabulary on the train
split, encodes text to token ids and saves/loads `tokenizer.json`.

Stable V2 expectations:

- Special ids are stable: `<pad>` (0), `<unk>` (1), `<bos>` (2), `<eos>` (3) and
  `<mask>` (4). `<mask>` backs the masked-token objective; tokenizers saved
  before it load with `<mask>` mapped onto `<unk>`.
- The tokenizer is fitted on the training corpus only.
- Modes: `word`, `char_chunk` (formerly `subword`), `bpe`, and a genuine
  `unigram` LM tokenizer. `unigram` is the operational default for `lm-train`
  (lowest Persian OOV); its size is set by `--unigram-num-pieces`.
- Punctuation is tokenized separately, Persian numeric separators are
  normalized, and hamza/ezafe folding is configurable. See
  [Persian Tokenizer](persian_tokenizer.md).
- Newly saved tokenizer artifacts use `tokenizer_version = 3`.

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

`build_graph_lm_graph` builds the token graph from the training split. The V2
default graph is a multi-relation graph with co-occurrence, PMI, dependency,
stem, subword, word-document and topic-document relations. The
semantic-similarity relation stays opt-in because it scales with the square of
the graph vocabulary. An edge belonging to several relations is kept as parallel
edges (one `edge_type` per relation) instead of collapsing to a single id.

Stable V2 default graph:

```text
graph_relations = cooccurrence pmi dependency stem subword word_document topic_document
graph_weighting = distance
graph_scope = document
dynamic_graph = false
```

Historical v1 simple graph baseline:

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
- `rgcn`

`graph_encoder = none` disables this component and defines the no-graph
baseline.

V2 can consume relation IDs through graph relation modes `bias`, `embedding` and
`rgcn`; `embedding` (a learned vector per relation) is the operational default
because it best exploits the multi-relational parallel-edge graph. The encoder
can also use node importance and subgraph pooling.

Every graph node receives a learned **node-type embedding** (token, document,
topic, sentence, …) as its initial feature, so non-token nodes no longer start
from all-zero vectors; token nodes add their token embedding on top. Disable it
with `--no-graph-node-type-embedding`. See
[Graph Reasoning Core](graph_reasoning_core.md).

### Fusion Layer

Implementation: `rakhshai_graph_nlp/lm/model.py`

`GraphTokenFusion` combines normal token embeddings with graph embeddings.
Current supported fusion modes are:

- `gated`
- `context_gated`
- `add`

For V2 runs, `context_gated` and adaptive graph-text controls are available for
token, sentence and subgraph-level fusion. For v1 comparisons, `gated` remains
the stable simple-graph default.

#### Zero-Init Gating

The `gated` and `context_gated` modes (and the sentence/subgraph levels) are
residual with a zero-initialised scalar gate:

```
hidden = token_embeddings + tanh(alpha) * (1 - gate) * graph_embeddings
```

`alpha` starts at zero, so an untrained Graph-LM is exactly equivalent to the
no-graph baseline and graph information only enters once training finds it
useful. This replaces the earlier formulation
`gate * token + (1 - gate) * graph`, which substituted part of every token
embedding with a static graph vector and measurably hurt next-token perplexity
on small corpora. The learned gate openness is reported per level in
`metrics.json` fusion stats as `token_alpha_tanh`, `sentence_alpha_tanh` and
`subgraph_alpha_tanh`; values near zero mean the model is effectively ignoring
the graph. Checkpoints saved before this change load with `alpha = 0`, i.e.
their graph contribution is disabled.

### Language Model

Implementation: `rakhshai_graph_nlp/lm/model.py`

`GraphCausalLM` is a decoder-only causal Transformer language model. It embeds
tokens, applies optional graph fusion, runs a stack of pre-norm decoder layers
and predicts the next token with a tied LM head.

The decoder defaults to a modern architecture, each configurable on
`GraphLMConfig`:

- `position_encoding` (default `rope`): rotary position embeddings applied
  inside attention; no learned position table and no fixed length ceiling. Set
  to `learned` for the previous absolute position embeddings.
- `ffn_type` (default `swiglu`): gated SiLU feed-forward. Set to `gelu` for the
  classic two-layer block.
- `norm_type` (default `rmsnorm`): RMSNorm. Set to `layernorm` for LayerNorm.
- `rope_theta` (default `10000.0`): RoPE base frequency.

Causality is enforced inside attention from absolute positions (so it composes
with the KV cache); padding is handled with a key-padding mask.

> **Checkpoint compatibility.** This decoder replaces the previous
> `nn.TransformerEncoderLayer` stack, so checkpoints saved before the change no
> longer match the parameter layout. `from_pretrained` loads them with
> `strict=False` (it will not crash) but the transformer weights are effectively
> reinitialised — retrain to use the new architecture.

### Trainer

Implementation: `rakhshai_graph_nlp/lm/trainer.py`

`LMTrainer` owns the training loop, validation loop, best-checkpoint saving,
loss history and perplexity reporting.

Best-checkpoint selection and early stopping are driven by
`--checkpoint-metric`, which defaults to `next_token`: the saved model is the
best *language model* (lowest next-token loss / perplexity), not the lowest
weighted sum of the auxiliary multi-task terms. Set `--checkpoint-metric total`
to restore selection on the full multi-task `validation_loss`. The masked-token
objective uses the dedicated `<mask>` id (`config.mask_token_id`) rather than
`<unk>`.

Stable V2 metrics:

- `train_loss`
- `validation_loss` (total multi-task loss; selection signal only when
  `--checkpoint-metric total`)
- `validation_next_token_loss` (next-token cross-entropy only; default selection
  and early-stopping signal)
- `perplexity` (computed from `validation_next_token_loss`, not the total
  multi-task loss)
- `best_validation_loss`
- `best_next_token_loss`
- `best_perplexity` (computed from `best_next_token_loss`)
- graph gate statistics when fusion exposes them
- overfitting and early-stopping reports when low-data training options are used

### Generation

Implementation: `GraphCausalLM.generate` and `rgnn-cli generate`

Generation reloads the saved model, tokenizer, generation config and graph
artifact when present. V2 checkpoints can also store graph memory so generation
retrieves prompt-related nodes and subgraphs instead of always passing the full
training graph. The no-graph baseline must generate without `graph.pt`.

With RoPE (the default), a static graph and input-level fusion, generation uses
a **KV cache**: the graph is encoded once per call and each step only encodes
the new token while attending to cached keys/values, giving the same logits as a
full re-encode. Dynamic graphs, learned positions, or per-layer fusion fall back
to re-encoding the sliding-window context each step.

## Checkpoint Contract

Each completed Graph-LM V2 run should write:

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

Graph-memory runs also write the graph memory artifact used by generation.

The no-graph baseline should not require `graph.pt`.

## Baselines

### Baseline Without Graph

```bash
rgnn-cli lm-train \
  --corpus data/mini_persian_lm.txt \
  --graph-encoder none \
  --output-dir runs/v2-baseline-none \
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

This remains the historical simple graph control for comparing V2 against v1.

### V2 Multi-Relation Graph

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --fusion gated \
  --output-dir runs/v2-multi-relation \
  --device cpu \
  --seed 0
```

This is the default graph-enabled V2 comparison point.

## V2 Release Surface

- Package metadata reports `2.2.0`.
- The default Graph-LM graph is multi-relation (includes `dependency`) with
  parallel edges per relation.
- The tokenizer writes `tokenizer_version = 3` and reserves a `<mask>` token.
- The tokenizer can opt into byte fallback for native UTF-8 coverage.
- Independent pretraining can use `lm-build-corpus`, `lm-tokenize`,
  `lm-pretrain`, `lm-ablation`, `lm-eval`, `lm-sft` and `lm-run-report`.
- Trainer scaling controls include gradient accumulation, precision mode,
  activation checkpointing, SDPA backend selection, named model profiles and
  registry/checkpoint manifests.
- Relation-aware graph reasoning and adaptive graph-text fusion are documented.
- Low-data training controls are part of the supported Graph-LM path.
- Generation can use checkpointed graph memory when available.

## Definition of Done

The architecture part of V2 is complete when:

- The current V2 component boundaries are documented.
- The baseline and V2 graph modes are documented.
- The checkpoint contract is documented.
- Existing Graph-LM and CLI tests pass.

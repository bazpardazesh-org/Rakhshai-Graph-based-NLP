# Rakhshai Multi-Relation Persian Graph

Phase 3 extends the Graph-LM graph builder from a single token co-occurrence
graph to a configurable multi-relation Persian graph.

## Default

The default Graph-LM graph is now the Phase 3 multi-relation preset:

```text
cooccurrence pmi stem subword word_document topic_document
```

This is intentionally stronger than the old v1 graph while avoiding the
heavier optional relations. `dependency` and `semantic_similarity` remain
opt-in because they can be slower or more sensitive to corpus/tokenizer choice.

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn
```

To reproduce the old v1 simple graph baseline, set the relation explicitly:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence \
  --output-dir runs/v1-simple-graph
```

## Relations

- `cooccurrence`: weighted token co-occurrence.
- `pmi` and `ppmi`: association edges computed from the same windows.
- `dependency`: a lightweight Persian dependency-style proximity relation.
- `stem`: edges between tokens sharing a light Persian stem.
- `subword`: edges between related subword pieces.
- `semantic_similarity`: character n-gram similarity between Persian token forms.
- `word_document`: document context nodes linked to token nodes.
- `topic_document`: lightweight topic nodes linked to documents and tokens.

## Example

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --fusion gated \
  --relation-weights cooccurrence=1,pmi=0.7,stem=0.5,subword=0.4 \
  --topic-top-k 8 \
  --output-dir runs/phase3-multi-relation
```

Each run writes the usual Graph-LM checkpoint files. `graph_config.json` now
also records:

- `enabled_relations`
- `relation_weights`
- `edge_types`
- `relation_edge_counts`
- `node_type_counts`

`graph.pt` stores `edge_type` alongside `edge_index` and `edge_weight`, so the
existing GCN, GraphSAGE and GAT encoders can consume relation IDs through the
current edge-type bias path. More expressive relation-aware message passing is
left to Phase 4.

## Presets

Useful explicit presets:

- `default`: `cooccurrence pmi stem subword word_document topic_document`
- `simple-v1`: `cooccurrence`
- `lexical`: `cooccurrence pmi stem subword`
- `document`: `cooccurrence pmi word_document topic_document`
- `full`: `cooccurrence pmi dependency stem subword semantic_similarity word_document topic_document`

Keep `simple-v1` as the control run when comparing against older experiments.

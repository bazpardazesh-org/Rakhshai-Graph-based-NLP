# Rakhshai Graph Reasoning Core

Phase 4 upgrades the Graph-LM encoder from a graph embedding helper into a
relation-aware reasoning core. The previous behaviour remains the default, so
older checkpoints and training commands continue to work.

## Encoder Modes

The Graph-LM path still supports:

- `gcn`
- `graphsage`
- `gat`
- `rgcn`

`rgcn` is the Phase 4 relation-aware encoder. It consumes `edge_type` directly
and uses one relation channel per graph relation saved by the Phase 3 graph
builder.

## Relation Modes

The graph builder emits **multi-relational parallel edges**: a node pair that
participates in several relations contributes one edge per relation, each with
its own `edge_type`, so the encoder sees every relation an edge belongs to rather
than only the last one written.

Use `--graph-relation-mode` to control how `edge_type` is used:

- `bias`: relation ids modulate edge weights with a learned scalar bias (the
  lightest option; the historical Phase 3 path).
- `embedding` (operational default): relation ids are converted into learned edge
  attributes. This gives `GAT` and `GraphSAGE` a richer relation signal while
  keeping the same encoder family, and best exploits the parallel-edge graph.
- `rgcn`: switches the graph encoder to an R-GCN relation-aware message passing
  path.

Example:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder graphsage \
  --graph-relation-mode embedding \
  --graph-relations cooccurrence pmi stem word_document topic_document
```

R-GCN example:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-relation-mode rgcn \
  --graph-relations cooccurrence pmi stem word_document topic_document
```

## Node Importance And Pooling

Phase 4 adds optional node-importance scoring and lightweight graph pooling.

- `--graph-node-importance` enables a scorer over encoded nodes.
- `--graph-pooling mean` adds a pooled non-token subgraph vector to node
  embeddings.
- `--graph-pooling attention` learns attention weights for the pooled vector.

Pooling uses `node_type_id` metadata exported by the Graph-LM graph builder.
Token nodes are always type `0`; document, sentence and topic nodes receive
positive ids.

## Node-Type Embeddings

Every graph node is initialised with a learned **node-type embedding** keyed by
`node_type_id`, so non-token nodes (document, topic, sentence, context) no longer
start from all-zero features and gain meaning only through message passing. Token
nodes add their token embedding on top of the type vector.

- Enabled by default; disable with `--no-graph-node-type-embedding` to restore
  the zero-initialised non-token nodes.
- Config fields: `graph_node_type_embedding` (bool) and `graph_num_node_types`.

## Benchmark Contract

For Phase 4 comparisons, keep the corpus, seed, tokenizer and graph relations
fixed, then compare:

- no graph
- `gcn` with `bias`
- `graphsage` with `bias`
- `gat` with `bias`
- `graphsage` with `embedding`
- `gat` with `embedding`
- `rgcn`

Track:

- validation loss
- perplexity
- epoch time
- graph node/edge counts
- saved `graph_edge_types`


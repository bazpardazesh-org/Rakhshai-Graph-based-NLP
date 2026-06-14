# Rakhshai Multi-Relation Persian Graph

Phase 3 extends the Graph-LM graph builder from a single token co-occurrence
graph to a configurable multi-relation Persian graph.

## Default

The default Graph-LM graph is a multi-relation preset that now includes real
syntactic structure:

```text
cooccurrence pmi dependency stem subword word_document topic_document
```

`dependency` uses a real Persian parser when available (see
[Linguistic backend and semantic method](#linguistic-backend-and-semantic-method))
and falls back to a heuristic otherwise, so it is safe to keep enabled by
default. `semantic_similarity` stays opt-in because computing it scales with the
square of the graph vocabulary.

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
- `dependency`: real Persian syntactic dependency edges (head ↔ dependent) from
  Stanza when installed, with a light-verb proximity heuristic as fallback.
- `stem`: edges between tokens sharing a light Persian stem, enriched with real
  Stanza lemma groups when the backend is available.
- `subword`: edges between related subword pieces.
- `semantic_similarity`: distributional similarity by default (PPMI-weighted
  co-occurrence context vectors compared with cosine); pass
  `--semantic-method orthographic` for the older character n-gram overlap.
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
- `semantic_method` (`distributional` or `orthographic`)
- `linguistic_backend` (`auto`, `stanza` or `heuristic`)
- `dependency_backend` (the backend actually used: `stanza` or `heuristic`)

`graph.pt` stores `edge_type` alongside `edge_index` and `edge_weight`. Edges are
now **multi-relational parallel edges**: when a node pair participates in several
relations (for example cooccurrence *and* stem *and* semantic) each relation is
emitted as its own column with its own `edge_type` and weight, instead of the
last relation overwriting a single id. Relation-aware encoders (`bias`,
`embedding`, `rgcn`) therefore see every relation an edge belongs to — see
[Graph Reasoning Core](graph_reasoning_core.md).

## Linguistic backend and semantic method

The `dependency` and `stem` relations can use a real Persian NLP backend:

- `--linguistic-backend auto` (default): use Stanza when it is installed and the
  Persian model is downloaded, otherwise fall back to the heuristics.
- `--linguistic-backend stanza`: prefer Stanza (still falls back if it cannot be
  loaded).
- `--linguistic-backend heuristic`: never call Stanza.

The `semantic_similarity` relation chooses its scoring with `--semantic-method`:

- `distributional` (default): PPMI-weighted co-occurrence context vectors
  compared with cosine — genuine count-based semantics, pure NumPy, no model
  download.
- `orthographic`: the older character n-gram (Jaccard) overlap.

To enable the real syntactic/lemma backend:

```bash
pip install stanza
python -m stanza.download fa
```

Without Stanza the dependency/stem relations still work via the heuristics, and
`graph_config.json` records `dependency_backend = heuristic` so runs stay
reproducible and transparent.

## Presets

Useful explicit presets:

- `default`: `cooccurrence pmi dependency stem subword word_document topic_document`
- `simple-v1`: `cooccurrence`
- `lexical`: `cooccurrence pmi stem subword`
- `document`: `cooccurrence pmi word_document topic_document`
- `full`: `cooccurrence pmi dependency stem subword semantic_similarity word_document topic_document`

Keep `simple-v1` as the control run when comparing against older experiments.

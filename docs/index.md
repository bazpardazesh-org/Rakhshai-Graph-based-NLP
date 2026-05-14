# Rakhshai Graph-based NLP

Rakhshai Graph-based NLP is a research library for representing Persian text as graphs and applying graph algorithms or graph neural networks.

## Installation

Requires Python 3.9 or newer. Optional extras enable additional features such as sparse graph operations, machine-learning dependencies and documentation tooling:

```bash
pip install -e .
# extras
pip install -e .[sparse]
pip install -e .[ml]
pip install -e .[fa]
pip install -e .[docs]
```

## Key capabilities

- **Graph construction**: `build_cooccurrence_graph`, `build_text_graph`, `build_document_graph`, `build_dependency_graph`, `build_semantic_graph` with lexical relations, FarsNet-style exports and embedding similarity.
- **End-to-end text classification**: `TextGraphClassifier` provides `fit`, `evaluate`, `predict`, `save` and `load` for a complete TextGCN-style Persian classification workflow.
- **Modelling utilities**: PyTorch Geometric based `train_gcn_classifier`/`train_node_classifier` for GCN, GraphSAGE and GAT models plus `compute_social_embeddings`.
- **Training pipeline**: `rgnn-cli` can train on CSV, TSV or JSONL text datasets, create train/validation/test splits, report accuracy and macro-F1, and save model checkpoints.
- **Application tasks**: `textrank_summarise`, `gat_summarise`, `recommend_similar`, `HateSpeechDetector` and others.

The strongest supported path today is Persian text classification. Semantic
graphs, hate-speech detection and GAT-based summarisation now have executable
APIs that can be extended with stronger domain data.

FarsNet support is available through `load_farsnet_relations` and
`build_semantic_graph_from_farsnet`. The project does not bundle the FarsNet
database; provide your licensed JSON, CSV or TSV export when building semantic
graphs.

## Quick examples

### Document classification
```python
from rakhshai_graph_nlp import TextGraphClassifier

docs = [
    "انتخابات و دولت و مجلس",
    "قانون و نمایندگان مجلس",
    "فوتبال و تیم ملی",
    "گل و مسابقه فوتبال",
]
labels = ["politics", "politics", "sports", "sports"]

clf = TextGraphClassifier(model="gcn", num_epochs=20)
clf.fit(docs, labels)
print(clf.evaluate(docs, labels))
print(clf.predict(["تیم فوتبال امروز تمرین کرد"]))

clf.save("runs/news-textgraph")
loaded = TextGraphClassifier.load("runs/news-textgraph")
print(loaded.predict(["مجلس درباره قانون جدید بحث کرد"]))
```

### Text summarisation
```python
from rakhshai_graph_nlp.tasks.summarization import gat_summarise, textrank_summarise
text = "این یک متن نمونه برای خلاصه‌سازی است. ما می‌خواهیم ببینیم آیا الگوریتم می‌تواند جملات مهم را پیدا کند."
print(textrank_summarise(text, top_k=2))
print(gat_summarise(text, top_k=2))
```

### Hate-speech detection
```python
from rakhshai_graph_nlp.tasks.hate_speech import HateSpeechDetector

texts = ["متن آرام و محترمانه", "پیام بد و سمی"]
labels = [False, True]

detector = HateSpeechDetector(num_epochs=20)
detector.fit(texts, labels)
print(detector.predict(["پیام حاوی نفرت"]))
```

### FarsNet semantic graph
```python
from rakhshai_graph_nlp.graphs.semantic import build_semantic_graph_from_farsnet

words = ["ماشین", "خودرو", "پزشک", "دکتر"]
graph = build_semantic_graph_from_farsnet(words, "data/farsnet.json")
print(graph.adjacency)
```

## Command line interface

The CLI supports both a small built-in smoke experiment and a practical
training/evaluation path for labelled text datasets. Without `--dataset`, it
runs the built-in smoke experiment; with `--dataset`, it loads labelled texts,
builds a word-document graph, trains `gcn`, `graphsage` or `gat`, and writes a
JSON evaluation report.

```bash
rgnn-cli --model gcn --device cpu
```

For a real dataset with `text` and `label` columns:

```bash
rgnn-cli \
  --dataset data/news.csv \
  --model gcn \
  --epochs 50 \
  --output-dir runs/news-gcn \
  --save-model runs/news-gcn/model.pt
```

The run writes `metrics.json` with split sizes, label mapping, validation
metrics and test metrics. CSV, TSV and JSONL inputs are supported; use
`--dataset-format`, `--text-column` and `--label-column` when the defaults do
not match your file. The split ratios are controlled by `--train-ratio`,
`--val-ratio` and `--test-ratio`.

Common training and graph options include `--epochs`, `--hidden-dim`,
`--learning-rate`, `--weight-decay`, `--dropout`, `--gat-heads`,
`--window-size`, `--min-count`, `--seed` and `--device`. Use `--output-dir` or
`--report-path` to control report locations, and `--save-model` to write a
PyTorch checkpoint containing the model state, model metadata, label mapping
and metrics. The same options can be supplied from a JSON config file with
`--config config.json`, using option names without `--` and with underscores.
Run `rgnn-cli --help` for the complete list.

## Benchmarks

The repository includes a small repeatable Persian classification benchmark at
`benchmarks/persian_text_classification.csv`. It has 24 short news-like
documents across `politics`, `sports` and `art`, so it is intended for smoke
testing the full graph-classification pipeline rather than making broad model
quality claims.

Example CPU runs with `seed=0`, `epochs=50`, `hidden_dim=8`,
`learning_rate=0.01` and `dropout=0.2`:

| Model | Validation accuracy | Test accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: |
| `gcn` | 1.00 | 0.75 | 0.60 |
| `graphsage` | 1.00 | 1.00 | 1.00 |
| `gat` | 1.00 | 1.00 | 1.00 |

Run the benchmark yourself:

```bash
python -m rakhshai_graph_nlp.cli \
  --dataset benchmarks/persian_text_classification.csv \
  --model gcn \
  --epochs 50 \
  --hidden-dim 8 \
  --learning-rate 0.01 \
  --dropout 0.2 \
  --seed 0 \
  --device cpu \
  --output-dir runs/benchmarks/persian-classification-gcn
```

To compare the supported GNN models:

```bash
for model in gcn graphsage gat; do
  python -m rakhshai_graph_nlp.cli \
    --dataset benchmarks/persian_text_classification.csv \
    --model "$model" \
    --epochs 50 \
    --hidden-dim 8 \
    --learning-rate 0.01 \
    --dropout 0.2 \
    --seed 0 \
    --device cpu \
    --output-dir "runs/benchmarks/persian-classification-$model"
done
```

For larger public Persian benchmarks, convert the dataset to CSV, TSV or JSONL
with `text` and `label` fields and reuse the same CLI. Common choices include
[Hamshahri Corpus](https://en.wikipedia.org/wiki/Hamshahri_Corpus) for Persian
news categorisation, [SnappFood Persian Sentiment](https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis)
for review sentiment, [SentiPers](https://www.researchgate.net/publication/322694676_SentiPers_A_Sentiment_Analysis_Corpus_for_Persian)
for Persian sentiment analysis and [Pars-ABSA](https://arxiv.org/abs/1908.01815)
when aspect-level samples are flattened into rows.

## API reference

::: rakhshai_graph_nlp

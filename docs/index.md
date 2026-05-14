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

- **Graph construction**: `build_cooccurrence_graph`, `build_text_graph`, `build_document_graph`, `build_dependency_graph`, `build_semantic_graph`.
- **Modelling utilities**: PyTorch Geometric based `train_gcn_classifier`/`train_node_classifier` for GCN, GraphSAGE and GAT models plus `compute_social_embeddings`.
- **Training pipeline**: `rgnn-cli` can train on CSV, TSV or JSONL text datasets, create train/validation/test splits, report accuracy and macro-F1, and save model checkpoints.
- **Application tasks**: `textrank_summarise`, `recommend_similar`, `contains_hate_speech` and others.

## Quick examples

### Document classification
```python
import numpy as np
import torch
from rakhshai_graph_nlp.features.pyg_data import graph_to_data
from rakhshai_graph_nlp.features.tokenizer import tokenize
from rakhshai_graph_nlp.graphs.text_graph import build_text_graph
from rakhshai_graph_nlp.tasks.classification import train_gcn_classifier

docs = [
    "این یک خبر سیاسی است و دربارهٔ انتخابات صحبت می‌کند.",
    "تیم فوتبال امروز بازی مهمی دارد و همه منتظر نتیجه هستند.",
    "نمایشگاه جدید هنری با آثار نقاشان جوان افتتاح شد."
]
tokens = [tokenize(d) for d in docs]
graph = build_text_graph(tokens)
X = np.eye(len(graph.nodes))
device = "cuda" if torch.cuda.is_available() else "cpu"
clf, losses = train_gcn_classifier(graph, X, np.zeros(len(graph.nodes), dtype=int), device=device)
data = graph_to_data(graph, features=X).to(device)
preds = clf.predict(data)
```

### Text summarisation
```python
from rakhshai_graph_nlp.tasks.summarization import textrank_summarise
text = "این یک متن نمونه برای خلاصه‌سازی است. ما می‌خواهیم ببینیم آیا الگوریتم می‌تواند جملات مهم را پیدا کند."
print(textrank_summarise(text, top_k=2))
```

## Command line interface

The CLI supports both a small built-in smoke experiment and a practical
training/evaluation path for labelled text datasets:

```bash
rgnn-cli --model gcn --device cuda
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
metrics and test metrics. The same options can be supplied from a JSON config
file with `--config config.json`.

## API reference

::: rakhshai_graph_nlp

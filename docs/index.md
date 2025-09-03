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
- **Modelling utilities**: `train_gcn_classifier` for GCN models and `compute_social_embeddings` based on GraphSAGE.
- **Application tasks**: `textrank_summarise`, `recommend_similar`, `contains_hate_speech` and others.

## Quick examples

### Document classification
```python
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
# ... then train a GCN
clf, losses = train_gcn_classifier(graph)
```

### Text summarisation
```python
from rakhshai_graph_nlp.tasks.summarization import textrank_summarise
text = "این یک متن نمونه برای خلاصه‌سازی است. ما می‌خواهیم ببینیم آیا الگوریتم می‌تواند جملات مهم را پیدا کند."
print(textrank_summarise(text, top_k=2))
```

## Command line interface

A simple CLI is bundled for quick experiments:

```bash
rgnn-cli --model gcn
```

## API reference

::: rakhshai_graph_nlp

# Python API Usage Guide

This guide shows how to use the stable Rakhshai Graph-based NLP API in real
Python code. For the full list of supported names, see
[Stable Public API](api.md).

## 1. Installation

Create a virtual environment, then install the package:

```bash
python -m pip install -e .
```

Install optional extras for feature groups you need:

```bash
python -m pip install -e ".[ml]"    # document graphs, recommendation, TF-IDF
python -m pip install -e ".[nlp]"   # Stanza dependency parsing
python -m pip install -e ".[ui]"    # Gradio UI
python -m pip install -e ".[data]"  # dataset utilities
python -m pip install -e ".[docs]"  # documentation build tools
```

For dependency graphs with Stanza:

```bash
python -m stanza.download fa
```

## 2. Import The Stable API

Use either the package root or the explicit API facade. Both are stable:

```python
from rakhshai_graph_nlp import TextGraphClassifier, build_text_graph
from rakhshai_graph_nlp.api import PersianTokenizer, GraphCausalLM
```

You can inspect the active API contract:

```python
import rakhshai_graph_nlp as rgnn

print(rgnn.API_STATUS)       # stable
print(rgnn.__api_version__)  # 2.1
print(rgnn.stable_api())     # names covered by the stable contract
```

## 3. Normalize And Tokenize Persian Text

Use the shared Persian normalizer before graph construction or feature
extraction:

```python
from rakhshai_graph_nlp import (
    PersianNormalizer,
    PersianNormalizerConfig,
    tokenize_persian,
)

normalizer = PersianNormalizer(
    PersianNormalizerConfig(
        half_space="preserve",
        normalize_hamza=True,
        ezafe_mode="marker",
    )
)

text = "كلاس‌ها و خانهٔ قدیمی"
clean = normalizer.normalize(text)
tokens = tokenize_persian(clean)

print(clean)
print(tokens)
```

For lightweight Unicode tokenization and sentence splitting:

```python
from rakhshai_graph_nlp import tokenize, split_sentences

tokens = tokenize("سلام دنیا! امروز هوا خوب است.")
sentences = split_sentences("سلام دنیا! امروز هوا خوب است.")
```

## 4. Build Graphs From Text

### Co-Occurrence Graph

```python
from rakhshai_graph_nlp import build_cooccurrence_graph, tokenize_persian

docs = ["تهران امروز بارانی بود", "باران و باد در تهران"]
token_lists = [tokenize_persian(doc) for doc in docs]

graph = build_cooccurrence_graph(token_lists, window_size=2, min_count=1)
print(graph.nodes)
print(graph.to_edge_list())
```

### TextGCN Word-Document Graph

```python
from rakhshai_graph_nlp import build_text_graph, tokenize_persian

docs = [
    "دولت و مجلس درباره قانون بحث کردند",
    "تیم فوتبال امروز تمرین کرد",
]
token_lists = [tokenize_persian(doc) for doc in docs]

graph = build_text_graph(token_lists, window_size=5)
print(graph.node_types)  # word nodes first, document nodes last
```

### Document Similarity Graph

Requires `scikit-learn`:

```python
from rakhshai_graph_nlp import build_document_graph

docs = [
    "خبر سیاسی درباره انتخابات",
    "گزارش ورزشی از مسابقه فوتبال",
    "قانون جدید در مجلس بررسی شد",
]

graph = build_document_graph(docs, min_similarity=0.1)
```

### Semantic Graph

```python
from rakhshai_graph_nlp import build_semantic_graph

words = ["ماشین", "خودرو", "پزشک", "دکتر"]
relations = {
    "ماشین": ["خودرو"],
    "پزشک": ["دکتر"],
}

graph = build_semantic_graph(words, relations=relations)
```

### FarsNet-Style Semantic Graph

Provide your own licensed FarsNet-style JSON, CSV or TSV export:

```python
from rakhshai_graph_nlp import build_semantic_graph_from_farsnet

graph = build_semantic_graph_from_farsnet(
    ["ماشین", "خودرو", "پزشک", "دکتر"],
    "data/farsnet.json",
)
```

### Dependency Graph

Requires `stanza` and the Persian model:

```python
from rakhshai_graph_nlp import build_dependency_graph

graph = build_dependency_graph(["او کتاب را خواند."])
```

## 5. Convert Graphs To PyTorch Geometric

Use `graph_to_data` when you want to pass a `Graph` into PyTorch Geometric
models:

```python
import numpy as np

from rakhshai_graph_nlp import build_cooccurrence_graph, graph_to_data

graph = build_cooccurrence_graph([["a", "b", "c"], ["b", "c"]])
features = np.eye(len(graph.nodes), dtype=float)
labels = np.array([0, 1, 1])

data = graph_to_data(graph, features=features, labels=labels)
print(data.edge_index.shape)
```

## 6. Train A Text Classifier

`TextGraphClassifier` is the recommended high-level API for text
classification. It owns graph construction, feature alignment, model training,
prediction, evaluation and save/load.

```python
from rakhshai_graph_nlp import TextGraphClassifier

texts = [
    "انتخابات دولت مجلس",
    "قانون دولت مجلس",
    "فوتبال تیم گل",
    "مسابقه تیم فوتبال",
]
labels = ["politics", "politics", "sports", "sports"]

clf = TextGraphClassifier(
    model="gcn",          # "gcn", "graphsage" or "gat"
    hidden_dim=32,
    num_epochs=30,
    learning_rate=0.01,
    dropout=0.2,
    device="cpu",         # use "cuda" when available
    seed=0,
)

clf.fit(texts, labels)
print(clf.evaluate(texts, labels))
print(clf.predict(["تیم فوتبال امروز تمرین کرد"]))
```

Save and load the full pipeline:

```python
clf.save("runs/news-textgraph")

loaded = TextGraphClassifier.load("runs/news-textgraph")
print(loaded.predict(["مجلس درباره قانون جدید بحث کرد"]))
```

Use GPU when CUDA is available:

```python
clf = TextGraphClassifier(model="gat", device="cuda", num_epochs=50)
```

If CUDA is requested but unavailable, the classifier falls back to CPU.

## 7. Train Lower-Level Node Classifiers

Use this path when you already have a graph, node features and labels:

```python
import numpy as np

from rakhshai_graph_nlp import Graph, train_node_classifier, graph_to_data

graph = Graph(
    nodes=["doc_0", "doc_1", "doc_2"],
    adjacency=np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    ),
)
features = np.eye(3)
labels = np.array([0, 1, 0])
mask = np.array([True, True, False])

model, losses = train_node_classifier(
    graph,
    labels,
    X=features,
    mask=mask,
    model_type="graphsage",
    hidden_dim=8,
    num_epochs=10,
)

data = graph_to_data(graph, features=features, labels=labels)
print(model.predict(data))
```

## 8. Summarization

TextRank and GAT-based extractive summarization are available:

```python
from rakhshai_graph_nlp import gat_summarise, textrank_summarise

text = (
    "امروز باران شدیدی در تهران بارید. "
    "خیابان‌ها خلوت‌تر از روزهای گذشته بودند. "
    "کارشناسان هواشناسی ادامه بارش را پیش‌بینی کردند."
)

print(textrank_summarise(text, top_k=2))
print(gat_summarise(text, top_k=2))
```

These APIs require `scikit-learn` because sentence similarity is built with
TF-IDF.

## 9. Recommendation

Find the most similar documents to a query:

```python
from rakhshai_graph_nlp import recommend_similar

documents = [
    "خبر سیاسی درباره مجلس",
    "گزارش مسابقه فوتبال",
    "قانون تازه در دولت بررسی شد",
]

hits = recommend_similar("مجلس و قانون", documents, top_k=2)
for index, score in hits:
    print(index, score, documents[index])
```

## 10. Hate-Speech Detection

Use the fast keyword helper:

```python
from rakhshai_graph_nlp import contains_hate_speech

print(contains_hate_speech("این متن بد است", ["بد"]))
```

Train a detector:

```python
from rakhshai_graph_nlp import HateSpeechDetector

texts = [
    "متن آرام و محترمانه",
    "گفتگوی خوب و عادی",
    "توهین بد و نفرت",
    "پیام بد و سمی",
]
labels = [False, False, True, True]

detector = HateSpeechDetector(num_epochs=20, learning_rate=0.01)
detector.fit(texts, labels)

print(detector.predict(["متن محترمانه", "پیام نفرت"]))
print(detector.evaluate(texts, labels))

detector.save("runs/hate-detector")
loaded = HateSpeechDetector.load("runs/hate-detector")
```

## 11. Network Embeddings

Use GraphSAGE embeddings for graph or social-network analysis:

```python
import numpy as np

from rakhshai_graph_nlp import Graph, compute_social_embeddings

graph = Graph(
    nodes=["u1", "u2", "u3"],
    adjacency=np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=float,
    ),
)
features = np.random.rand(3, 5)

embeddings = compute_social_embeddings(graph, features, hidden_dims=[8, 4])
print(embeddings.shape)
```

## 12. Metrics

```python
from rakhshai_graph_nlp import accuracy, confusion_matrix, macro_f1

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]

print(accuracy(y_true, y_pred))
print(macro_f1(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
```

## 13. Graph-LM Tokenizer

`PersianTokenizer` is the stable tokenizer for Graph-LM:

```python
from rakhshai_graph_nlp import PersianTokenizer

tokenizer = PersianTokenizer(
    tokenizer_type="unigram",
    unigram_num_pieces=8000,
).fit([
    "امروز در تهران باران بارید",
    "خیابان‌ها خلوت‌تر بودند",
])

ids = tokenizer.encode("امروز در تهران", add_special_tokens=True)
text = tokenizer.decode(ids)

tokenizer.save("runs/tokenizer.json")
loaded = PersianTokenizer.load("runs/tokenizer.json")
```

## 14. Train Graph-LM From Python

For most experiments, the CLI is easier. The Python API is useful when you want
to integrate training inside another program.

```python
from rakhshai_graph_nlp import GraphLMConfig, LMTrainingConfig, train_graph_lm

corpus = [
    "امروز در تهران باران آرامی بارید",
    "مردم در خیابان‌های شهر قدم زدند",
    "هواشناسی برای فردا بارش پراکنده پیش‌بینی کرد",
]

result = train_graph_lm(
    corpus,
    training_config=LMTrainingConfig(
        output_dir="runs/api-graph-lm",
        epochs=1,
        batch_size=2,
        block_size=32,
        device="cpu",
    ),
    model_config=GraphLMConfig(
        vocab_size=1,          # replaced after tokenizer fitting
        max_seq_len=32,
        d_model=64,
        n_heads=4,
        n_layers=1,
        graph_encoder="none",
    ),
    graph_encoder="none",      # use "gcn", "graphsage", "gat" or "rgcn" for graph runs
)

print(result["checkpoint_dir"])
```

For graph-enabled runs:

```python
from rakhshai_graph_nlp import GraphLMConfig, LMTrainingConfig, train_graph_lm

result = train_graph_lm(
    corpus,
    training_config=LMTrainingConfig(
        output_dir="runs/api-graph-lm-gat",
        epochs=1,
        graph_relations=[
            "cooccurrence",
            "pmi",
            "dependency",
            "stem",
            "subword",
            "word_document",
            "topic_document",
        ],
        graph_relation_mode="embedding",
        fusion_levels="token,sentence",
        device="cpu",
    ),
    model_config=GraphLMConfig(vocab_size=1, graph_encoder="gat"),
    graph_encoder="gat",
    fusion="gated",
)
```

## 15. Load And Generate With Graph-LM

```python
import torch

from rakhshai_graph_nlp import GraphCausalLM, GraphMemoryConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer, generation_config, graph_config = GraphCausalLM.from_pretrained(
    "runs/api-graph-lm",
    map_location=device,
)
model.to(device)

graph_data, token_node_ids = GraphCausalLM.load_graph_artifacts(
    "runs/api-graph-lm",
    map_location=device,
)
if graph_data is not None:
    graph_data = graph_data.to(device)
if token_node_ids is not None:
    token_node_ids = token_node_ids.to(device)

text = model.generate(
    "امروز در تهران",
    tokenizer,
    graph_data=graph_data,
    token_node_ids=token_node_ids,
    graph_memory_config=GraphMemoryConfig(enabled=True),
    generation_config=generation_config,
    max_new_tokens=50,
)
print(text)
```

For no-graph checkpoints, `graph_data` and `token_node_ids` may be `None`; pass
them through unchanged.

## 16. Graph-LM Poem Recommendation

After training a Graph-LM checkpoint, build a poem index:

```python
from rakhshai_graph_nlp import PoemRecommender, build_poem_index

poems = [
    {"poet": "حافظ", "poem": "نمونه ۱", "text": "الا یا ایها الساقی ..."},
    {"poet": "سعدی", "poem": "نمونه ۲", "text": "بنی آدم اعضای یکدیگرند ..."},
]

build_poem_index("runs/api-graph-lm", poems, device="cpu")

recommender = PoemRecommender("runs/api-graph-lm", device="cpu")
for hit in recommender.search("ای ساقی بیا", top_k=3):
    print(hit["score"], hit.get("poet"), hit.get("poem"))
```

## 17. CLI And API Together

You can train with the CLI and load with the API:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gat \
  --fusion gated \
  --output-dir runs/wiki-graph-lm
```

Then:

```python
from rakhshai_graph_nlp import GraphCausalLM

model, tokenizer, generation_config, graph_config = GraphCausalLM.from_pretrained(
    "runs/wiki-graph-lm"
)
```

You can also train text classification from the CLI and use
`TextGraphClassifier` when you need full Python save/load pipeline control.

## 18. Common Errors

### `ImportError: scikit-learn is required`

Install:

```bash
python -m pip install -e ".[ml]"
```

This affects document graphs, recommendation and summarization.

### `ImportError: Stanza is required`

Install Stanza and download the Persian model:

```bash
python -m pip install -e ".[nlp]"
python -m stanza.download fa
```

This affects dependency graphs and the real linguistic backend.

### CUDA Requested But Not Available

High-level classifiers fall back to CPU when possible. For manual PyTorch code,
check:

```python
import torch

print(torch.cuda.is_available())
```

### Empty Or Tiny Corpora

Graph and LM builders need enough tokens to create useful examples. If you see
errors such as "corpus must contain at least two language-model tokens" or "No
words meet the frequency threshold", lower `min_count`, add more text, or reduce
`block_size` for smoke tests.

## 19. Recommended Import Style

For application code:

```python
from rakhshai_graph_nlp import TextGraphClassifier, PersianTokenizer
```

For explicit API auditing:

```python
from rakhshai_graph_nlp import api

assert "TextGraphClassifier" in api.stable_api()
```

For advanced internal work, submodule imports still work:

```python
from rakhshai_graph_nlp.lm.graph_builder import build_graph_lm_graph
```

The stable contract, however, is the root package plus `rakhshai_graph_nlp.api`.

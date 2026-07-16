English | [فارسی](README.fa.md)

# Rakhshai Graph-based NLP (RGN)

[![CI](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml)

**Build Persian graph NLP and native Graph-LM products from one open-source
stack.**

RGN is the first integrated graph-oriented NLP platform for Persian,
developed by
[Aria Haman Mehr Parseh](https://ariahaman.ir/en/). We publicly unveiled RGN
on September 4, 2025, 10 months and 11 days before unveiling the Haman model
on July 15, 2026. RGN combines Persian text preparation, multi-relation graph
construction, graph neural networks, native language-model training,
inference, checkpointing, MCP integration, and an RTL web interface in one
Python, CLI, and UI stack.

Use RGN to deploy the released Haman article model, train a native Persian
Graph-LM on your own corpus, or build graph-based classification,
summarization, recommendation, and semantic-analysis workflows.

## Featured Model: Haman Persian Article Graph-LLM 125M

Continuing this path, on July 15, 2026, we unveiled
[Haman Persian Article Graph-LLM 125M](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m)
as the first output of Aria Haman Mehr Parseh's NLP project. This model is the
first Iranian model developed with an Iranian architecture so that we can
subsequently create a framework focused on Persian. It generates structured
Persian articles from topic, audience, tone, and section controls by combining
a decoder-only Transformer with a corpus-level lexical GCN and context-gated
graph-token fusion.

| Resource | Link |
| --- | --- |
| Model card and weights | [Hugging Face model](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m) |
| Training dataset | [Haman Persian Wikipedia Articles 186K](https://huggingface.co/datasets/aria-haman/haman-fa-wikipedia-articles-186k) |
| Reproducible training workflow | [Clean Google Colab notebook](https://colab.research.google.com/drive/1E50ISg1ANoW_rrFeRNBRcDn6C0cwfLyT?usp=sharing) |
| Runtime and source | [Rakhshai Graph-based NLP (RGN)](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP) |

The model remains in its dedicated Hugging Face repository so users can fetch
versioned weights without adding large checkpoint files to this source
repository. For files, generation examples, deployment notes, and the model
license, see its model card.

### Use Haman

Install RGN, download the published snapshot, and generate an article:

```bash
python -m pip install -e .
python -m pip install huggingface_hub
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="aria-haman/haman-fa-article-graph-llm-125m", local_dir="models/haman-fa-article-graph-llm-125m")'

rgnn-cli article-generate \
  --model models/haman-fa-article-graph-llm-125m \
  --topic "آینده هوش مصنوعی در آموزش فارسی" \
  --audience "دانشجویان" \
  --tone "تحلیلی" \
  --sections 4 \
  --max-new-tokens 700 \
  --output-format markdown \
  --output-path haman-article.md
```

> Rakhshai Graph-based NLP (RGN) has received knowledge-based
> certification in Iran. Official inquiry details are provided below.

## Product Platform

RGN provides four connected product layers. Teams can use each layer
independently or combine them into a complete Persian AI workflow.

| Layer | What it provides |
| --- | --- |
| Persian NLP | Normalization, tokenization, linguistic analysis, and reusable features |
| Graph intelligence | Textual, syntactic, semantic, document, and multi-relation graphs |
| Native model engine | `rakhshai_graph_nlp.lm` for tokenizer, Graph-LM, training, evaluation, graph memory, and inference |
| Product workflows | `rakhshai_graph_nlp.llm` for task-specific native LLM products, beginning with structured Persian article generation |

The high-level `llm` workflows use the lower-level `lm` engine; they do not
replace it. This separation keeps the core model engine reusable while giving
product teams stable commands and structured outputs for specific use cases.

The open-source platform is created and maintained by the
[RakhshAI](https://rakhshai.com/) team at Aria Haman Mehr Parseh and is released
under the MIT license.

## Product Capabilities

- **Integrated Persian graph NLP platform:** Move from raw Persian text to
  preprocessing, graph construction, model training, evaluation, inference,
  and saved pipelines in one stack.
- **Released Persian article model:** Use Haman 125M immediately, with public
  weights, dataset, training workflow, and a structured generation interface.
- **Native LLM production workflow:** Prepare and audit data, train and ablate
  checkpoints, and generate Markdown or JSON through the high-level
  `rakhshai_graph_nlp.llm` layer.
- **Reusable Graph-LM engine:** Build with a Persian-aware tokenizer, modern
  causal decoder, GCN/GraphSAGE/GAT/RGCN encoders, adaptive graph-text fusion,
  Graph Memory, evaluation, and complete checkpointing.
- **Graph intelligence toolkit:** Create co-occurrence, word-document, document
  similarity, dependency, semantic, and multi-relation graphs.
- **Application-ready components:** Build classification, summarization,
  recommendation, hate-speech detection, semantic analysis, and network
  analysis workflows.
- **Multiple product interfaces:** Use the stable Python API, `rgnn-cli`, the
  RTL web UI, or MCP integration for agents and automated systems.
- **Operational controls:** Run on CPU or supported GPU paths, compare graph
  and no-graph variants, save complete artifacts, and control generation
  through temperature, top-k, minimum length, and repetition settings.

## MCP Integration

RGN exposes its Persian text analysis, graph construction, graph-based
summarization, Graph Memory retrieval, Graph-LM generation, explainability, and
project resources through MCP. Product teams can connect these capabilities to
agents, IDEs, chatbots, and automation workflows through a controlled tool
surface.

- [MCP integration and deployment guide](docs/mcp.md)
- [Published MCP evaluation report](docs/mcp_single_poem_evaluation.md)

## RGN Graph-LM Architecture

RGN's Graph-LM runtime follows this path:

```text
Persian Text
→ PersianTokenizer
→ LM Dataset
→ Multi-Relation Persian Graph
→ RGN Graph Encoder (GCN / GraphSAGE / GAT / RGCN)
→ Adaptive Graph-Text Fusion
→ Low-Data Training Engine
→ Prompt-aware Graph Memory
→ Transformer Causal LM
→ Text Generation
```

The core runtime combines:

```text
RGN Graph Encoder
+
Adaptive Graph-Text Fusion
+
Low-Data Training Engine
+
Persian Causal LM
```

In simple terms, RGN can use graph relationships between words instead of
acting as a purely sequential language model. When building the final embedding
for each token, the model learns how much to trust the text embedding and how
much to trust the graph embedding. The combination is not fixed by hand; it is
learned through a gate. This gate can operate at several levels: token level for
each sequence position, sentence level for controlling overall graph strength in
the context, and subgraph level for injecting a summary of non-token nodes such
as documents or topics.

> **Checkpoint compatibility:** Older checkpoints load in compatibility mode,
> but retraining is required to use the current decoder and tokenizer layout.
> See the [Graph-LM V2 guide](docs/graph_lm_v2.md) for migration details.

## Choose Your Product Path

| Goal | Recommended path |
| --- | --- |
| Generate structured Persian articles now | Use the [released Haman model](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m) |
| Build another task-specific native LLM | Start with the high-level [`llm` workflows](docs/llm.md) |
| Train or extend the reusable model engine | Use the lower-level [`lm` engine](docs/graph_lm_v2.md) |
| Build graph-based NLP applications | Use the graph builders, GNN models, tasks, Python API, or CLI in this repository |

## Train a Custom Graph-LM

A Persian Wikipedia sample for a quick local start is included at:

```text
data/wiki_fa_50k.txt
```

You can train and generate directly with this file. To rebuild it or create a
larger sample, use:

```bash
python scripts/download_fa_wiki_sample.py \
  --output data/wiki_fa_50k.txt \
  --max-rows 50000 \
  --min-length 200
```

Example no-graph baseline run:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder none \
  --output-dir runs/wiki-baseline-lm
```

Example Graph-LM run with GAT and gated fusion:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder gat \
  --fusion gated \
  --output-dir runs/wiki-graph-lm
```

Example text generation from a Graph-LM model:

```bash
rgnn-cli generate \
  --model runs/wiki-graph-lm \
  --prompt "امروز در تهران" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50 \
  --repetition-penalty 1.2
```

Independent engine-level LM training is also available without external model
dependencies:

```bash
rgnn-cli lm-build-corpus --input data/wiki_fa_test.txt --output-dir runs/fa-corpus
rgnn-cli lm-tokenize --corpus-dir runs/fa-corpus --output-dir runs/fa-shards
rgnn-cli lm-pretrain \
  --shard-manifest runs/fa-shards/shard_manifest.json \
  --output-dir runs/fa-pretrain \
  --model-profile tiny-test \
  --device cpu
rgnn-cli lm-ablation \
  --corpus runs/fa-corpus/train.txt \
  --output-dir runs/fa-ablation \
  --graph-encoders none gat \
  --device cpu
rgnn-cli lm-eval --model runs/fa-pretrain --eval-file runs/fa-corpus/validation.txt
rgnn-cli lm-run-report --run-dir runs/fa-pretrain
```

The native LM engine does not use external pretrained LMs, pretrained
embeddings, distillation, LLM-generated synthetic data or external LLM judges.

## Native Persian Article LLM Workflow

This is the product workflow behind the released
[Haman Persian Article Graph-LLM 125M](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m).
You can use the published checkpoint with its
[186K-article dataset](https://huggingface.co/datasets/aria-haman/haman-fa-wikipedia-articles-186k)
and [clean Colab workflow](https://colab.research.google.com/drive/1E50ISg1ANoW_rrFeRNBRcDn6C0cwfLyT?usp=sharing),
or train a task-specific native checkpoint on your own Persian corpus with the
commands below.

The workflow is exposed through `rakhshai_graph_nlp.llm.article`. It packages
the lower-level `rakhshai_graph_nlp.lm` tokenizer, graph builder, model,
trainer, and graph memory into an article-focused training and generation
pipeline; the reusable Graph-LM engine remains independent.

Technical flow:

- `article-prepare` normalizes raw TXT/JSONL/CSV/TSV article data, formats it
  for article-style language-model training, and writes `corpus.txt`,
  `train.txt`, `validation.txt`, `prepared_articles.jsonl`,
  `rejected_records.jsonl`, and `manifest.json`.
- `article-audit` checks native corpus quality, duplicate risk, Persian surface
  statistics and tokenizer behavior before expensive training.
- `article-train` trains an article-focused Graph-LM checkpoint with the normal
  model, tokenizer, graph, graph-memory and metrics artifacts, plus
  `article_llm_config.json` for article workflow metadata.
- `article-ablation` runs no-graph/graph/scope/relation variants and records
  validation metrics, fusion statistics and zero-gate reports.
- `article-generate` loads that checkpoint and returns a structured Persian
  article as Markdown or JSON. The CLI keeps the top-level `article-*` commands,
  and checkpoint artifact names remain unchanged.

Complete build-and-train flow:

1. Collect article data as TXT, JSONL, CSV or TSV. For structured files, `body`
   is required and `title`, `summary`, `keywords` and `metadata` are optional.
   Persian Wikipedia-style JSONL can use `title` and `text` with
   `--training-format wikipedia_prompt`.
2. Run `article-prepare` to normalize records, reject too-short bodies and write
   deterministic `train.txt` and `validation.txt` splits.
3. Run `article-audit` before expensive training. It reports Persian character
   coverage, duplicate risk, metadata/source coverage and tokenizer behavior;
   optionally enable tokenizer probe training when choosing between `word`,
   `bpe` and `unigram`.
4. Train with `article-train`. The command below uses CUDA, AMP, a
   `context_gated` graph fusion path, a Unigram tokenizer and a reusable graph
   cache. Use `--resume-from runs/article-llm-fa` to continue
   an interrupted run.
5. Inspect `metrics.json`, `article_llm_config.json`, `config.json`,
   `generation_config.json`, `tokenizer.json`, `model.pt`, `corpus.txt` and,
   for graph-enabled runs, `graph.pt` and `graph_memory.pt`.
6. Use the trained checkpoint with `article-generate` or the Python API shown
   below.

```bash
rgnn-cli article-prepare \
  --input data/persian_articles.jsonl \
  --output-dir runs/articles-prepared \
  --input-format jsonl \
  --min-body-chars 400 \
  --validation-ratio 0.1

rgnn-cli article-audit \
  --input data/persian_articles.jsonl \
  --output-dir runs/articles-audit \
  --input-format jsonl \
  --min-body-chars 400 \
  --tokenizer-types word bpe unigram

rgnn-cli article-train \
  --corpus runs/articles-prepared/corpus.txt \
  --output-dir runs/article-llm-fa \
  --device cuda \
  --amp \
  --batch-size 16 \
  --epochs 10 \
  --block-size 256 \
  --graph-encoder gat \
  --fusion context_gated \
  --graph-cache-dir runs/graph-cache \
  --tokenizer-type unigram \
  --unigram-num-pieces 32000
```

Use the trained article checkpoint from the CLI:

```bash
rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "آینده هوش مصنوعی در آموزش فارسی" \
  --audience "دانشجویان" \
  --tone "تحلیلی" \
  --sections 4 \
  --max-new-tokens 700 \
  --output-format markdown \
  --output-path runs/article-llm-fa/education_article.md

rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "اقتصاد دیجیتال ایران" \
  --sections 4 \
  --output-format json \
  --output-path runs/article-llm-fa/economy_article.json
```

When `article-train` receives the prepared `corpus.txt`, it uses the sibling
`train.txt` and `validation.txt` files created by `article-prepare`.

In `article-generate`, Graph Memory is enabled by default. If the checkpoint
contains `graph_memory.pt`, that memory is loaded. If it does not, and
`corpus.txt` exists inside the checkpoint, the memory is rebuilt from the corpus
and `graph_config.json`. To disable memory for an article-generation run:

```bash
rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "آینده آموزش فارسی" \
  --graph-memory off
```

Use the same checkpoint from Python:

```python
from rakhshai_graph_nlp.llm.article import (
    ArticleGenerationConfig,
    generate_persian_article,
)

article = generate_persian_article(
    ArticleGenerationConfig(
        model_dir="runs/article-llm-fa",
        topic="آینده آموزش فارسی",
        audience="دانشجویان",
        tone="تحلیلی",
        sections=4,
        max_new_tokens=700,
        graph_memory=True,
        device="cuda",
    )
)

print(article.full_markdown)
print(article.to_json())
```

## Why Graphs?

Text is not only a sequence of words. In Persian, word relationships,
co-occurrence patterns, syntactic dependencies, document similarity, and links
between concepts matter a great deal. Graphs make these relationships explicit
and give them to the model.

In the TextGCN approach, words and documents become nodes in a graph. Word-word
relationships can be built with PMI, and word-document relationships can be
weighted with TF-IDF. A graph model such as GCN or GAT then propagates
information through this network and predicts labels for document nodes.

Put simply: instead of looking at each text in isolation, we view the whole text
collection as a network. This is especially useful for Persian, because the
meaning and role of words often become clearer through their relationships with
nearby words and other documents.

## Main Components

As of `2.2.0`, the Python API is marked stable (`API_STATUS = "stable"`,
`__api_version__ = "2.2"`). Application code can import the supported surface
directly from `rakhshai_graph_nlp` or from the explicit facade
`rakhshai_graph_nlp.api`:

```python
from rakhshai_graph_nlp import TextGraphClassifier, build_text_graph
from rakhshai_graph_nlp.api import GraphCausalLM, PersianTokenizer
```

The full step-by-step Python guide is in
[`docs/api_usage.md`](docs/api_usage.md), and the stable reference is in
[`docs/api.md`](docs/api.md).

| Component | Description |
| --- | --- |
| `Graph` | Lightweight graph container with adjacency, node metadata, self-loops, degree and normalization helpers |
| `TextGraphClassifier` | Main class for training, evaluation, prediction, saving, and loading text classification models |
| `train_node_classifier` / `train_gcn_classifier` | Lower-level GNN node-classification training helpers |
| `build_text_graph` | Builds a word-document graph with PMI and TF-IDF for TextGCN |
| `build_cooccurrence_graph` | Builds word co-occurrence graphs with a word window |
| `build_document_graph` | Builds document similarity graphs with TF-IDF or embeddings |
| `build_dependency_graph` | Builds dependency graphs with Stanza |
| `build_semantic_graph` | Builds semantic graphs from lexical relations and embedding similarity |
| `build_semantic_graph_from_farsnet` | Builds semantic graphs from FarsNet JSON/CSV/TSV output |
| `load_farsnet_relations` | Loads FarsNet relations from a file and converts them into graph-usable relations |
| `tokenize`, `tokenize_persian`, `PersianNormalizer` | Stable text tokenization and Persian normalization helpers |
| `graph_to_data`, `build_feature_matrix` | Bridge dense graphs and Persian text features into PyTorch Geometric |
| `PersianTokenizer` | Numeric LM tokenizer with half-space support, Persian cleanup, standalone punctuation tokens, numeric-separator and configurable hamza/ezafe normalization, a `<mask>` special token, and `word`/`char_chunk`/`bpe`/`unigram` modes (`unigram` is the operational default) |
| `LMDataset` | Prepares `input_ids` and `target_ids` for next-token prediction |
| `build_graph_lm_graph` | Builds a word co-occurrence graph from a corpus for Graph-LM |
| `GraphCausalLM` | Persian language model with GNN encoder, gated graph-token fusion, and a modern decoder-only Transformer (RoPE, SwiGLU, RMSNorm, KV-cache generation) |
| `RakhshaiGraphEncoder` | Graph-LM graph core with `gcn`, `graphsage`, `gat`, `rgcn`, and relation-aware encoding support |
| `GraphMemoryArtifact` / `GraphMemoryConfig` | Prompt-aware graph memory used during generation |
| `LMTrainer`, `LMTrainingConfig`, `train_graph_lm` | Stable Graph-LM training APIs with checkpointing, validation and perplexity |
| `TextAugmentationConfig`, `augment_text`, `augment_graph_data` | Low-data text and graph regularization helpers |
| `PoemRecommender`, `build_poem_index` | Graph-LM powered poem embedding, indexing and search |
| `rakhshai_graph_nlp.llm.article` | High-level Persian article LLM workflow namespace built on top of the Graph-LM engine |
| `ArticleCorpusConfig`, `prepare_article_corpus` | Article dataset preparation API for raw TXT/JSONL/CSV/TSV inputs |
| `ArticleAuditConfig`, `audit_article_corpus` | Native article corpus audit and tokenizer benchmark API |
| `ArticleTrainingConfig`, `train_article_llm` | Article-focused training profile that still writes normal Graph-LM checkpoint artifacts |
| `ArticleAblationConfig`, `run_article_ablation` | Native article graph ablation runner |
| `ArticleGenerationConfig`, `generate_persian_article`, `PersianArticle` | Structured Persian article generation API with Markdown and JSON output |
| `--graph-encoder none` | No-graph baseline for comparing against Graph-LM and measuring the true effect of GNN/fusion |
| `GCNClassifier` | GCN model for node classification |
| `GraphSAGEClassifier` | GraphSAGE model for neighborhood-based learning |
| `GATClassifier` | GAT model with graph attention |
| `textrank_summarise`, `gat_summarise` | Extractive summarization APIs |
| `recommend_similar`, `HateSpeechDetector`, `compute_social_embeddings` | Recommendation, hate-speech detection and network-analysis APIs |
| `accuracy`, `macro_f1`, `confusion_matrix` | Stable evaluation metrics |
| `rgnn-cli` | Command-line tool for quick training and evaluation |

## Analytical Tasks

- **Persian text classification:** Classify news, messages, user comments,
  social media content, or any other labeled text.
- **Extractive summarization:** Select important sentences with
  `textrank_summarise` or the GAT-based ranker `gat_summarise`.
- **Content recommendation:** Find documents close to a text or query with
  `recommend_similar`.
- **Hate-speech detection:** Use quick term-based checks with
  `contains_hate_speech` and trainable models with `HateSpeechDetector`.
- **Network analysis:** Build node embeddings with GraphSAGE and use them for
  clustering, relationship analysis, or influence analysis.
- **Persian semantic graphs:** Connect related words through FarsNet, manual
  relations, synonyms, or Persian embeddings.

## GPU Support

The project supports GPU execution for PyTorch Geometric graph models. If your
system has an NVIDIA GPU, a suitable driver, and a CUDA-compatible PyTorch
installation, you can train GCN, GraphSAGE, and GAT on GPU.

GPU paths include:

- `TextGraphClassifier(..., device="cuda")`
- `train_node_classifier(..., device="cuda")`
- `train_gcn_classifier(..., device="cuda")`
- CLI execution with `--device cuda`

Python example:

```python
from rakhshai_graph_nlp import TextGraphClassifier

clf = TextGraphClassifier(model="gcn", device="cuda", num_epochs=50)
clf.fit(texts, labels)
print(clf.predict(["تیم فوتبال امروز تمرین کرد"]))
```

CLI example:

```bash
rgnn-cli \
  --dataset benchmarks/persian_text_classification.csv \
  --model gcn \
  --device cuda \
  --epochs 50 \
  --output-dir runs/news-gcn-gpu
```

If CUDA is not available, the CLI falls back to CPU so execution does not stop.
To check CUDA in your environment:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Important note: algorithms such as TextRank, the TF-IDF recommender, and some
NumPy-oriented utilities are CPU-oriented by nature. GPU is most useful for GNN
training and PyTorch/PyG model execution.

## Simple Installation

Installing inside a virtual environment is recommended.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

On most Macs, the project runs on CPU. If you use Apple Silicon, PyTorch parts
may use PyTorch accelerators, but the official GPU path of this project is
currently focused on `cuda`.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

For GPU on Linux, first install a PyTorch version compatible with your CUDA
setup, then install the project:

```bash
python -c "import torch; print(torch.cuda.is_available())"
rgnn-cli --model gcn --device cuda
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

For GPU on Windows, the NVIDIA driver and a CUDA-compatible PyTorch version must
be installed correctly:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
rgnn-cli --model gcn --device cuda
```

### Optional Extras

```bash
python -m pip install -e ".[ml]"    # scikit-learn tools for document graphs and recommendation
python -m pip install -e ".[nlp]"   # Stanza support for dependency graphs
python -m pip install -e ".[fa]"    # Faiss support
python -m pip install -e ".[docs]"  # documentation build tools
python -m pip install -e ".[dev]"   # testing, linting, build, and publishing tools
```

> **Accuracy note:** Installing Stanza is strongly recommended when you use
> dependency graphs or the default Graph-LM multi-relation graph, which includes
> `dependency`. If Stanza is not installed, the `auto` linguistic backend falls
> back to a simpler heuristic backend. The project will still run, but the
> quality of `dependency`/lemma relations, and therefore the accuracy of models
> that rely on those linguistic relations, may be lower.

### Running Tests

```bash
python -m pip install pytest
python -m pytest
```

## 🖥️ Graphical UI

The fastest way to see everything the project can do is a complete, fully
Persian, right-to-left web interface in [`app.py`](app.py).

### Launch the UI (easiest way)

```bash
# Option 1 — one-command launcher (installs prerequisites for you)
./run_ui.sh

# Option 2 — run directly
pip install -r requirements-ui.txt   # or: pip install -e ".[ui]"
python app.py
```

Then open **http://127.0.0.1:7860** in your browser (the launcher opens it for
you automatically).

### Installing prerequisites

The launcher installs the **core** requirements and the UI for you automatically.
Beyond those, the project has a few **optional packages** you can install only if
you need the features they unlock:

> ⚡ **To use the project at its full power, all prerequisites should be
> installed.** The easiest way is to install them all at once:
> `pip install -e ".[all]"`

| Package | Install | Unlocks |
| --- | --- | --- |
| **scikit-learn** | `pip install "scikit-learn>=1.2"` | ML metrics, TF-IDF summarization, document graphs |
| **stanza** | `pip install "stanza>=1.6"` | Advanced Persian NLP: POS tagging, lemmatization, dependency parsing |
| **faiss-cpu** | `pip install "faiss-cpu>=1.7.4"` | Fast vector similarity search |

> **Stanza** also needs its Persian language model the first time:
> `python -m stanza.download fa`

**Install everything at once** (all optional packages):

```bash
pip install -e ".[all]"
```

**Check what you already have** — this prints each optional package with whether
it is installed or not (and the command to install the missing ones):

```bash
./run_ui.sh check
```

Example output:

```
Optional packages — install only the ones you need:

  • scikit-learn — ML metrics, TF-IDF summarization, document graphs
      ✅ installed
  • stanza — advanced Persian NLP: POS, lemma, dependency parsing
      ❌ not installed   →  python3 -m pip install "stanza>=1.6"
  • faiss-cpu — fast vector similarity search
      ❌ not installed   →  python3 -m pip install "faiss-cpu>=1.7.4"
```

### Controlling the server

The `run_ui.sh` launcher gives you simple commands to start and stop the UI:

```bash
./run_ui.sh                 # start the UI
./run_ui.sh --share         # start and create a temporary public link
./run_ui.sh --port 8080     # start on a custom port
./run_ui.sh stop            # stop the server and free the port
./run_ui.sh restart         # stop, then start again
./run_ui.sh status          # show whether it is running, and on which port
./run_ui.sh help            # show all options
```

- **To stop while it is running**, press **`Ctrl + C`** in the same terminal —
  this shuts the server down cleanly and frees the port. (Avoid `Ctrl + Z`; that
  only *suspends* the process and keeps the port busy.)
- **From another terminal**, you can stop it any time with `./run_ui.sh stop`.
- If the port was left busy by a previous run, the launcher **frees it
  automatically** before starting, so you never hit a "port already in use" error.

> The launcher prints status messages in English because most terminals do not
> render right-to-left Persian correctly. The web UI itself is fully Persian and
> right-to-left.

### What's inside the UI

The UI bundles the Persian tokenizer, an interactive multi-relation text-graph
viewer, **live-progress Graph-LM training**, graph-memory text generation,
graph-based classification, the analytical tasks (summarization, recommendation,
hate-speech), and a guided **"Full Power"** step-by-step tour.

## Graph Classification Quick Start

```python
from rakhshai_graph_nlp import TextGraphClassifier

texts = [
    "انتخابات و دولت و مجلس",
    "قانون و نمایندگان مجلس",
    "فوتبال و تیم ملی",
    "گل و مسابقه فوتبال",
]
labels = ["politics", "politics", "sports", "sports"]

clf = TextGraphClassifier(model="gcn", num_epochs=20)
clf.fit(texts, labels)

print(clf.evaluate(texts, labels))
print(clf.predict(["تیم فوتبال امروز تمرین کرد"]))

clf.save("runs/news-textgraph")

loaded = TextGraphClassifier.load("runs/news-textgraph")
print(loaded.predict(["مجلس درباره قانون جدید بحث کرد"]))
```

## Command-Line Interface

`rgnn-cli` exposes three paths: classic graph-based Persian NLP through the
default classification command; the lower-level `lm` engine through commands
such as `lm-build-corpus`, `lm-pretrain`, `lm-train`, `lm-eval`, and `generate`;
and high-level `llm` product workflows through `article-prepare`,
`article-audit`, `article-train`, `article-ablation`, and `article-generate`.
If you do not provide a subcommand, the classic classification path runs.

Train Graph-LM:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --fusion gated \
  --output-dir runs/graph-lm
```

By default, the Graph-LM path builds the full multi-relation graph:

```text
cooccurrence + pmi + dependency + stem + subword + word_document + topic_document
```

To reproduce the older simple baseline, explicitly limit relations:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence \
  --output-dir runs/simple-graph-baseline
```

To enable Graph Reasoning Core, pass richer relations into the encoder. A light
GraphSAGE relation-aware example:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder graphsage \
  --graph-relation-mode embedding \
  --graph-pooling attention \
  --graph-node-importance \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/graphsage-reasoning
```

Direct R-GCN example:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-relation-mode rgcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/rgcn-reasoning
```

To enable Adaptive Graph-Text Fusion, explicitly configure fusion levels and
graph usage strength:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --fusion context_gated \
  --fusion-levels token,sentence,subgraph \
  --graph-fusion-scale 0.75 \
  --graph-fusion-dropout 0.1 \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/adaptive-fusion
```

Train a no-graph baseline for comparison:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder none \
  --output-dir runs/baseline-lm
```

Multi-task Graph-LM training is enabled by default. In every batch, in addition
to `next_token`, the trainer computes `masked_token`, `edge`, `neighbor`,
`node_relation`, `graph_text`, and `sentence_graph` signals when the required
data is available. If you run the model without a graph, graph losses are
automatically skipped and the no-graph baseline remains comparable.

Controlled multi-task loss weighting example:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --task-losses next_token,masked_token,edge,node_relation,graph_text,sentence_graph \
  --next-token-weight 1.0 \
  --masked-token-weight 0.25 \
  --edge-prediction-weight 0.1 \
  --node-relation-weight 0.1 \
  --graph-text-alignment-weight 0.1 \
  --sentence-graph-alignment-weight 0.1 \
  --mask-probability 0.15 \
  --negative-samples 1 \
  --output-dir runs/multitask-graph-lm
```

**Low-Data Training Engine** is enabled by default. It is designed for small
corpora and uses text augmentation, graph dropout, subgraph sampling,
contrastive consistency, curriculum learning, and early stopping to reduce
memorization and improve validation behavior.

Explicit run with the recommended default settings:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --augmentation-ratio 0.5 \
  --token-dropout 0.05 \
  --punctuation-dropout 0.5 \
  --node-dropout 0.05 \
  --edge-dropout 0.1 \
  --subgraph-sampling-ratio 0.9 \
  --contrastive-weight 0.05 \
  --early-stopping-patience 3 \
  --output-dir runs/low-data-training-engine
```

For ablation or simpler training, disable these features:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --no-text-augmentation \
  --edge-dropout 0 \
  --node-dropout 0 \
  --subgraph-sampling-ratio 1 \
  --contrastive-weight 0 \
  --no-curriculum \
  --early-stopping-patience 0 \
  --output-dir runs/no-low-data-regularization
```

Generate text with a saved checkpoint:

```bash
rgnn-cli generate \
  --model runs/graph-lm \
  --prompt "امروز در تهران" \
  --max-new-tokens 100 \
  --min-new-tokens 20 \
  --temperature 0.8 \
  --top-k 50 \
  --repetition-penalty 1.2 \
  --graph-memory on \
  --graph-memory-top-k 32 \
  --graph-memory-depth 1 \
  --graph-memory-report-path runs/graph-lm/memory-report.json
```

When `--graph-memory on` is enabled, `generate` first connects prompt tokens to
memory nodes, scores related neighbors with relation weights, builds a limited
subgraph, and passes that subgraph to Graph-LM. This reduces unrelated
information from the full graph entering generation. When `--graph-memory off`
is used, no retrieval is performed and the model behaves as before, using
`graph.pt`, a dynamic graph, or the no-graph mode.

The Graph-LM path is also prepared for larger corpora. These features make graph
building, graph storage, training, and training resume less limited to small
corpora. Scalability support includes batched graph statistics, reusable graph
cache, scalability metrics, DataLoader tuning, optional CUDA AMP, and more
complete training checkpoint resume.

Scalable run example for a medium or large corpus:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --graph-top-k 8 \
  --graph-build-batch-size 1000 \
  --graph-cache-dir runs/graph-cache \
  --dataloader-num-workers 2 \
  --dataloader-pin-memory \
  --amp \
  --output-dir runs/scalable-graph-lm
```

If training is interrupted, resume from the training checkpoint:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder gcn \
  --graph-cache-dir runs/graph-cache \
  --resume-from runs/scalable-graph-lm \
  --epochs 10 \
  --output-dir runs/scalable-graph-lm
```

Comparison of scalability features when enabled and disabled:

| Feature | When enabled | When disabled |
| --- | --- | --- |
| `--graph-build-batch-size` | Co-occurrence graph statistics are merged in smaller batches, making memory pressure more controllable for large corpora. | Text units are processed through the previous path; simpler and sufficient for small corpora. |
| `--graph-cache-dir` | The built graph is stored with a hash derived from the corpus, tokenizer, and graph config; the next run with the same settings reads from cache faster. | Every training run rebuilds the graph from scratch; acceptable for small quick ablations, but repeated time cost is high on large corpora. |
| `--no-reuse-graph-cache` | When used with cache, previous cache is ignored and a fresh artifact is built and replaced. | By default, a compatible cache is reused when found. |
| `--graph-top-k` | Limits the number of neighbors per node; the graph is sparser, RAM usage is lower, and training is faster, with the risk of dropping weak relations. | More edges are kept; graph signal is more complete but memory and compute cost are higher. |
| Disabling heavy relations such as `semantic_similarity` | Graph construction is faster and lighter for large vocabularies. | Semantic relations may improve quality, but pairwise comparison makes them expensive on large vocabularies. |
| `--dataloader-num-workers` | Batch preparation can be more parallel and the GPU waits less on DataLoader. | DataLoader stays single-process; simple and low-cost for small corpora. |
| `--dataloader-pin-memory` | When the device is actually CUDA, batch transfer to GPU is smoother. | Normal memory transfer is used; little effect on CPU/MPS. |
| `--amp` | Uses automatic mixed precision on CUDA, usually reducing GPU memory use and training time. | Training uses standard PyTorch precision; the most stable path for CPU and debugging. |
| `--resume-from` | `model.pt`, optimizer state, epoch, best validation, and RNG state are loaded from `training_state.pt`, and training continues from the next epoch. | Training starts from scratch, even if a previous checkpoint exists in the output directory. |
| `graph_scalability` metrics | `metrics.json` reports whether cache was enabled, whether there was a cache hit, graph build time, number of graph batches, and graph node/edge counts. | Training still works without these reports, but graph build and cache cost analysis is less transparent. |

The language-model checkpoint output is complete and is not limited to
`model.pt`:

```text
runs/graph-lm/
├── model.pt
├── training_state.pt
├── config.json
├── tokenizer.json
├── graph_config.json
├── graph.pt
├── graph_memory.pt
├── graph_memory_config.json
├── generation_config.json
├── metrics.json
└── corpus.txt
```

Important Graph-LM options:

| Option | Use |
| --- | --- |
| `--corpus` | Path to the raw Persian text file for LM training |
| `--tokenizer-type` | `word`, `char_chunk`, `bpe`, or `unigram` (default); `unigram` gives the lowest Persian OOV |
| `--unigram-num-pieces` | Target subword vocabulary size for the unigram tokenizer (default `8000`) |
| `--graph-encoder` | Select `gcn`, `gat`, `graphsage`, `rgcn`, or `none` for the no-graph baseline |
| `--graph-relations` | Select graph relations; if omitted, the default multi-relation preset (now including `dependency`) is used |
| `--semantic-method` | `semantic_similarity` scoring: `distributional` (PPMI-cosine, default) or `orthographic` (character n-gram) |
| `--linguistic-backend` | Backend for `dependency`/lemma relations: `auto` (Stanza if installed, else heuristic), `stanza`, or `heuristic` |
| `--relation-weights` | Set relation weights, such as `pmi=0.7,stem=0.5` |
| `--graph-relation-mode` | How `edge_type` is consumed in the encoder; one of `bias`, `embedding` (default), or `rgcn` |
| `--graph-pooling` | Optional pooling over subgraphs; one of `none`, `mean`, or `attention` |
| `--graph-node-importance` | Enable the internal scorer for detecting more important nodes |
| `--no-graph-node-type-embedding` | Disable learned node-type embeddings (non-token nodes start from zero) |
| `--fusion` | Select text/graph embedding fusion, such as `gated` |
| `--fusion-levels` | Select fusion levels, such as `token` or `token,sentence,subgraph` |
| `--graph-fusion-scale` | Strength multiplier for graph embeddings before fusion |
| `--graph-fusion-dropout` | Dropout over graph embeddings to reduce over-dependence on the graph |
| `--task-losses` | Select multi-task losses; by default all multi-task objectives are enabled |
| `--next-token-weight` and `--masked-token-weight` | Weights for causal language modeling and masked-token losses |
| `--edge-prediction-weight` and `--neighbor-prediction-weight` | Weights for edge and neighbor prediction losses in the graph |
| `--node-relation-weight` | Weight for predicting relation type from `edge_type` |
| `--graph-text-alignment-weight` and `--sentence-graph-alignment-weight` | Weights for text/graph representation alignment |
| `--mask-probability` | Probability of selecting non-padding tokens for masked-token prediction |
| `--negative-samples` | Number of negative samples per positive edge in graph losses |
| `--augmentation-ratio` | Ratio of augmented samples added to the train split; Low-Data Training Engine is enabled by default |
| `--token-dropout` and `--punctuation-dropout` | Text augmentation strength for low-data corpora |
| `--edge-dropout` and `--node-dropout` | Graph regularization during training to reduce memorization of fixed structure |
| `--subgraph-sampling-ratio` | Share of edges kept in each graph training view |
| `--contrastive-weight` | Weight of the contrastive consistency loss between graph views |
| `--no-text-augmentation` and `--no-curriculum` | Disable text augmentation or curriculum for ablation |
| `--checkpoint-metric` | Validation signal for best-checkpoint selection and early stopping: `next_token` (perplexity, default) or `total` multi-task loss |
| `--early-stopping-patience` and `--early-stopping-min-delta` | Control early stopping based on the `--checkpoint-metric` signal |
| `--max-grad-norm` | Gradient clipping limit in the trainer |
| `--graph-build-batch-size` | Batched graph-statistics construction for larger corpora |
| `--graph-cache-dir` | Directory for caching built graphs for reuse in later runs |
| `--no-reuse-graph-cache` | Force graph rebuilding even if a compatible cache exists |
| `--dataloader-num-workers` | Number of DataLoader workers in Graph-LM training |
| `--dataloader-pin-memory` | Enable pinned memory, useful when training on CUDA |
| `--amp` | Enable mixed precision on CUDA to reduce memory use and increase speed |
| `--resume-from` | Resume training from a checkpoint containing model and training state |
| `--output-dir` | Directory for checkpoints, configs, tokenizer, and reports |
| `--temperature` | Control generation randomness; lower values produce more conservative output |
| `--top-k` | Limit sampling to the top k most likely tokens |
| `--min-new-tokens` | Minimum number of new tokens in generation |
| `--repetition-penalty` | Reduce token repetition in `generate` output |
| `--graph-memory` | Control graph memory in `generate`; default is `on`, and `off` disables it |
| `--graph-memory-top-k` | Maximum number of memory nodes retrieved for the prompt |
| `--graph-memory-depth` | Neighbor expansion depth from prompt nodes in memory |
| `--graph-memory-max-edges` | Maximum retrieved subgraph edges for cost and noise control |
| `--graph-memory-min-score` | Minimum node score required for entry into the memory subgraph |
| `--graph-memory-relation-weights` | Relation weights during retrieval, such as `pmi=0.5,word_document=1.2` |
| `--graph-memory-report-path` | Save a JSON report of retrieved memory nodes, relations, and coverage |

In the classification path, if you do not pass `--dataset`, a small internal
experiment runs as a smoke test for the installation and base model. If you pass
a dataset, the CLI reads texts, builds a word-document graph, trains one of
`gcn`, `graphsage`, or `gat`, and writes train/validation/test reports.

Run the small internal experiment:

```bash
rgnn-cli --model gcn --device cpu
```

Run the pipeline on a dataset with `text` and `label` columns:

```bash
rgnn-cli \
  --dataset benchmarks/persian_text_classification.csv \
  --text-column text \
  --label-column label \
  --model gcn \
  --epochs 50 \
  --device cpu \
  --output-dir runs/news-gcn \
  --save-model runs/news-gcn/model.pt
```

Run the same path on GPU:

```bash
rgnn-cli \
  --dataset benchmarks/persian_text_classification.csv \
  --model gat \
  --device cuda \
  --epochs 50 \
  --output-dir runs/news-gat-gpu
```

The dataset input can be CSV, TSV, or JSONL. By default, `text` and `label`
columns are read, but you can change the column names:

```bash
rgnn-cli \
  --dataset data/comments.jsonl \
  --dataset-format jsonl \
  --text-column comment \
  --label-column sentiment \
  --model graphsage \
  --output-dir runs/comments-graphsage
```

All options can also be passed through a JSON file. File keys are the same as
CLI option names without `--` and with underscores:

```json
{
  "dataset": "benchmarks/persian_text_classification.csv",
  "model": "gat",
  "epochs": 50,
  "hidden_dim": 16,
  "learning_rate": 0.005,
  "dropout": 0.3,
  "device": "cpu",
  "output_dir": "runs/news-gat"
}
```

```bash
rgnn-cli --config config.json
```

By default, run output is saved to `OUTPUT_DIR/metrics.json`. It includes the
dataset name, model, device, label mapping, split sizes, train/validation/test
accuracy and macro-F1, and final loss. Use `--report-path` to choose an exact
report path. Use `--save-model` to save the PyTorch checkpoint, model type,
dimensions, label mapping, and metrics. For higher-level pipeline persistence
and full save/load, use `TextGraphClassifier.save` in the Python API.

Important CLI options:

| Option | Use |
| --- | --- |
| `--dataset` | Run real training/evaluation on CSV, TSV, or JSONL |
| `--dataset-format` | Set input format; `auto` detects it from the file extension |
| `--text-column` and `--label-column` | Text and label field names in the dataset |
| `--train-ratio`, `--val-ratio`, `--test-ratio` | Train, validation, and test split ratios |
| `--model` | Select architecture from `gcn`, `graphsage`, and `gat` |
| `--epochs`, `--hidden-dim`, `--learning-rate`, `--weight-decay`, `--dropout` | Model training settings |
| `--gat-heads` | Number of attention heads for the `gat` model |
| `--window-size` and `--min-count` | Control text graph construction with co-occurrence window and minimum word count |
| `--device` | Run on CPU or CUDA; if CUDA is unavailable, the CLI falls back to CPU |
| `--output-dir` and `--report-path` | Paths for saving run reports |
| `--save-model` | Save the trained model checkpoint |
| `--config` | Read these options from a JSON file |
| `--log-level` and `--log-to` | Control logging and optional connection to `wandb` or `mlflow` in the internal experiment |

To see the full option list:

```bash
rgnn-cli --help
```

## Practical Examples

### Summarization

```python
from rakhshai_graph_nlp.tasks.summarization import (
    gat_summarise,
    textrank_summarise,
)

text = (
    "هوش مصنوعی در سال‌های اخیر رشد زیادی داشته است. "
    "بسیاری از شرکت‌ها از مدل‌های زبانی برای تحلیل متن استفاده می‌کنند. "
    "در زبان فارسی هم ابزارهای NLP در حال بهتر شدن هستند."
)

print(textrank_summarise(text, top_k=2))
print(gat_summarise(text, top_k=2))
```

### Content Recommendation

```python
from rakhshai_graph_nlp.tasks.recommendation import recommend_similar

query = "این نمایشگاه جذاب است."
docs = [
    "این یک خبر سیاسی است و درباره انتخابات صحبت می‌کند.",
    "تیم فوتبال امروز بازی مهمی دارد.",
    "نمایشگاه جدید هنری با آثار نقاشان جوان افتتاح شد.",
]

print(recommend_similar(query, docs, top_k=2))
```

### Hate-Speech Detection

```python
from rakhshai_graph_nlp.tasks.hate_speech import (
    HateSpeechDetector,
    contains_hate_speech,
)

hate_terms = ["نفرت", "لعنت"]
print(contains_hate_speech("این پیام حاوی نفرت است", hate_terms))

texts = [
    "متن آرام و محترمانه",
    "گفتگوی خوب و عادی",
    "توهین بد و نفرت",
    "پیام بد و سمی",
]
labels = [False, False, True, True]

detector = HateSpeechDetector(num_epochs=20)
detector.fit(texts, labels)
print(detector.predict(["پیام حاوی نفرت"]))
detector.save("runs/hate-detector")
```

### Semantic Graph

```python
from rakhshai_graph_nlp.graphs.semantic import build_semantic_graph

words = ["گربه", "سگ", "ماشین"]
relations = {"گربه": ["سگ"]}
embeddings = {
    "گربه": [1.0, 0.0],
    "سگ": [0.9, 0.1],
    "ماشین": [0.0, 1.0],
}

graph = build_semantic_graph(
    words,
    relations=relations,
    embedding_lookup=embeddings,
    similarity_threshold=0.8,
)
print(graph.adjacency)
```

### Semantic Graph With FarsNet

FarsNet is a Persian WordNet and can be a stronger source for building semantic
relations between Persian words. Because of access and licensing issues, the
FarsNet database is not copied into this repository. The project does include a
ready loader, so if you have FarsNet output as JSON/CSV/TSV, you can build a
semantic graph from it directly.

Usable JSON example:

```json
{
  "synsets": [
    {"id": "s1", "lemmas": ["ماشین", "خودرو", "اتومبیل"]},
    {"id": "s2", "lemmas": ["پزشک", "دکتر"]}
  ]
}
```

Use in the project:

```python
from rakhshai_graph_nlp.graphs.semantic import (
    build_semantic_graph_from_farsnet,
)

words = ["ماشین", "خودرو", "پزشک", "دکتر"]

graph = build_semantic_graph_from_farsnet(
    words,
    farsnet_path="data/farsnet.json",
)
print(graph.adjacency)
```

If your FarsNet output is CSV/TSV, two common forms are supported:

```csv
source,target
ماشین,خودرو
پزشک,دکتر
```

or:

```csv
synset_id,word
s1,ماشین
s1,خودرو
s2,پزشک
s2,دکتر
```

## Project Structure

```text
rakhshai_graph_nlp/
├── features/        # tokenization, preprocessing, and conversion to PyG
├── graphs/          # graph construction functions
├── models/          # GNN models
├── lm/              # PersianTokenizer, LMDataset, GraphCausalLM, Graph Memory, LMTrainer, and generate
├── llm/             # high-level native LLM workflows built on the Graph-LM engine
├── article_llm/     # compatibility alias for the native article workflow
├── tasks/           # practical tasks
├── explain/         # initial explanation tools
├── metrics.py       # evaluation metrics
├── cli.py           # command-line interface
└── utils/           # helpers

benchmarks/          # small reproducible datasets
docs/                # MkDocs documentation
tests/               # unit and end-to-end tests
```

## Validation and Quality

RGN ships with unit, integration, CLI, checkpoint, graph, and end-to-end
tests. Small datasets under `benchmarks/` and `data/` are included for
implementation checks and reproducible local runs.

```bash
python -m pytest
```

Additional architecture, benchmark methodology, and evaluation details are
available in the technical documentation:

- [Graph-LM V2](docs/graph_lm_v2.md)
- [Graph Reasoning Core](docs/graph_reasoning_core.md)
- [Low-Data Training Engine](docs/low_data_training_engine.md)
- [MCP evaluation report](docs/mcp_single_poem_evaluation.md)

## Deployment Notes

- Generation quality depends on the training corpus, tokenizer, checkpoint,
  decoding settings, and selected graph encoder and fusion path.
- `build_text_graph` uses a dense matrix and may require a more scalable graph
  construction path for very large collections.
- Haman 125M is a base article-generation checkpoint, not a chat assistant or
  a factual authority. Review generated content before publication.
- Train and evaluate sensitive classifiers, including hate-speech detection,
  with representative data, error analysis, and bias controls before use.
- Keep secrets outside the repository and review the policies of every
  external service connected through MCP or optional integrations.

## Documentation

| Topic | Guide |
| --- | --- |
| Documentation index | [`docs/index.md`](docs/index.md) |
| Stable Python API | [`docs/api.md`](docs/api.md) |
| Python API walkthrough | [`docs/api_usage.md`](docs/api_usage.md) |
| Native LLM workflows | [`docs/llm.md`](docs/llm.md) |
| Persian article workflow | [`docs/article_llm.md`](docs/article_llm.md) |
| Graph-LM architecture | [`docs/graph_lm_v2.md`](docs/graph_lm_v2.md) |
| Graph Reasoning Core | [`docs/graph_reasoning_core.md`](docs/graph_reasoning_core.md) |
| Low-Data Training Engine | [`docs/low_data_training_engine.md`](docs/low_data_training_engine.md) |
| Persian tokenizer | [`docs/persian_tokenizer.md`](docs/persian_tokenizer.md) |
| Multi-relation graph | [`docs/multi_relation_persian_graph.md`](docs/multi_relation_persian_graph.md) |
| MCP integration | [`docs/mcp.md`](docs/mcp.md) |

## About the Developer

RGN is created and maintained by the [RakhshAI](https://rakhshai.com/)
team at [Aria Haman Mehr Parseh](https://ariahaman.ir/en/), an Iranian
knowledge-based software company.

### Knowledge-Based Certification

Rakhshai Graph-based NLP (RGN), developed by Aria Haman Mehr Parseh, has
received knowledge-based certification in Iran.
The company can be verified in the
[official inquiry system](https://pub.daneshbonyan.ir/dashboard) by national ID.

| Field | Value |
| --- | --- |
| Company | Aria Haman Mehr Parseh |
| National ID | `14009192677` |
| Product | Rakhshai Graph-based NLP (RGN) |
| Field | Persian NLP, graph-based text modeling, and AI infrastructure |

## Licenses

- RGN source code: [MIT](LICENSE)
- Haman Persian Article Graph-LLM 125M:
  [Haman Model License 1.0](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m/blob/main/LICENSE)
- Haman Persian Wikipedia Articles 186K: CC BY-SA 4.0; see the
  [dataset card](https://huggingface.co/datasets/aria-haman/haman-fa-wikipedia-articles-186k)

For technical questions and bug reports, use
[GitHub Issues](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/issues).

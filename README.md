English | [فارسی](README.fa.md)

# Rakhshai Graph-based NLP V2

[![CI](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml)

**Rakhshai Graph-based NLP V2**: from raw Persian text to graph neural networks
and Persian Graph-LM.

Rakhshai is the first integrated graph-oriented NLP framework for Persian. It
turns raw Persian text into graphs that can be analyzed, trained on, and used in
downstream models. The project models relationships between words, documents,
and linguistic structures as graphs, and provides graph neural network models
such as GCN, GraphSAGE, and GAT for tasks such as text classification,
summarization, content recommendation, and text analysis.

Rakhshai is designed to shorten the path to building Graph-based NLP pipelines
for Persian: from Persian preprocessing and textual, syntactic, and semantic
graph construction to training, evaluation, prediction on new text, and model
save/load. Its goal is to create a practical bridge between Persian language
processing, graph modeling, and deep graph learning.

## Knowledge-Based Certification

**Graph-Based NLP Service Library (Rakhshai-Graph-based-NLP)**, developed by **Aria Haman Mehr Parseh**, has received knowledge-based certification in Iran.

The company’s knowledge-based status can be verified through the official knowledge-based companies inquiry system by searching the company’s national ID.

- **Official inquiry link:** [Knowledge-Based Companies Inquiry System](https://pub.daneshbonyan.ir/dashboard)
- **Company national ID:** `14009192677`

> Company: Aria Haman Mehr Parse  
> National ID: `14009192677`  
> Product: Graph-Based NLP Service Library (Rakhshai-Graph-based-NLP)  
> Field: Persian NLP, graph-based text modeling, and AI infrastructure

## Rakhshai V2
Rakhshai V2 is no longer only a graph classification toolkit; it also includes a
real **Persian Graph-LM** path. In this path, Persian text is
converted into numeric tokens, a word co-occurrence graph is built from the same
corpus, a GNN produces graph embeddings, and token embeddings are combined with
graph embeddings through **Gated Graph-Token Fusion** inside a modern
decoder-only Transformer (RoPE positions, SwiGLU feed-forward, RMSNorm pre-norm,
and a KV cache for generation). The model output is `batch x sequence x
vocab_size`, designed for next-token prediction and Persian text generation.

An early experiment on an expanded Persian corpus reported lower perplexity for
the `Graph-LM / GCN + gated` path than for a no-graph baseline, but that result
was produced before two implementation fixes (embedding initialization and
perplexity computed from the next-token loss only). After the fixes, the graph
fusion was redesigned with **zero-initialized gating**: the untrained model is
exactly equivalent to the no-graph baseline, and graph information only enters
through a learnable `tanh(alpha)` gate once training finds it useful. With this
design, the graph path matches the baseline on the small evaluation corpus
(no measurable advantage yet, and no harm either). This is a provisional,
small-data observation, not a final verdict: Graph-LM is designed to exploit
rich structures and wide-ranging relations, and a fair evaluation of that
capacity requires training and testing on much larger Persian corpora. The
bundled small corpus is suitable only for smoke tests and implementation-health
checks.

This open-source library is created and maintained by the
[RakhshAI](https://rakhshai.com/) development team, affiliated with Aria Haman
Mehr Parseh, and is released under the MIT license.

## Highlights In Plain Language

- **The first integrated graph-oriented NLP framework for Persian:** Rakhshai
  provides a complete path from raw Persian text to graph construction, model
  training, evaluation, prediction, and pipeline save/load.
- **Dedicated GCN, GraphSAGE, and GAT implementations for Persian Graph-based
  NLP:** Rakhshai provides three major graph neural network models in an
  executable, integrated framework for learning on graphs built from Persian
  text, with support for training, evaluation, prediction, and pipeline
  save/load.
- **Ready-to-use text classification:** With `TextGraphClassifier`, you can
  provide texts and labels, train a model, evaluate it, and predict labels for
  new text.
- **Complete model and pipeline persistence:** The saved artifact includes more
  than model weights; it also stores the vocabulary, label mapping, and graph
  settings.
- **GPU support for graph models:** If CUDA and an NVIDIA GPU are available,
  PyTorch Geometric paths can run on GPU.
- **Multiple graph types for Persian text:** Word co-occurrence graphs,
  word-document graphs, document similarity graphs, dependency graphs, and
  semantic graphs with FarsNet support.
- **Several ready-made tasks:** Classification, summarization, content
  recommendation, hate-speech detection, and network analysis.
- **Command-line interface:** `rgnn-cli` lets you train and evaluate on
  CSV/TSV/JSONL files without writing much code.
- **Persian Graph-LM with a Rakhshai-specific architecture:** The `lm-train`
  path brings together a Persian tokenizer, graph builder, GNN encoder, gated
  graph-token fusion, a Transformer causal LM, LM-specific trainer, perplexity,
  complete checkpointing with sparse graph artifacts, and text generation.
- **Modern Transformer decoder:** The causal LM uses rotary position embeddings
  (RoPE), a SwiGLU feed-forward, RMSNorm with pre-norm residuals, and a KV cache
  during generation. Each can be switched back to the classic variant
  (`position_encoding`, `ffn_type`, `norm_type` on `GraphLMConfig`).
- **Persian-aware tokenization:** Punctuation is tokenized separately (Persian
  marks are no longer glued to words and ASCII marks are no longer dropped),
  Persian decimal/thousands separators are normalized, hamza folding and ezafe
  handling are configurable, and `unigram` is a genuine Unigram LM tokenizer.
- **Graph Reasoning Core for multi-relation graphs:** The graph encoder can use
  multi-relation graph relation IDs through `bias`, `embedding`, or `rgcn`
  modes, use `RGCN` for relation-aware message passing, and optionally use node
  importance and subgraph pooling.
- **Adaptive Graph-Text Fusion:** The model can control text/graph fusion at the
  `token`, `sentence`, and `subgraph` levels. Graph usage strength is controlled
  by scale/dropout, and gate statistics are stored in `metrics.json` so you can
  inspect where the model actually used graph information.
- **Low-Data Training Engine:** The Graph-LM path runs by default with text
  augmentation, node/edge dropout, subgraph sampling, contrastive learning,
  curriculum learning, early stopping, and overfitting reports so it memorizes
  less and generalizes better on small corpora.
- **Graph Memory for generation time:** Graph checkpoints now store a separate
  graph memory. The `generate` command retrieves prompt-related nodes and
  subgraphs by default and passes only that limited subgraph into fusion, so the
  model uses prompt-relevant graph memory instead of the entire graph.
- **No-graph baseline for fair comparison:** With `--graph-encoder none`, you
  can train the same Transformer causal LM without GNN and fusion, then measure
  the true effect of the graph with validation loss and perplexity.
- **Controllable text generation:** The `generate` command supports options
  such as `--temperature`, `--top-k`, `--min-new-tokens`, and
  `--repetition-penalty`.

## Rakhshai Graph-LM Technical Signature

The new Graph-LM architecture in Rakhshai implements this path:

```text
Persian Text
→ PersianTokenizer
→ LM Dataset
→ Multi-Relation Persian Graph
→ Rakhshai Graph Encoder (GCN / GraphSAGE / GAT / RGCN)
→ Adaptive Graph-Text Fusion
→ Low-Data Training Engine
→ Prompt-aware Graph Memory
→ Transformer Causal LM
→ Text Generation
```

The technical signature of this part of the project is:

```text
Rakhshai Graph Encoder
+
Adaptive Graph-Text Fusion
+
Low-Data Training Engine
+
Persian Causal LM
```

In simple terms, Rakhshai can use graph relationships between words instead of
acting as a purely sequential language model. When building the final embedding
for each token, the model learns how much to trust the text embedding and how
much to trust the graph embedding. The combination is not fixed by hand; it is
learned through a gate. This gate can operate at several levels: token level for
each sequence position, sentence level for controlling overall graph strength in
the context, and subgraph level for injecting a summary of non-token nodes such
as documents or topics.

> **Checkpoint compatibility note.** The decoder was modernized (RoPE, SwiGLU,
> RMSNorm, pre-norm) and the `unigram` tokenizer is now a real Unigram LM, while
> the default `ezafe_mode` is now `marker`. Checkpoints saved before these
> changes still load (`from_pretrained` uses `strict=False`) but their
> transformer weights no longer match the new layout, so retrain to use the new
> architecture. Tokenizer configs saved before `ezafe_mode` existed keep the old
> `collapse` behaviour on load.

## Initial Graph-LM Result

To validate the new path, an expanded Persian corpus was created and two models
were compared under similar conditions: the graph model `Graph-LM / GCN + gated`
and a no-graph baseline. The original numbers published here (perplexity
1344.77 vs 1634.99, about 18% in favour of the graph model) were produced
before two implementation fixes: embedding initialization with a small std, and
perplexity computed from the next-token cross-entropy only instead of the total
multi-task loss. Rerunning the identical experiment (same corpus,
hyperparameters, and seeds 0-2) with the fixed code gives:

| Model | best perplexity (mean ± std, 3 seeds) |
| --- | ---: |
| `Baseline / no graph` | 121.7 ± 4.6 |
| `Graph-LM / GCN + gated`, original fusion | 179.2 ± 18.5 |
| `Graph-LM / GCN + gated`, zero-init gating | 120.4 ± 24.4 |

With the fixed code, the original input-replacing fusion actively hurt the
language model; the earlier 18% advantage was an artifact of the implementation
issues, not of graph fusion. After redesigning the fusion with zero-initialized
gating, the graph model reaches the same perplexity as the no-graph baseline.
The learned gate (`tanh(alpha)` reported in `metrics.json` as
`token_alpha_tanh`) stays close to zero on this corpus — the model finds little
useful graph signal in 71 sentences, which is expected.

**Scope of this benchmark.** These numbers are a provisional, small-data
observation. No conclusion about the value of graph fusion for Persian language
modeling should be drawn from this corpus: it exists for smoke testing,
implementation-health checks, and quick comparisons of settings. Graph-LM is
designed to exploit rich structures and wide-ranging lexical relations, and
judging that capacity fairly requires training and evaluation on substantially
larger Persian corpora.

**Design philosophy.** Rakhshai develops and evaluates a native, self-contained
Persian language-model architecture. For now the project intentionally does not
use external pretrained language models, knowledge distillation from other
models, pretrained embeddings, or LLM-generated synthetic data, so that the
capabilities and limits of the Graph-LM architecture itself can be measured
transparently, without borrowing knowledge from other models.

Sample generated output with a corpus smaller than large language models:

```text
prompt: امروز در تهران
output: امروز در تهران باران آرامی بارید و خیابان‌ها خلوت‌تر از روزهای گذشته بودند ...
```

For a more scientific result, repeat the same experiment with several seeds,
several graph encoders such as `GCN`, `GAT`, `GraphSAGE`, and `RGCN`, several
relation-aware modes such as `bias` and `embedding`, and several fusion methods
such as `gated` and `additive`.

## Ready Dataset Sample For Testing Graph-LM

A ready Persian Wikipedia sample for quick Graph-LM testing is available at:

```text
data/wiki_fa_50k.txt
```

If you do not want to download a dataset separately, you can train and generate
directly with this file. To rebuild the same file or create a larger sample, use:

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

In this command, Graph Memory is enabled by default. If the checkpoint contains
`graph_memory.pt`, that memory is loaded. If it does not, and `corpus.txt` exists
inside the checkpoint, the memory is rebuilt from the corpus and
`graph_config.json`. To disable memory and return to the previous generation
behavior:

```bash
rgnn-cli generate \
  --model runs/wiki-graph-lm \
  --prompt "امروز در تهران" \
  --max-new-tokens 100 \
  --graph-memory off
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

| Component | Description |
| --- | --- |
| `TextGraphClassifier` | Main class for training, evaluation, prediction, saving, and loading text classification models |
| `build_text_graph` | Builds a word-document graph with PMI and TF-IDF for TextGCN |
| `build_cooccurrence_graph` | Builds word co-occurrence graphs with a word window |
| `build_document_graph` | Builds document similarity graphs with TF-IDF or embeddings |
| `build_dependency_graph` | Builds dependency graphs with Stanza |
| `build_semantic_graph` | Builds semantic graphs from lexical relations and embedding similarity |
| `build_semantic_graph_from_farsnet` | Builds semantic graphs from FarsNet JSON/CSV/TSV output |
| `load_farsnet_relations` | Loads FarsNet relations from a file and converts them into graph-usable relations |
| `PersianTokenizer` | Numeric LM tokenizer with half-space support, Persian cleanup, standalone punctuation tokens, numeric-separator and configurable hamza/ezafe normalization, and `word`/`char_chunk`/`bpe`/`unigram` modes |
| `LMDataset` | Prepares `input_ids` and `target_ids` for next-token prediction |
| `build_graph_lm_graph` | Builds a word co-occurrence graph from a corpus for Graph-LM |
| `GraphCausalLM` | Persian language model with GNN encoder, gated graph-token fusion, and a modern decoder-only Transformer (RoPE, SwiGLU, RMSNorm, KV-cache generation) |
| `RakhshaiGraphEncoder` | Graph-LM graph core with `gcn`, `graphsage`, `gat`, `rgcn`, and relation-aware encoding support |
| `--graph-encoder none` | No-graph baseline for comparing against Graph-LM and measuring the true effect of GNN/fusion |
| `LMTrainer` | LM-specific trainer with validation loss, perplexity, and complete checkpointing |
| `GCNClassifier` | GCN model for node classification |
| `GraphSAGEClassifier` | GraphSAGE model for neighborhood-based learning |
| `GATClassifier` | GAT model with graph attention |
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

### Running Tests

```bash
python -m pip install pytest
python -m pytest
```

## Quick Start

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

`rgnn-cli` has two main paths: the older stable graph-based Persian text
classification path, and the newer Graph-LM path for training graph-oriented
Persian language models. If you do not provide a subcommand, the classification
path runs. For LM, use `lm-train` and `generate`.

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
cooccurrence + pmi + stem + subword + word_document + topic_document
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
| `--graph-encoder` | Select `gcn`, `gat`, `graphsage`, `rgcn`, or `none` for the no-graph baseline |
| `--graph-relations` | Select graph relations; if omitted, the default multi-relation preset is used |
| `--relation-weights` | Set relation weights, such as `pmi=0.7,stem=0.5` |
| `--graph-relation-mode` | How `edge_type` is consumed in the encoder; one of `bias`, `embedding`, or `rgcn` |
| `--graph-pooling` | Optional pooling over subgraphs; one of `none`, `mean`, or `attention` |
| `--graph-node-importance` | Enable the internal scorer for detecting more important nodes |
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
| `--early-stopping-patience` and `--early-stopping-min-delta` | Control early stopping based on validation loss |
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
├── tasks/           # practical tasks
├── explain/         # initial explanation tools
├── metrics.py       # evaluation metrics
├── cli.py           # command-line interface
└── utils/           # helpers

benchmarks/          # small reproducible datasets
docs/                # MkDocs documentation
tests/               # unit and end-to-end tests
```

## Initial Graph-LM Benchmark

To test the Graph-LM path, an expanded Persian corpus at
`data/expanded_persian_lm.txt` was used. The corpus includes sentences about
cities, politics, education, economy, sports, art, and the Graph-LM architecture
itself, so political, weather, education, and graph-modeling lexical
relationships are visible in the co-occurrence graph.

Comparison with the fixed implementation (seeds 0-2; the pre-fix artifacts are
kept in `runs/compare-graph-lm` and `runs/compare-baseline-lm` for reference,
but their metrics are not valid):

| Model | best perplexity (mean ± std) | Output path |
| --- | ---: | --- |
| `Baseline / no graph` | 121.7 ± 4.6 | `runs/compare-fixed/baseline-s*` |
| `Graph-LM / GCN + gated`, original fusion | 179.2 ± 18.5 | `runs/compare-fixed/graph-lm-s*` |
| `Graph-LM / GCN + gated`, zero-init gating | 120.4 ± 24.4 | `runs/compare-fixed/graph-lm-zeroinit-s*` |

This benchmark is a smoke test of the implementation, not an evaluation of
model quality; see "Scope of this benchmark" above.

Sample generated text with the demo model:

```text
prompt: امروز در تهران
output: امروز در تهران باران آرامی بارید و خیابان‌ها خلوت‌تر از روزهای گذشته بودند ...
```

This small benchmark proves the path and provides an initial comparison.

## Reproducible Benchmarks

A small Persian benchmark is available at
`benchmarks/persian_text_classification.csv` so the full path of graph
construction, training, evaluation, and report saving can be checked quickly.
The dataset contains 24 short news texts in three classes, `politics`, `sports`,
and `art`. It is useful for smoke testing and comparing settings, not for making
final model-quality claims.

Reproducible CPU runs with `seed=0`:

| Model | validation accuracy | test accuracy | test macro-F1 | Report path |
| --- | ---: | ---: | ---: | --- |
| `gcn` | 1.00 | 0.75 | 0.60 | `runs/benchmarks/persian-classification-gcn/metrics.json` |
| `graphsage` | 1.00 | 1.00 | 1.00 | `runs/benchmarks/persian-classification-graphsage/metrics.json` |
| `gat` | 1.00 | 1.00 | 1.00 | `runs/benchmarks/persian-classification-gat/metrics.json` |

Example `metrics.json` output:

```json
{
  "dataset": "benchmarks/persian_text_classification.csv",
  "model": "gcn",
  "device": "cpu",
  "num_documents": 24,
  "num_nodes": 175,
  "num_classes": 3,
  "splits": {
    "train": {"count": 16, "accuracy": 1.0, "macro_f1": 1.0},
    "validation": {"count": 4, "accuracy": 1.0, "macro_f1": 1.0},
    "test": {"count": 4, "accuracy": 0.75, "macro_f1": 0.6}
  }
}
```

Run the same benchmark yourself:

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

Compare the three main models:

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

After running, each model report is stored in `metrics.json`. To see a summary:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("runs/benchmarks").glob("*/metrics.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    test = report["splits"]["test"]
    print(
        f"{report['model']:10s} "
        f"test_acc={test['accuracy']:.3f} "
        f"test_macro_f1={test['macro_f1']:.3f} "
        f"report={path}"
    )
PY
```

For a more serious evaluation, run the same pipeline on known Persian
benchmarks. Convert the dataset to CSV/TSV/JSONL with `text` and `label` columns,
then use the same CLI command.

| Benchmark | Common use | Preparation note |
| --- | --- | --- |
| [Hamshahri Corpus](https://en.wikipedia.org/wiki/Hamshahri_Corpus) | Persian news classification and information retrieval | Put article text in `text` and news category in `label`. |
| [SnappFood Persian Sentiment](https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis) | User review sentiment analysis | Put review text in `text` and positive/negative label in `label`. |
| [SentiPers](https://www.researchgate.net/publication/322694676_SentiPers_A_Sentiment_Analysis_Corpus_for_Persian) | Persian sentiment analysis | If you have multiple polarity levels, convert them into stable text labels. |
| [Pars-ABSA](https://arxiv.org/abs/1908.01815) | Persian aspect-based sentiment analysis | For this CLI, convert each sample to one general label, or place each aspect in a separate row. |

Example run on your own dataset:

```bash
python -m rakhshai_graph_nlp.cli \
  --dataset data/my_persian_dataset.csv \
  --text-column text \
  --label-column label \
  --model gat \
  --epochs 80 \
  --hidden-dim 16 \
  --learning-rate 0.005 \
  --dropout 0.3 \
  --seed 42 \
  --device cuda \
  --output-dir runs/my-persian-benchmark-gat
```

If CUDA is not available on your system, use `--device cpu`. For fair model
comparison, keep split, seed, epoch count, and preprocessing fixed, and report
macro-F1 in addition to accuracy, especially when classes are imbalanced.

## Current Limitations

- `build_text_graph` uses a dense matrix and may be memory-limited for very
  large collections.
- The Graph-LM path is currently experimental. The initial benchmark shows that
  the pipeline works and that the graph has an early positive effect; it is not
  a final-quality claim comparable to large LLMs.
- Graph-LM text generation quality depends on corpus size and diversity, number
  of epochs, tokenizer, sampling settings, and graph encoder/fusion selection.
- Model output quality depends on data quality, tokenization, and training
  settings.
- `HateSpeechDetector` must be trained with real data, error analysis, and bias
  control before being used in sensitive applications.
- For stronger semantic graphs, use FarsNet, reliable lexical relations, or
  high-quality Persian embeddings.

## Inspiration Sources

- **TextGCN:** Word-document graphs with PMI and TF-IDF for text
  classification.
- **GCN / GraphSAGE / GAT:** Graph neural network models for propagating and
  aggregating information in graphs.
- **TextRank:** Ranking sentences or words with PageRank over a similarity
  graph.
- **Causal Language Modeling:** Training a model to predict the next token and
  generate text.
- **Transformer Decoder:** The sequential language-model core that also uses
  graph embeddings through graph-token fusion.
- **Stanza:** Linguistic analysis tools for tokenization, lemmatization, and
  dependency parsing.

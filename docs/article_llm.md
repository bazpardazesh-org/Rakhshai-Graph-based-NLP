# Native Persian Article LLM

Rakhshai includes an article-writing layer on top of the existing Graph-LM
engine. It trains a native Persian article generator from your own corpus
without external pretrained language models, distillation, pretrained embeddings
or LLM-generated synthetic data.

The public workflow namespace is `rakhshai_graph_nlp.llm.article`.
`rakhshai_graph_nlp.lm` remains the lower-level Graph-LM engine used by this
workflow. `rakhshai_graph_nlp.article_llm` is kept as a compatibility alias.

This feature is a pipeline, not a quality guarantee. Article quality depends on
corpus size and quality, training time, hardware, tokenizer choice, graph
encoder, fusion mode and generation settings.

## Architecture Boundary

The article workflow is intentionally separate from the Graph-LM engine:

| Layer | Responsibilities |
| --- | --- |
| `rakhshai_graph_nlp.llm.article` | Article dataset normalization, prompt-completion formatting, article-specific training defaults, structured Markdown/JSON generation and CLI command wiring. |
| `rakhshai_graph_nlp.lm` | Tokenization, Graph-LM graph construction, model definition, training loop, graph memory retrieval and generic text generation. |

Use `rakhshai_graph_nlp.llm.article` when the goal is to build or run the
Persian article workflow. Use `rakhshai_graph_nlp.lm` when you need the raw
Graph-LM building blocks for a custom model, experiment or future workflow.

The old `rakhshai_graph_nlp.article_llm` import path is still available as a
compatibility alias, but new code and documentation should prefer
`rakhshai_graph_nlp.llm.article`.

## End-to-End Workflow

The complete native article LLM path is:

1. Prepare raw Persian article data as TXT, JSONL, CSV or TSV.
2. Run `article-prepare` to normalize records, filter short records and write
   `corpus.txt`, `train.txt`, `validation.txt`, `prepared_articles.jsonl`,
   `rejected_records.jsonl` and `manifest.json`.
3. Run `article-audit` before expensive training. Use the audit to catch
   duplicate-heavy corpora, non-Persian noise, missing metadata/source coverage
   and tokenizer problems.
4. Run `article-train` against the prepared `corpus.txt`. When sibling
   `train.txt` and `validation.txt` files exist, the trainer uses those exact
   prepared splits instead of splitting the full corpus again.
5. Inspect the checkpoint artifacts and metrics. The main files are `model.pt`,
   `tokenizer.json`, `config.json`, `generation_config.json`, `metrics.json`,
   `article_llm_config.json` and `corpus.txt`; graph-enabled runs also write
   `graph.pt`, `graph_config.json` and graph-memory artifacts.
6. Generate from the trained checkpoint with `article-generate` or load it with
   `ArticleGenerationConfig` and `generate_persian_article`.
7. Optionally run `article-ablation` to compare no-graph, graph-encoder,
   graph-scope and relation-group variants under the same native recipe.

## Dataset Format

Use `article-prepare` with TXT, JSONL, CSV or TSV input. Only `body` is
required; `title`, `summary`, `keywords` and `metadata` are optional.

```json
{"title":"اقتصاد دیجیتال","body":"متن کامل مقاله فارسی...","summary":"خلاصه کوتاه","keywords":["اقتصاد","فناوری"],"metadata":{"source":"local"}}
```

```bash
rgnn-cli article-prepare \
  --input data/persian_articles.jsonl \
  --output-dir runs/articles-prepared \
  --input-format jsonl \
  --min-body-chars 400 \
  --validation-ratio 0.1
```

The output directory contains `corpus.txt`, `train.txt`, `validation.txt`,
`prepared_articles.jsonl`, `rejected_records.jsonl` and `manifest.json`.
When `article-train` receives this prepared `corpus.txt`, it uses the sibling
`train.txt` and `validation.txt` files instead of splitting the full corpus
again. If the prepared `validation.txt` is empty, the trainer falls back to its
own `--validation-ratio` split instead of training without validation.

Every training format starts each corpus line with the same conditioning
header used by `article-generate` prompts (`موضوع مقاله`, `مخاطب`, `لحن`,
`ساختار`, `مقاله:`), so the fields you pass at generation time are tokens the
model actually saw during training. Per-record `metadata.audience` and
`metadata.tone` values override the corpus-wide `--prompt-audience` and
`--prompt-tone` defaults, which lets one corpus teach multiple tones.

For Persian Wikipedia-style JSONL records with `title` and `text` fields, use
the prompt-completion training format. It turns the title into the requested
topic and the article text into the target answer shape used by
`article-generate`:

```bash
rgnn-cli article-prepare \
  --input data/fa_wikipedia_articles.jsonl \
  --output-dir runs/fa-wiki-articles-prepared \
  --input-format jsonl \
  --training-format wikipedia_prompt \
  --min-body-chars 400
```

## Audit Before Training

Use `article-audit` before expensive native training runs. It does not use
external LMs, pretrained embeddings, distillation, synthetic data or an external
LLM judge. The audit records Persian character ratios, half-space indicators,
exact and near-duplicate bodies, metadata/source coverage and tokenizer
benchmarks.

```bash
rgnn-cli article-audit \
  --input data/persian_articles.jsonl \
  --output-dir runs/articles-audit \
  --input-format jsonl \
  --min-body-chars 400 \
  --tokenizer-types word bpe unigram
```

The default tokenizer benchmark is lightweight and reports token counts,
tokens-per-character, unknown-token rate, continuation-piece rate and
half-space token rate. To compare tokenizers with a fixed native training
budget, enable the optional probe:

```bash
rgnn-cli article-audit \
  --input data/persian_articles.jsonl \
  --output-dir runs/articles-audit \
  --tokenizer-types word bpe unigram \
  --tokenizer-probe-epochs 1 \
  --tokenizer-probe-block-size 128
```

The audit writes machine-readable reports under the chosen output directory.
Treat it as a preflight check: it does not prove model quality, but it makes
data issues visible before spending GPU time.

## Train With CUDA

The article trainer wraps `train_graph_lm` and exposes the important Graph-LM
controls for larger runs:

```bash
rgnn-cli article-train \
  --corpus runs/articles-prepared/corpus.txt \
  --output-dir runs/article-llm-fa \
  --device cuda \
  --amp \
  --batch-size 16 \
  --dataloader-num-workers 4 \
  --dataloader-pin-memory \
  --graph-cache-dir runs/graph-cache \
  --graph-encoder gat \
  --fusion context_gated \
  --tokenizer-type unigram \
  --unigram-num-pieces 32000 \
  --epochs 10 \
  --block-size 256
```

Training uses AdamW with a cosine learning-rate schedule and a linear warmup
over the first 5% of optimizer steps by default. Use `--lr-scheduler
none|cosine|linear` and `--warmup-ratio` to change this. Curriculum ordering
(shorter targets first) applies only to the first epoch; later epochs shuffle
normally.

A trained checkpoint writes the normal Graph-LM files plus
`article_llm_config.json`. Graph-enabled runs also include `graph.pt` and graph
memory artifacts for prompt-aware generation.

To continue an interrupted run, pass the saved training state:

```bash
rgnn-cli article-train \
  --corpus runs/articles-prepared/corpus.txt \
  --output-dir runs/article-llm-fa \
  --resume-from runs/article-llm-fa/training_state.pt \
  --device cuda
```

After training, inspect `metrics.json` first. The most useful fields are
`best_next_token_loss`, `best_perplexity`, `best_epoch`, `validation_available`
(when `false`, the "validation" figures are training-loss fallbacks),
`article_data_split`,
`fusion_stats` and `zero_gate_report`. The article-specific metadata is saved in
`article_llm_config.json`; it records that the workflow is native and did not
use external pretrained LMs, distillation, pretrained embeddings or synthetic
data.

For `gated` and `context_gated` fusion, the checkpoint metrics include
`zero_gate_report`. It records whether the zero-init gate guarantee applies and
which `*_alpha_tanh` values were observed after training. Values near zero mean
the model is still mostly ignoring graph updates; that is a training/evidence
signal, not a failure of the native workflow.

The artifact name `article_llm_config.json` is kept stable for compatibility
with checkpoints already produced by this workflow. It describes the article
workflow metadata; it does not imply that the Python namespace is still
`rakhshai_graph_nlp.article_llm`.

## Native Graph Ablations

Use `article-ablation` to measure whether graph components help under the same
native training recipe. The runner can compare `none`, `gat` and `rgcn`
encoders, document/sentence graph scopes, relation groups and optional
graph-memory generation probes.

```bash
rgnn-cli article-ablation \
  --corpus runs/articles-prepared/corpus.txt \
  --output-dir runs/article-ablation-fa \
  --graph-encoders none gat rgcn \
  --graph-scopes document sentence \
  --relation-groups "cooccur=cooccurrence;pmi=pmi;dependency=dependency;topic=topic_document" \
  --fusion context_gated \
  --epochs 3 \
  --probe-topic "آینده آموزش فارسی"
```

The report is written to `article_ablation_report.json` and includes one row
per variant with validation metrics, fusion statistics, `zero_gate_report` and,
when `--probe-topic` is set, lightweight generation probes with graph memory on
and off.

## Generate Articles

After the checkpoint exists, `article-generate` is the normal runtime command.
It loads the model, tokenizer, generation config and graph artifacts from
`--model`, builds the article prompt from `--topic`, `--audience`, `--tone` and
`--sections`, then returns structured Markdown or JSON.

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
```

For developer-facing JSON:

```bash
rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "اقتصاد دیجیتال ایران" \
  --output-format json \
  --output-path runs/article-llm-fa/economy_article.json
```

Graph Memory is enabled by default for graph-enabled article checkpoints. If
`graph_memory.pt` exists, it is loaded. If it is missing and `corpus.txt` exists
inside the checkpoint directory, the runtime rebuilds memory from the saved
corpus and graph config. Disable it for a no-memory generation comparison:

```bash
rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "آینده آموزش فارسی" \
  --graph-memory off
```

Use these generation controls when comparing outputs:

| Option | Purpose |
| --- | --- |
| `--min-new-tokens`, `--max-new-tokens` | Control article length. |
| `--temperature`, `--top-k` | Control sampling randomness. |
| `--repetition-penalty` | Reduce repeated phrases. |
| `--graph-memory-top-k`, `--graph-memory-depth`, `--graph-memory-max-edges` | Tune prompt-linked graph memory retrieval. |
| `--output-format markdown` / `json` | Choose human-readable Markdown or structured JSON. |

## Python API

```python
from rakhshai_graph_nlp.llm.article import (
    ArticleCorpusConfig,
    ArticleGenerationConfig,
    ArticleTrainingConfig,
    generate_persian_article,
    prepare_article_corpus,
    train_article_llm,
)

manifest = prepare_article_corpus(
    ArticleCorpusConfig(
        input_path="data/persian_articles.jsonl",
        output_dir="runs/articles-prepared",
        min_body_chars=400,
    )
)

train_article_llm(
    ArticleTrainingConfig(
        corpus_path=manifest["corpus_path"],
        output_dir="runs/article-llm-fa",
        device="cuda",
        amp=True,
        epochs=10,
    )
)

article = generate_persian_article(
    ArticleGenerationConfig(
        model_dir="runs/article-llm-fa",
        topic="آینده آموزش فارسی",
        sections=4,
    )
)

print(article.full_markdown)
print(article.to_json())
```

The same objects are also re-exported through the stable top-level API for
compatibility:

```python
from rakhshai_graph_nlp import ArticleGenerationConfig, generate_persian_article
```

Prefer the namespace import in new workflow-specific code so the boundary
between `llm` workflows and the lower-level `lm` engine stays explicit.

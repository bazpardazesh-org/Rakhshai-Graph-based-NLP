# LLM Workflows

Rakhshai separates product-shaped LLM workflows from the lower-level Graph-LM
engine. The `rakhshai_graph_nlp.lm` package is still the engine layer: it owns
the tokenizer, graph builder, causal model, trainer, graph memory and reusable
training utilities. The `rakhshai_graph_nlp.llm` namespace is the workflow
layer: it groups higher-level use cases that combine those engine pieces into a
task-specific pipeline.

This keeps future LLM work discoverable without turning the internal Graph-LM
engine into a collection of product workflows. New workflows should live under
`rakhshai_graph_nlp.llm.<workflow_name>` and reuse `rakhshai_graph_nlp.lm`
components instead of duplicating tokenizer, model or trainer internals.

## Namespace Roles

| Namespace | Role |
| --- | --- |
| `rakhshai_graph_nlp.lm` | Low-level Graph-LM engine components and reusable training/generation primitives. |
| `rakhshai_graph_nlp.llm` | High-level LLM workflow namespace for application-shaped pipelines. |
| `rakhshai_graph_nlp.llm.article` | Native Persian article workflow: dataset preparation, article training profile and structured article generation. |
| `rakhshai_graph_nlp.article_llm` | Compatibility alias for older in-repo imports before the workflow namespace was introduced. |

## Available Workflows

- [Native Persian Article LLM](article_llm.md): prepare article corpora, train
  article-focused Graph-LM checkpoints and generate structured Persian articles.

## CLI And Artifacts

The Python API uses the workflow namespace, but the CLI keeps short top-level
commands for now. For article writing, use `article-prepare`, `article-train`
and `article-generate`. For preflight and evaluation work, add `article-audit`
before training and `article-ablation` when you need no-graph/graph/relation
comparisons.

For independent engine-level LM training, use:

- `lm-build-corpus` for native cleaning, deduplication, quality reports and
  train/validation/test splits.
- `lm-tokenize` for native tokenizer fitting/loading and memory-mapped token
  shards.
- `lm-pretrain` for text-only native pretraining from corpus text or token
  shards.
- `lm-ablation` for native no-graph/graph encoder and graph-scope comparisons
  at the engine layer.
- `lm-eval` for local perplexity, prompt scoring, multiple-choice and QA-style
  reports.
- `lm-sft` for human-authored local instruction data.
- `lm-run-report` for consolidated run, registry and checkpoint metadata.

These commands do not load external pretrained LMs, pretrained embeddings,
distillation targets, LLM-generated synthetic data or external LLM judges.

Checkpoint artifact names are intentionally unchanged. For example,
`article_llm_config.json` remains the article workflow metadata file even though
the Python namespace is now `rakhshai_graph_nlp.llm.article`.

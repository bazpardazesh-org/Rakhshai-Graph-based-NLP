# Rakhshai MCP Integration

MCP support is an integration layer that exposes Rakhshai's Persian
Graph-NLP capabilities to AI agents and developer tools, without changing the
core architecture.

Rakhshai Core remains focused on Persian Graph-NLP, Graph-LM, Graph Memory,
GNNs, text analysis, summarisation, classification and generation.  The MCP
server is a professional adapter around that core: it lets agents, IDEs,
chatbots and development tools call Rakhshai capabilities through standard MCP
tools, resources and prompts.

## Message

```text
Persian Text -> Knowledge Graph -> Graph Reasoning -> Graph Memory -> Explainable Output
```

## Architecture

```text
MCP Client
  -> Rakhshai MCP Server
  -> Rakhshai Core API
  -> Tokenizer / Graph Builder / GNN / Graph-LM / Graph Memory
  -> Models / Graphs / Reports / Metrics
```

## Package Layout

```text
rakhshai_mcp/
  server.py
  tools/
    analyze.py
    graph.py
    generate.py
    memory.py
    reports.py
  resources/
    models.py
    graphs.py
    runs.py
  prompts/
    persian_analysis.py
    graph_reasoning.py
  schemas/
    inputs.py
    outputs.py
  security.py
  config.py
```

## Phase 1 Tools

- `rakhshai_analyze_persian_text`: analyzes Persian text, extracts keyword
  candidates, entity candidates and graph signals.
- `rakhshai_build_knowledge_graph`: converts text or documents into a bounded
  Graph-LM multi-relation Persian graph.  The default graph exposes relations
  such as `cooccurrence`, `pmi`, `dependency`, `word_document` and
  `topic_document`; `graph_type="cooccurrence"` keeps the simpler baseline.
- `rakhshai_graph_summarize`: summarizes text with graph-based sentence ranking
  and returns token-level graph evidence.
- `rakhshai_graph_memory_generate`: retrieves prompt-linked evidence through
  `GraphMemoryArtifact`.  When `model_dir` is provided, it can load a whitelisted
  `GraphCausalLM` checkpoint and generate with saved graph and graph-memory
  artifacts.
- `rakhshai_explain_result`: explains outputs through top nodes, important
  relations and graph reasoning paths.
- `rakhshai_optimize_persian_prompt`: rewrites a raw Persian user task into a
  clearer graph-aware prompt before sending it to an LLM such as OpenAI.  It
  normalizes the task shape, adds response expectations and injects compact
  Rakhshai graph evidence when relevant.

Each tool returns the same envelope:

```json
{
  "status": "success",
  "task": "graph_summarization",
  "input_language": "fa",
  "summary": "...",
  "keywords": [],
  "entities": [],
  "graph": {
    "nodes": [],
    "edges": []
  },
  "explanation": {
    "top_nodes": [],
    "important_relations": [],
    "reasoning_path": []
  },
  "artifacts": {
    "graph_id": "...",
    "run_id": "..."
  }
}
```

## Resources

- `rakhshai://models`
- `rakhshai://models/{model_name}`
- `rakhshai://graphs`
- `rakhshai://graphs/{graph_id}`
- `rakhshai://runs`
- `rakhshai://runs/{run_id}/metrics`
- `rakhshai://docs/api`
- `rakhshai://docs/examples`

Resources expose safe metadata from whitelisted project paths.  They do not
give MCP clients arbitrary filesystem access.

## Prompts

- `persian_text_analysis_prompt`
- `graph_reasoning_prompt`
- `graph_memory_generation_prompt`
- `research_report_prompt`
- `model_comparison_prompt`
- `explainable_nlp_prompt`

## Security Rules

The first version deliberately does not expose shell execution, file deletion,
package installation, direct training or arbitrary writes.  Inputs are bounded
by text length, document count and output graph size.  Model, graph and run
resources are read only and constrained to whitelisted directories.

Tool outputs filter common Persian function words from top-node evidence before
the result is sent to an agent.  This keeps graph explanations focused on
conceptual nodes rather than tokens such as conjunctions, object markers or
light verbs.

For OpenAI workflows, the recommended first production path is:

```text
raw Persian question
  -> rakhshai_optimize_persian_prompt
  -> OpenAI model
  -> grounded Persian response
```

Training should be added later as a job-based API:

```text
queued -> running -> completed / failed / cancelled
```

Suggested future tools:

- `rakhshai_start_training_job`
- `rakhshai_get_training_status`
- `rakhshai_cancel_training_job`
- `rakhshai_get_training_artifacts`

## Running

Install the optional MCP dependency:

```bash
pip install -e ".[mcp]"
```

Run the server over stdio:

```bash
rakhshai-mcp
```

For Streamable HTTP:

```bash
RAKHSHAI_MCP_TRANSPORT=streamable-http rakhshai-mcp
```

The adapter follows the stable MCP Python SDK v1.x line with an upper bound
below v2, because v2 is still a pre-release line at the time this integration
was added.

## OpenAI Comparison Test

To check whether graph evidence improves Persian poetry understanding, run the
single-poem comparison harness. It sends the same Persian poem and question to
an OpenAI model in two modes:

- direct OpenAI prompt without Rakhshai context
- OpenAI prompt with compact graph evidence from local Rakhshai MCP tools

The documented, reproducible benchmark and the latest result live in
`docs/mcp_single_poem_evaluation.md`.

First rotate any exposed API key, then set the new key only in `.env.local` or
your local shell:

```bash
export OPENAI_API_KEY="..."
```

Install the optional OpenAI SDK:

```bash
pip install -e ".[openai]"
```

Run a dry run that builds prompts and local MCP evidence without calling
OpenAI:

```bash
python scripts/evaluate_openai_mcp_persian.py --dry-run
```

Run the documented `gpt-5.4` comparison and refresh the docs report:

```bash
python scripts/evaluate_openai_mcp_persian.py \
  --model gpt-5.4 \
  --temperature 0 \
  --top-p 1 \
  --seed 42 \
  --max-output-tokens 3000 \
  --direct-manual-scores 5,5,4,3,4,0 \
  --mcp-manual-scores 5,5,5,5,5,0 \
  --report-path docs/mcp_single_poem_evaluation.md
```

Outputs are written to `runs/openai_mcp_eval/` as JSON, and the human-readable
report is written into the docs page above.

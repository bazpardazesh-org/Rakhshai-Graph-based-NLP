# Rakhshai Low-Data Training Engine

Phase 7 makes Graph-LM training more robust when the Persian corpus is small.
The feature set is enabled by default in `lm-train`, because the recommended
training path now favours generalisation over the older minimal baseline.

## Default Active Features

- Text augmentation on the training split only.
- Graph edge dropout.
- Graph node dropout through incident-edge masking.
- Subgraph edge sampling for each training view.
- Graph-view contrastive consistency loss.
- Curriculum ordering for language-model windows.
- Early stopping with validation-loss patience.
- Generalisation-gap reporting in `metrics.json`.

Validation text is never augmented, and the tokenizer is still fitted on the
original training split before augmented examples are added. This keeps
validation leakage controlled while still giving the graph builder and LM
dataset stronger low-data signal.

## CLI Example

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --augmentation-ratio 0.5 \
  --token-dropout 0.05 \
  --edge-dropout 0.1 \
  --node-dropout 0.05 \
  --subgraph-sampling-ratio 0.9 \
  --contrastive-weight 0.05 \
  --early-stopping-patience 3 \
  --output-dir runs/low-data-graph-lm
```

## Ablation Controls

Use these switches to compare against the old training path:

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

## Metrics

Each run records a `low_data_training` block in `metrics.json` and
`graph_config.json`, including augmentation counts and graph regularisation
settings. Every epoch also records:

- `generalization_gap`
- task-loss stats, including `contrastive` when active
- early-stopping state at run level


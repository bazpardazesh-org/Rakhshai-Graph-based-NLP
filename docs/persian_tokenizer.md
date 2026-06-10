# Rakhshai Persian Tokenizer

Phase 2 introduces a Persian-specific tokenizer path for Graph-LM while keeping
the v1 checkpoint contract compatible.

## Capabilities

- Shared Persian normalization for LM and feature tokenizers.
- Configurable half-space handling: `preserve` or `split`.
- Arabic/Persian character normalization such as `ي` to `ی` and `ك` to `ک`.
- Arabic and Persian digit normalization to ASCII digits.
- Optional light morphology splitting for common Persian prefixes and suffixes.
- Optional compound verb joining for patterns such as `تصمیم گرفت`.
- Tokenizer modes:
  - `word`
  - `char_chunk`, the backward-compatible replacement for the old `subword`
  - `bpe`, a lightweight built-in BPE trainer
  - `unigram`, currently backed by the same lightweight subword path

## CLI Examples

Train Graph-LM with the default word tokenizer:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --tokenizer-type word \
  --output-dir runs/fa-tokenizer-word
```

Train with BPE and Persian-specific options:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --tokenizer-type bpe \
  --tokenizer-bpe-merges 300 \
  --tokenizer-morph-splitting \
  --tokenizer-compound-verb-mode join \
  --output-dir runs/fa-tokenizer-bpe
```

Compare tokenizers under one controlled setup:

```bash
python scripts/compare_tokenizers.py \
  --corpus data/expanded_persian_lm.txt \
  --output-dir runs/tokenizer-ablation \
  --tokenizers word char_chunk bpe \
  --epochs 1 \
  --device cpu
```

The comparison writes `tokenizer_comparison.json` and each run writes the usual
Graph-LM checkpoint files plus `tokenizer_stats` in `metrics.json`.

## Compatibility

Older tokenizer files without `tokenizer_type` still load as `word`. The old
CLI value `subword` is accepted and mapped to `char_chunk`.

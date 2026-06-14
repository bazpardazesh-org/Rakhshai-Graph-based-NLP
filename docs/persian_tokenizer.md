# Rakhshai Persian Tokenizer

Phase 2 introduces a Persian-specific tokenizer path for Graph-LM while keeping
the v1 checkpoint contract compatible.

## Capabilities

- Shared Persian normalization for LM and feature tokenizers.
- Configurable half-space handling: `preserve` or `split`.
- Arabic/Persian character normalization such as `ي` to `ی` and `ك` to `ک`.
- Arabic and Persian digit normalization to ASCII digits, including the Persian
  decimal/thousands separators (`٫` → `.`, `٬` → `,`) so numbers such as
  `۱۲٫۵` stay a single token instead of splitting.
- Punctuation is emitted as standalone tokens. Persian marks (`،`, `؛`, `؟`, …)
  are no longer glued onto adjacent words and ASCII marks (`.`, `!`, `:`, …) are
  no longer dropped, so sentence boundaries are preserved for the LM.
- Configurable hamza folding (`normalize_hamza`) and ezafe handling
  (`ezafe_mode`); see [Normalization options](#normalization-options).
- Optional light morphology splitting for common Persian prefixes and suffixes.
- Optional compound verb joining for patterns such as `تصمیم گرفت`.
- Tokenizer modes:
  - `word`
  - `char_chunk`, the backward-compatible replacement for the old `subword`
  - `bpe`, a lightweight built-in BPE trainer
  - `unigram`, a genuine Unigram LM tokenizer trained with hard-EM (seed
    substring vocabulary, Viterbi segmentation, iterative pruning to
    `unigram_num_pieces`, single-character fallback for full coverage)

## Operational default and special tokens

For Graph-LM training (`lm-train`), `unigram` is now the **operational default**
because it gives the lowest Persian out-of-vocabulary rate (word-level
tokenization drops far more held-out tokens to `<unk>`). Pass
`--tokenizer-type word` to opt back into whole-word tokens. The unigram
vocabulary size is controlled by `--unigram-num-pieces` (default `8000`).

The tokenizer reserves five stable special ids: `<pad>` (0), `<unk>` (1),
`<bos>` (2), `<eos>` (3) and `<mask>` (4). `<mask>` backs the masked-token
training objective so it no longer collides with genuine unknown tokens.
Newly saved tokenizer artifacts use `tokenizer_version = 3`.

## Normalization options

`PersianNormalizerConfig` exposes two language-aware switches:

- `normalize_hamza` (default `True`): folds hamza-bearing letters onto their
  plain forms (`ئ` → `ی`, `ؤ` → `و`, e.g. `مسائل` → `مسایل`). Disable it to keep
  the original orthography at the cost of a more fragmented vocabulary.
- `ezafe_mode` (default `marker`): rewrites the ezafe (`ۀ` or `ه` + U+0654) as the
  explicit `ه‌ی` marker so the grammatical construction survives as a learnable
  token (`خانهٔ` → `خانه‌ی`). Set it to `collapse` to fold the ezafe onto a plain
  `ه` (`خانهٔ` → `خانه`), the previous behaviour.

## CLI Examples

Train Graph-LM with the default unigram tokenizer (set the vocabulary size):

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --unigram-num-pieces 8000 \
  --output-dir runs/fa-tokenizer-unigram
```

Opt back into the whole-word tokenizer:

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
CLI value `subword` is accepted and mapped to `char_chunk`. Tokenizers saved
before `<mask>` existed load with `<mask>` mapped onto `<unk>`, so masked-token
training stays well-defined for legacy checkpoints.

Tokenizer configs serialised before `ezafe_mode` existed were produced under the
old collapse behaviour, so they load with `ezafe_mode = "collapse"` to reproduce
their original normalization faithfully. Only newly created tokenizers default to
`marker`.

from __future__ import annotations

import json

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm.shards import (
    TokenShardConfig,
    TokenShardDataset,
    tokenizer_audit,
    write_token_shards,
)
from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer


def test_tokenizer_byte_fallback_round_trips_oov_word():
    tokenizer = PersianTokenizer(tokenizer_type="word", byte_fallback=True).fit(
        ["سلام به جهان"]
    )

    ids = tokenizer.encode("واژهتازه", add_special_tokens=False)
    decoded = tokenizer.decode(ids)

    assert tokenizer.unk_id not in ids
    assert decoded == "واژهتازه"
    restored = PersianTokenizer.from_dict(tokenizer.to_dict())
    assert restored.byte_fallback is True
    assert restored.decode(ids) == "واژهتازه"


def test_write_token_shards_and_lazy_dataset(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "train.txt").write_text(
        "این متن اول برای شارد آموزشی است\n"
        "این متن دوم برای ساخت پنجره‌های آموزشی است\n",
        encoding="utf-8",
    )
    (corpus_dir / "validation.txt").write_text(
        "این متن ارزیابی کوتاه است\n",
        encoding="utf-8",
    )
    output = tmp_path / "shards"

    manifest = write_token_shards(
        TokenShardConfig(
            corpus_dir=str(corpus_dir),
            output_dir=str(output),
            tokenizer_type="unigram",
            tokenizer_unigram_num_pieces=80,
            byte_fallback=True,
            block_size=6,
            tokens_per_shard=8,
        )
    )

    assert (output / "tokenizer.json").exists()
    assert (output / "shard_manifest.json").exists()
    assert manifest["native_independence"]["uses_external_llm_judge"] is False
    assert any(row["split"] == "train" for row in manifest["shards"])
    dataset = TokenShardDataset(output / "shard_manifest.json", split="train", block_size=6)
    input_ids, labels = dataset[0]
    assert input_ids.shape == labels.shape == (6,)
    assert labels.ne(-100).any()


def test_lm_tokenize_cli_writes_manifest(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "این متن برای دستور توکن‌سازی است\n"
        "متن دوم نیز فارسی و قابل استفاده است\n",
        encoding="utf-8",
    )
    output = tmp_path / "tok"

    result = main(
        [
            "lm-tokenize",
            "--input",
            str(corpus),
            "--output-dir",
            str(output),
            "--tokenizer-type",
            "unigram",
            "--unigram-num-pieces",
            "64",
            "--tokenizer-byte-fallback",
            "--block-size",
            "6",
            "--tokens-per-shard",
            "10",
        ]
    )

    assert result == 0
    manifest = json.loads((output / "shard_manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit"]["train"]["byte_fallback"] is True
    assert manifest["block_size"] == 6


def test_tokenizer_audit_reports_rates():
    tokenizer = PersianTokenizer(tokenizer_type="unigram", unigram_num_pieces=40).fit(
        ["خانه کتابخانه"]
    )
    report = tokenizer_audit(tokenizer, ["خانه کتابخانه"])
    assert report["tokens"] > 0
    assert report["tokens_per_character"] > 0
    assert "unknown_rate" in report


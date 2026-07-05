from __future__ import annotations

import json

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm.model import GraphLMConfig
from rakhshai_graph_nlp.lm.sft import SFTConfig, format_sft_record, load_sft_texts, train_sft


def test_sft_formats_prompt_completion_and_messages(tmp_path):
    config = SFTConfig(input_path=str(tmp_path / "sft.jsonl"), output_dir=str(tmp_path / "out"))
    direct = format_sft_record({"prompt": "سلام کن", "completion": "سلام"}, config)
    messages = format_sft_record(
        {
            "messages": [
                {"role": "user", "content": "خلاصه کن"},
                {"role": "assistant", "content": "خلاصه آماده است"},
            ]
        },
        config,
    )

    assert "دستور:" in direct and "پاسخ:" in direct
    assert "خلاصه آماده است" in messages


def test_train_sft_writes_manifest(tmp_path):
    data = tmp_path / "sft.jsonl"
    data.write_text(
        json.dumps({"prompt": "یک جمله فارسی بنویس", "completion": "این یک جمله فارسی است."}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "sft-run"

    metrics = train_sft(
        SFTConfig(
            input_path=str(data),
            output_dir=str(output),
            epochs=1,
            batch_size=1,
            block_size=8,
            validation_ratio=0.0,
            tokenizer_unigram_num_pieces=64,
            device="cpu",
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            graph_encoder="none",
        ),
    )

    assert metrics["sft_manifest"]["records_used"] == 1
    assert (output / "sft_manifest.json").exists()
    assert load_sft_texts(SFTConfig(input_path=str(data), output_dir=str(output)))


def test_lm_sft_cli_writes_checkpoint(tmp_path):
    data = tmp_path / "sft.jsonl"
    data.write_text(
        json.dumps({"prompt": "پاسخ کوتاه بده", "completion": "باشه."}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "cli-sft"

    result = main(
        [
            "lm-sft",
            "--input",
            str(data),
            "--output-dir",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
            "--validation-ratio",
            "0",
            "--unigram-num-pieces",
            "64",
            "--d-model",
            "16",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--dim-feedforward",
            "32",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert (output / "model.pt").exists()
    assert (output / "sft_manifest.json").exists()


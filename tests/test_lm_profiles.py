from __future__ import annotations

import torch

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm.model import GraphCausalLM, GraphLMConfig
from rakhshai_graph_nlp.lm.profiles import (
    CONTEXT_PRESETS,
    available_model_profiles,
    build_graph_lm_config_from_profile,
)


def test_named_model_profile_builds_graph_lm_config():
    cfg = build_graph_lm_config_from_profile(
        "tiny-test",
        vocab_size=128,
        overrides={"graph_encoder": "none", "attention_backend": "math"},
    )

    assert cfg.vocab_size == 128
    assert cfg.d_model == 32
    assert cfg.graph_encoder == "none"
    assert cfg.attention_backend == "math"
    assert "125m" in available_model_profiles()
    assert 2048 in CONTEXT_PRESETS


def test_activation_checkpointing_forward_backward():
    cfg = GraphLMConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_heads=2,
        n_layers=2,
        dim_feedforward=32,
        graph_encoder="none",
        activation_checkpointing=True,
    )
    model = GraphCausalLM(cfg)
    model.train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    output["loss"].backward()

    assert output["logits"].shape == (2, 8, cfg.vocab_size)
    assert model.token_embedding.weight.grad is not None


def test_lm_train_accepts_tiny_model_profile(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "این متن برای پروفایل کوچک آزمایشی است\n"
        "متن دوم باعث ساخته شدن چند پنجره آموزشی می‌شود\n",
        encoding="utf-8",
    )
    output = tmp_path / "run"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output),
            "--model-profile",
            "tiny-test",
            "--graph-encoder",
            "none",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--tokenizer-type",
            "unigram",
            "--unigram-num-pieces",
            "64",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert (output / "model.pt").exists()


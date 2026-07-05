from __future__ import annotations

import json

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm.distributed import get_distributed_info, maybe_wrap_distributed
from rakhshai_graph_nlp.lm.model import GraphCausalLM, GraphLMConfig
from rakhshai_graph_nlp.lm.trainer import LMTrainingConfig, train_graph_lm


TEXTS = [
    "این متن برای آزمایش انباشت گرادیان نوشته شده است",
    "مدل زبانی مستقل باید بدون مدل بیرونی آموزش ببیند",
    "پیکره کوچک برای آزمون سریع کافی است",
]


def test_trainer_records_scaling_metadata_and_checkpoint_manifest(tmp_path):
    output = tmp_path / "run"
    metrics = train_graph_lm(
        TEXTS,
        training_config=LMTrainingConfig(
            output_dir=str(output),
            epochs=1,
            batch_size=1,
            block_size=8,
            validation_ratio=0.0,
            task_losses="next_token",
            gradient_accumulation_steps=2,
            warmup_steps=1,
            min_lr_ratio=0.1,
            adam_beta1=0.85,
            adam_beta2=0.95,
            adam_eps=1e-7,
            precision="bf16",
            sharded_checkpoint=True,
            early_stopping_patience=0,
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
        graph_encoder="none",
    )

    assert metrics["trainer_scaling"]["gradient_accumulation_steps"] == 2
    assert metrics["trainer_scaling"]["precision"] == "bf16"
    assert metrics["history"][0]["optimizer_steps"] >= 1
    manifest = json.loads((output / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sharded_checkpoint"] is True
    assert "model.pt" in manifest["model_files"]


def test_distributed_helpers_are_noops_without_process_group():
    model = GraphCausalLM(GraphLMConfig(vocab_size=16, graph_encoder="none"))
    info = get_distributed_info("ddp")
    wrapped = maybe_wrap_distributed(model, "ddp")

    assert info.world_size >= 1
    assert wrapped is model


def test_lm_train_cli_accepts_scaling_options(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(TEXTS) + "\n", encoding="utf-8")
    output = tmp_path / "cli-run"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output),
            "--graph-encoder",
            "none",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
            "--gradient-accumulation-steps",
            "2",
            "--warmup-steps",
            "1",
            "--min-lr-ratio",
            "0.1",
            "--precision",
            "fp32",
            "--adam-beta1",
            "0.85",
            "--adam-beta2",
            "0.95",
            "--adam-eps",
            "1e-7",
            "--sharded-checkpoint",
            "--distributed-backend",
            "none",
            "--task-losses",
            "next_token",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["trainer_scaling"]["gradient_accumulation_steps"] == 2

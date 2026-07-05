from __future__ import annotations

import json

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm.eval import (
    NativeEvalConfig,
    evaluate_lm_checkpoint,
    export_human_review,
)
from rakhshai_graph_nlp.lm.model import GraphLMConfig
from rakhshai_graph_nlp.lm.trainer import LMTrainingConfig, train_graph_lm


TEXTS = [
    "این متن برای ارزیابی محلی مدل زبانی است",
    "مدل مستقل باید با داده فارسی سنجیده شود",
]


def _train_tiny_checkpoint(tmp_path):
    output = tmp_path / "model"
    train_graph_lm(
        TEXTS,
        training_config=LMTrainingConfig(
            output_dir=str(output),
            epochs=1,
            batch_size=1,
            block_size=8,
            validation_ratio=0.0,
            task_losses="next_token",
            early_stopping_patience=0,
            tokenizer_type="unigram",
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
        graph_encoder="none",
    )
    return output


def test_native_eval_reports_perplexity_mc_and_qa(tmp_path):
    model_dir = _train_tiny_checkpoint(tmp_path)
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        "\n".join(
            [
                json.dumps({"text": TEXTS[0]}, ensure_ascii=False),
                json.dumps(
                    {
                        "prompt": "زبان متن چیست؟",
                        "choices": ["فارسی", "انگلیسی"],
                        "answer": "فارسی",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "prediction": "مدل مستقل",
                        "answer": ["مدل مستقل", "سامانه مستقل"],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    report = evaluate_lm_checkpoint(
        NativeEvalConfig(
            model_dir=str(model_dir),
            eval_path=str(eval_path),
            output_path=str(report_path),
            block_size=8,
            device="cpu",
        )
    )

    assert report["native_independence"]["uses_external_llm_judge"] is False
    assert report["perplexity"]["perplexity"] > 0
    assert "multiple_choice" in report
    assert report["extractive_qa"]["exact_match"] == 1.0
    assert report_path.exists()


def test_lm_eval_cli_writes_report(tmp_path):
    model_dir = _train_tiny_checkpoint(tmp_path)
    eval_path = tmp_path / "eval.txt"
    eval_path.write_text(TEXTS[0] + "\n", encoding="utf-8")
    report_path = tmp_path / "cli-report.json"

    result = main(
        [
            "lm-eval",
            "--model",
            str(model_dir),
            "--eval-file",
            str(eval_path),
            "--output-path",
            str(report_path),
            "--block-size",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["row_count"] == 1


def test_export_human_review_templates(tmp_path):
    paths = export_human_review(["پرسش"], ["پاسخ"], tmp_path)

    assert all((tmp_path / name).exists() for name in ["human_review_items.jsonl", "human_review_items.csv", "human_review_schema.json"])
    schema = json.loads((tmp_path / "human_review_schema.json").read_text(encoding="utf-8"))
    assert schema["uses_external_llm_judge"] is False
    assert paths["jsonl"].endswith(".jsonl")


from __future__ import annotations

import json

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm.corpus import CorpusBuildConfig, build_lm_corpus, persian_ratio


def test_build_lm_corpus_writes_quality_gate_artifacts(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"text": "این متن فارسی برای آموزش مدل زبانی مستقل آماده شده است."}, ensure_ascii=False),
                json.dumps({"text": "این متن فارسی برای آموزش مدل زبانی مستقل آماده شده است."}, ensure_ascii=False),
                json.dumps({"text": "short"}, ensure_ascii=False),
                json.dumps({"text": "Another mostly English row without enough Persian words."}, ensure_ascii=False),
                json.dumps({"body": "گزارش دوم درباره داده تمیز و ارزیابی کیفیت پیکره فارسی است."}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "corpus"

    manifest = build_lm_corpus(
        CorpusBuildConfig(
            input_paths=[str(source)],
            output_dir=str(output),
            input_format="jsonl",
            text_fields=["text", "body"],
            min_chars=12,
            validation_ratio=0.25,
            test_ratio=0.0,
            seed=7,
        )
    )

    assert manifest["quality_report"]["records_accepted"] == 2
    assert manifest["quality_report"]["reject_counts"]["exact_duplicate"] == 1
    assert manifest["quality_report"]["native_independence"]["uses_external_pretrained_lm"] is False
    for name in ["corpus.txt", "train.txt", "validation.txt", "test.txt", "manifest.json", "quality_report.json", "rejected_records.jsonl"]:
        assert (output / name).exists()
    assert persian_ratio("متن فارسی") == 1.0


def test_lm_build_corpus_cli(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(
        "این یک متن فارسی بلند برای ساخت پیکره است\n"
        "این متن دوم برای تقسیم داده آموزشی استفاده می‌شود\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"

    result = main(
        [
            "lm-build-corpus",
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--min-chars",
            "10",
            "--test-ratio",
            "0",
            "--validation-ratio",
            "0.5",
        ]
    )

    assert result == 0
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert report["records_accepted"] == 2
    assert report["split_counts"]["validation"] == 1


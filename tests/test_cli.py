import json

from rakhshai_graph_nlp.cli import _load_text_dataset, _split_indices, main


def test_load_text_dataset_reads_csv(tmp_path):
    dataset = tmp_path / "docs.csv"
    dataset.write_text(
        "text,label\n"
        "خبر سیاسی,politics\n"
        "خبر ورزشی,sports\n",
        encoding="utf-8",
    )

    texts, labels = _load_text_dataset(str(dataset))

    assert texts == ["خبر سیاسی", "خبر ورزشی"]
    assert labels == ["politics", "sports"]


def test_split_indices_keeps_train_validation_and_test_for_small_dataset():
    train, validation, test = _split_indices(
        6,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=0,
    )

    assert len(train) == 4
    assert len(validation) == 1
    assert len(test) == 1
    assert sorted([*train, *validation, *test]) == list(range(6))


def test_cli_dataset_pipeline_writes_metrics_and_model(tmp_path):
    dataset = tmp_path / "docs.csv"
    dataset.write_text(
        "text,label\n"
        "انتخابات و دولت,politics\n"
        "مجلس و قانون,politics\n"
        "فوتبال و تیم,sports\n"
        "گل و مسابقه,sports\n"
        "نقاشی و نمایشگاه,art\n"
        "هنر و گالری,art\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    model_path = output_dir / "model.pt"

    result = main(
        [
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--save-model",
            str(model_path),
            "--epochs",
            "1",
            "--hidden-dim",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    report_path = output_dir / "metrics.json"
    assert report_path.exists()
    assert model_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_documents"] == 6
    assert report["num_classes"] == 3
    assert set(report["splits"]) == {"train", "validation", "test"}
    assert report["splits"]["train"]["count"] == 4

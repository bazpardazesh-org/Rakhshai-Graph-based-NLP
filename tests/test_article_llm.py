import json

import torch

from rakhshai_graph_nlp import article_llm
from rakhshai_graph_nlp.llm.article import (
    ArticleAblationConfig,
    ArticleAuditConfig,
    ArticleCorpusConfig,
    ArticleGenerationConfig,
    ArticleTrainingConfig,
    PersianArticle,
    audit_article_corpus,
    generate_persian_article,
    prepare_article_corpus,
    run_article_ablation,
)
from rakhshai_graph_nlp.llm.article.core import _parse_generated_article
from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm.model import GraphLMConfig, GraphTokenFusion


def test_article_llm_compatibility_alias():
    assert article_llm.PersianArticle is PersianArticle


def test_prepare_article_corpus_reads_txt_jsonl_and_csv(tmp_path):
    txt_path = tmp_path / "articles.txt"
    txt_path.write_text(
        "# آموزش زبان فارسی\n"
        "زبان فارسی برای پژوهش و نوشتن مقاله به داده‌های تمیز نیاز دارد.\n\n"
        "این متن کوتاه است.\n",
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "articles.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "title": "گراف دانش",
                "body": (
                    "گراف دانش رابطه میان مفهوم‌ها را برای مدل زبانی "
                    "روشن می‌کند."
                ),
                "keywords": ["گراف", "مدل"],
                "metadata": {"source": "unit"},
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps({"title": "ناقص", "body": ""}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "articles.csv"
    csv_path.write_text(
        "title,body,summary,keywords\n"
        "مقاله فارسی,این مقاله درباره ساخت دیتاست فارسی برای آموزش مدل "
        "است.,خلاصه,داده\n",
        encoding="utf-8",
    )

    for source in (txt_path, jsonl_path, csv_path):
        manifest = prepare_article_corpus(
            ArticleCorpusConfig(
                input_path=str(source),
                output_dir=str(tmp_path / f"prepared-{source.suffix[1:]}"),
                min_body_chars=20,
                validation_ratio=0.5,
                seed=1,
            )
        )
        assert manifest["accepted_records"] >= 1
        assert manifest["splits"]["train"] >= 1
        prepared_dir = tmp_path / f"prepared-{source.suffix[1:]}"
        assert (prepared_dir / "corpus.txt").exists()
        assert (prepared_dir / "manifest.json").exists()
        if source == txt_path:
            prepared = (prepared_dir / "prepared_articles.jsonl").read_text(
                encoding="utf-8"
            )
            first = json.loads(prepared.splitlines()[0])
            assert first["title"] == "آموزش زبان فارسی"


def test_prepare_article_corpus_wikipedia_prompt_format(tmp_path):
    source = tmp_path / "wiki.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "1",
                "url": "https://fa.wikipedia.org/wiki/هوش_مصنوعی",
                "title": "هوش مصنوعی",
                "text": (
                    "هوش مصنوعی شاخه‌ای از علوم رایانه است. این شاخه به ساخت "
                    "سامانه‌هایی می‌پردازد که توانایی یادگیری و تصمیم‌گیری "
                    "دارند. کاربردهای آن در پزشکی، آموزش و صنعت دیده می‌شود."
                ),
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = prepare_article_corpus(
        ArticleCorpusConfig(
            input_path=str(source),
            output_dir=str(tmp_path / "prepared-wiki"),
            training_format="wikipedia_prompt",
            min_body_chars=20,
            validation_ratio=0,
        )
    )

    corpus_text = (tmp_path / "prepared-wiki" / "corpus.txt").read_text(
        encoding="utf-8"
    )
    prepared_text = (tmp_path / "prepared-wiki" / "prepared_articles.jsonl").read_text(
        encoding="utf-8"
    )
    assert manifest["accepted_records"] == 1
    assert manifest["config"]["training_format"] == "wikipedia_prompt"
    assert "موضوع مقاله: هوش مصنوعی" in corpus_text
    assert "مخاطب: عمومی" in corpus_text
    assert "لحن: دانشنامه‌ای" in corpus_text
    assert "مقاله: | # هوش مصنوعی" in corpus_text
    assert "## بخش 1: هوش مصنوعی" in corpus_text
    assert "## جمع‌بندی" in corpus_text
    assert json.loads(prepared_text)["body"].startswith("هوش مصنوعی شاخه‌ای")


def test_parse_article_fields_generation_normalizes_labels():
    article = _parse_generated_article(
        (
            "عنوان: هوش مصنوعی فارسی | "
            "خلاصه: معرفی کوتاه برای خواننده فارسی | "
            "متن مقاله: بخش نخست درباره داده فارسی است. "
            "بخش دوم درباره گراف متن است. "
            "جمع‌بندی نهایی این است که آموزش بومی مهم است."
        ),
        ArticleGenerationConfig(
            model_dir="runs/article-llm-fa",
            topic="هوش مصنوعی فارسی",
            sections=2,
        ),
    )

    assert article.title == "هوش مصنوعی فارسی"
    assert article.introduction == "معرفی کوتاه برای خواننده فارسی"
    assert article.conclusion == "جمع‌بندی نهایی این است که آموزش بومی مهم است."
    assert [section["body"] for section in article.sections] == [
        "بخش نخست درباره داده فارسی است.",
        "بخش دوم درباره گراف متن است.",
    ]
    assert "عنوان:" not in article.full_markdown
    assert "متن مقاله:" not in article.full_markdown


def test_fallback_article_parser_does_not_duplicate_conclusion():
    article = _parse_generated_article(
        "این مقدمه کوتاه است. بخش اصلی مقاله درباره داده فارسی است. جمع‌بندی پایانی.",
        ArticleGenerationConfig(
            model_dir="runs/article-llm-fa",
            topic="داده فارسی",
            sections=2,
        ),
    )

    assert article.introduction == "این مقدمه کوتاه است."
    assert article.conclusion == "جمع‌بندی پایانی."
    assert article.sections[0]["body"] == "بخش اصلی مقاله درباره داده فارسی است."
    assert all(
        section["body"] != article.conclusion
        for section in article.sections
    )


def test_article_audit_reports_native_quality_and_tokenizer_stats(tmp_path):
    source = tmp_path / "articles.jsonl"
    body = (
        "مدل زبانی فارسی برای نوشتن مقاله باید ساختار عنوان، مقدمه و بخش‌ها "
        "را از داده‌های انسانی یاد بگیرد."
    )
    source.write_text(
        json.dumps({"title": "مدل فارسی", "body": body}, ensure_ascii=False)
        + "\n"
        + json.dumps({"title": "مدل فارسی تکراری", "body": body}, ensure_ascii=False)
        + "\n"
        + json.dumps(
            {
                "title": "گراف مقاله",
                "body": (
                    "گراف متن رابطه میان مفهوم‌ها را نگه می‌دارد و در زمان "
                    "تولید مقاله به حافظه ساختاری کمک می‌کند."
                ),
                "metadata": {"source": "unit"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_article_corpus(
        ArticleAuditConfig(
            input_path=str(source),
            output_dir=str(tmp_path / "audit"),
            min_body_chars=20,
            validation_ratio=0.34,
            tokenizer_types=["word", "unigram"],
            tokenizer_unigram_num_pieces=40,
            near_duplicate_threshold=0.8,
        )
    )

    assert report["native_constraints"]["uses_external_pretrained_lm"] is False
    assert report["duplicates"]["exact_duplicate_bodies"] == 1
    assert report["tokenizer_benchmark"]["validation_examples"] >= 1
    assert set(report["tokenizer_benchmark"]["tokenizers"]) == {"word", "unigram"}
    assert (tmp_path / "audit" / "article_audit.json").exists()


def test_zero_init_gate_keeps_initial_graph_fusion_equivalent_to_text():
    torch.manual_seed(0)
    token_embeddings = torch.randn(2, 3, 8)
    graph_embeddings = torch.randn(2, 3, 8)
    for fusion_name in ("gated", "context_gated"):
        fusion = GraphTokenFusion(
            GraphLMConfig(
                vocab_size=16,
                d_model=8,
                n_heads=2,
                graph_encoder="gcn",
                fusion=fusion_name,
            )
        )
        output, stats = fusion(
            token_embeddings,
            graph_embeddings,
            return_stats=True,
        )
        assert torch.allclose(output, token_embeddings, atol=1e-6)
        assert float(stats["token_alpha_tanh"].detach()) == 0.0


def test_article_ablation_runner_writes_native_report(tmp_path):
    corpus = tmp_path / "corpus.txt"
    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    train.write_text(
        "عنوان: مدل فارسی | متن مقاله: آموزش بومی مدل فارسی با داده انسانی انجام می‌شود.\n"
        "عنوان: گراف مقاله | متن مقاله: گراف متن رابطه واژه‌ها را برای مدل نگه می‌دارد.\n",
        encoding="utf-8",
    )
    validation.write_text(
        "عنوان: ارزیابی مقاله | متن مقاله: ارزیابی بومی باید بدون داور خارجی انجام شود.\n",
        encoding="utf-8",
    )
    corpus.write_text(
        train.read_text(encoding="utf-8")
        + validation.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report = run_article_ablation(
        ArticleAblationConfig(
            training_config=ArticleTrainingConfig(
                corpus_path=str(corpus),
                output_dir=str(tmp_path / "ablation-template"),
                epochs=1,
                batch_size=1,
                block_size=16,
                d_model=16,
                n_heads=2,
                n_layers=1,
                dim_feedforward=32,
                graph_hidden_dim=16,
                graph_encoder="none",
                task_losses="next_token",
                text_augmentation=False,
                contrastive_weight=0,
                tokenizer_type="word",
                validation_ratio=0,
                device="cpu",
            ),
            output_dir=str(tmp_path / "ablation"),
            graph_encoders=["none"],
            graph_scopes=["document"],
            relation_groups={},
        )
    )

    assert report["native_constraints"]["uses_external_llm_judge"] is False
    assert [row["name"] for row in report["variants"]] == ["no_graph"]
    assert report["variants"][0]["zero_gate_report"]["supports_zero_init_gate"] is True
    assert (tmp_path / "ablation" / "article_ablation_report.json").exists()


def test_article_cli_train_and_generate_outputs_developer_artifacts(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()
    corpus = prepared_dir / "corpus.txt"
    train = prepared_dir / "train.txt"
    validation = prepared_dir / "validation.txt"
    corpus_text = (
        "عنوان: هوش مصنوعی فارسی | متن مقاله: مدل زبانی فارسی باید ساختار متن "
        "و رابطه واژه‌ها را یاد بگیرد.\n"
        "عنوان: گراف متن | متن مقاله: گراف متن می‌تواند ارتباط واژه‌ها و سندها "
        "را برای تولید مقاله نگه دارد.\n"
        "عنوان: آموزش بومی | متن مقاله: آموزش بومی بدون مدل آماده به ارزیابی "
        "توان معماری کمک می‌کند.\n"
    )
    corpus.write_text(corpus_text, encoding="utf-8")
    train.write_text(
        "عنوان: هوش مصنوعی فارسی | متن مقاله: مدل زبانی فارسی باید ساختار متن "
        "و رابطه واژه‌ها را یاد بگیرد.\n"
        "عنوان: گراف متن | متن مقاله: گراف متن می‌تواند ارتباط واژه‌ها و سندها "
        "را برای تولید مقاله نگه دارد.\n",
        encoding="utf-8",
    )
    validation.write_text(
        "عنوان: آموزش بومی | متن مقاله: آموزش بومی بدون مدل آماده به ارزیابی "
        "توان معماری کمک می‌کند.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "article-model"

    result = main(
        [
            "article-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--graph-encoder",
            "gcn",
            "--fusion",
            "context_gated",
            "--graph-relations",
            "cooccurrence",
            "--task-losses",
            "next_token",
            "--no-text-augmentation",
            "--contrastive-weight",
            "0",
            "--edge-dropout",
            "0",
            "--node-dropout",
            "0",
            "--subgraph-sampling-ratio",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "16",
            "--d-model",
            "16",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--dim-feedforward",
            "32",
            "--graph-hidden-dim",
            "16",
            "--tokenizer-type",
            "word",
            "--validation-ratio",
            "0",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    for filename in [
        "model.pt",
        "tokenizer.json",
        "graph_config.json",
        "generation_config.json",
        "metrics.json",
        "graph.pt",
        "article_llm_config.json",
        "corpus.txt",
    ]:
        assert (output_dir / filename).exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["corpus_split"]["validation_source"] == "explicit"
    assert metrics["corpus_split"]["train_examples"] == 2
    assert metrics["corpus_split"]["validation_examples"] == 1
    article_config = json.loads(
        (output_dir / "article_llm_config.json").read_text(encoding="utf-8")
    )
    assert article_config["data_split"]["uses_prepared_splits"] is True

    article = generate_persian_article(
        ArticleGenerationConfig(
            model_dir=str(output_dir),
            topic="آموزش مدل فارسی",
            sections=2,
            min_new_tokens=0,
            max_new_tokens=4,
            graph_memory=False,
            device="cpu",
        )
    )
    assert article.title
    assert len(article.sections) == 2
    assert article.full_markdown.startswith("# ")

    output_path = tmp_path / "article.json"
    result = main(
        [
            "article-generate",
            "--model",
            str(output_dir),
            "--topic",
            "آموزش مدل فارسی",
            "--sections",
            "2",
            "--min-new-tokens",
            "0",
            "--max-new-tokens",
            "4",
            "--graph-memory",
            "off",
            "--output-format",
            "json",
            "--output-path",
            str(output_path),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload) >= {"title", "introduction", "sections", "full_markdown"}
    assert len(payload["sections"]) == 2

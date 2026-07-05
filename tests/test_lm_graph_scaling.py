from __future__ import annotations

import json

from rakhshai_graph_nlp.lm.graph_scaling import (
    LMGraphAblationConfig,
    run_lm_graph_ablation,
    write_graph_feature_store,
)
from rakhshai_graph_nlp.lm.model import GraphLMConfig
from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer
from rakhshai_graph_nlp.lm.trainer import LMTrainingConfig, train_graph_lm


TEXTS = [
    "کتابخانه دانشگاه برای پژوهشگران منابع فارسی فراهم کرد",
    "دانشجویان درباره مدل زبانی مستقل گزارش نوشتند",
    "استاد روش ارزیابی پیکره فارسی را توضیح داد",
]


def test_graph_feature_store_writes_summary(tmp_path):
    tokenizer = PersianTokenizer(tokenizer_type="unigram", unigram_num_pieces=80).fit(TEXTS)
    report = write_graph_feature_store(
        TEXTS,
        tokenizer,
        tmp_path / "graph_feature_store.json",
        graph_relations=["cooccurrence", "word_document"],
    )

    assert report["num_nodes"] > 0
    assert report["num_edges"] > 0
    assert report["native_independence"]["uses_external_pretrained_lm"] is False
    assert (tmp_path / "graph_feature_store.json").exists()


def test_graph_cache_writes_manifest(tmp_path):
    cache_dir = tmp_path / "cache"
    train_graph_lm(
        TEXTS,
        training_config=LMTrainingConfig(
            output_dir=str(tmp_path / "run"),
            epochs=1,
            batch_size=1,
            block_size=8,
            validation_ratio=0.0,
            task_losses="next_token",
            graph_cache_dir=str(cache_dir),
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
            graph_encoder="gcn",
            graph_hidden_dim=16,
        ),
        graph_encoder="gcn",
    )

    manifests = list(cache_dir.glob("*.graph.json"))
    assert manifests
    cache_report = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert cache_report["cache_enabled"] is True
    assert cache_report["num_nodes"] > 0


def test_lm_graph_ablation_writes_native_report(tmp_path):
    report = run_lm_graph_ablation(
        TEXTS,
        LMGraphAblationConfig(
            output_dir=str(tmp_path / "ablation"),
            training_config=LMTrainingConfig(
                output_dir=str(tmp_path / "unused"),
                epochs=1,
                batch_size=1,
                block_size=8,
                validation_ratio=0.0,
                task_losses="next_token",
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
                graph_hidden_dim=16,
            ),
            graph_encoders=["none", "gcn"],
            graph_scopes=["document"],
            relation_groups={"cooccur": ["cooccurrence"]},
        ),
    )

    assert len(report["variants"]) == 2
    assert report["native_independence"]["uses_external_llm_judge"] is False
    assert (tmp_path / "ablation" / "lm_ablation_report.json").exists()


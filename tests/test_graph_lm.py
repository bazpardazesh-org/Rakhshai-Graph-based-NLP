import json

import torch

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm import (
    GraphCausalLM,
    GraphLMConfig,
    LMDataset,
    PersianTokenizer,
    build_graph_lm_graph,
)
from rakhshai_graph_nlp.lm.model import RakhshaiGraphEncoder, WeightedSAGEConv
from rakhshai_graph_nlp.lm.trainer import LMTrainer, LMTrainingConfig, _split_corpus, train_graph_lm


def test_persian_tokenizer_normalizes_and_encodes():
    tokenizer = PersianTokenizer().fit(["مي روم كلاس", "می\u200cروم کلاس"])

    tokens = tokenizer.tokenize("مي\u200cروم به كلاس")
    ids = tokenizer.encode("مي\u200cروم به كلاس")

    assert "می\u200cروم" in tokens
    assert "كلاس" not in tokens
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


def test_subword_tokenizer_splits_long_persian_words():
    tokenizer = PersianTokenizer(tokenizer_type="subword", subword_chunk_size=3).fit(
        ["دانش‌آموزان پژوهشگران"]
    )

    tokens = tokenizer.tokenize("دانش‌آموزان")

    assert any(token.startswith("##") for token in tokens)
    assert tokenizer.decode(tokenizer.encode("دانش‌آموزان")) != ""


def test_lm_dataset_creates_next_token_targets():
    tokenizer = PersianTokenizer().fit(["من امروز به مدرسه رفتم"])
    dataset = LMDataset(["من امروز به مدرسه رفتم"], tokenizer, block_size=4)
    input_ids, target_ids = dataset[0]

    assert input_ids.shape == target_ids.shape == (4,)
    assert torch.equal(input_ids[1:], target_ids[:-1])


def test_graph_lm_forward_shape():
    texts = ["مجلس قانون را تصویب کرد", "دولت لایحه جدید آورد"]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(texts, tokenizer, window_size=2)
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=6,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            graph_encoder="gcn",
            graph_hidden_dim=16,
            fusion="gated",
            pad_token_id=tokenizer.pad_id,
        )
    )
    batch = torch.tensor([tokenizer.encode(texts[0])[:6]], dtype=torch.long)

    output = model(
        batch,
        graph_data=graph.to_pyg_data(),
        token_node_ids=graph.token_node_ids(tokenizer.vocab_size),
    )

    assert output["logits"].shape == (1, batch.size(1), tokenizer.vocab_size)


def test_graph_builder_supports_ppmi_directed_pruned_context_graph():
    texts = [
        "مجلس قانون را تصویب کرد. دولت لایحه آورد.",
        "مجلس درباره قانون تازه گفت.",
    ]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        window_size=2,
        weighting="ppmi",
        min_edge_weight=0.01,
        top_k=2,
        directed=True,
        graph_scope="sentence",
        context_node_type="sentence",
    )
    data = graph.to_pyg_data()

    assert graph.graph.directed
    assert graph.graph_config["weighting"] == "ppmi"
    assert "sentence" in set(graph.graph.node_types or [])
    assert data.edge_weight.numel() > 0
    assert hasattr(data, "edge_type")


def test_graph_builder_top_k_keeps_undirected_edges_symmetric():
    texts = [
        "الف ب پ ت ث",
        "الف ب ج چ ح",
        "الف پ ج خ د",
    ]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        window_size=3,
        top_k=1,
        directed=False,
    )
    edges = {tuple(edge) for edge in graph.to_pyg_data().edge_index.t().tolist()}

    assert edges
    assert all((dst, src) in edges for src, dst in edges)


def test_graph_lm_checkpoint_is_self_contained_for_generation(tmp_path):
    texts = ["مجلس قانون را تصویب کرد", "دولت لایحه جدید آورد"]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(texts, tokenizer, window_size=2)
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=6,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            graph_encoder="gcn",
            graph_hidden_dim=16,
            fusion="gated",
            pad_token_id=tokenizer.pad_id,
        )
    )
    model.save_pretrained(
        tmp_path,
        tokenizer=tokenizer,
        graph_config=graph.graph_config,
        graph_data=graph.to_pyg_data(),
        token_node_ids=graph.token_node_ids(tokenizer.vocab_size),
    )

    graph_data, token_node_ids = GraphCausalLM.load_graph_artifacts(tmp_path)

    assert (tmp_path / "graph.pt").exists()
    assert graph_data is not None
    assert token_node_ids is not None
    assert graph_data.edge_index.numel() > 0
    assert main(
        [
            "generate",
            "--model",
            str(tmp_path),
            "--prompt",
            "مجلس",
            "--max-new-tokens",
            "1",
            "--device",
            "cpu",
        ]
    ) == 0


def test_train_graph_lm_fits_tokenizer_on_train_split_only(tmp_path):
    corpus = [
        "alphaone betatwo gammathree",
        "deltafour epsilonfive zetasix",
        "etaseven thetaeight iotanine",
        "kappaten lambdaeleven mutwelve",
    ]
    _, validation_corpus = _split_corpus(corpus, validation_ratio=0.25, seed=0)
    validation_tokens = {
        token
        for text in validation_corpus
        for token in PersianTokenizer().tokenize(text)
    }
    output_dir = tmp_path / "lm"

    train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            validation_ratio=0.25,
            block_size=6,
            graph_directed=True,
            graph_weighting="distance",
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=6,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="none",
        ),
        graph_encoder="none",
    )
    tokenizer = PersianTokenizer.load(output_dir / "tokenizer.json")

    assert validation_tokens
    assert validation_tokens.isdisjoint(tokenizer.token_to_id)


def test_dynamic_graph_embeddings_use_only_position_prefixes(monkeypatch):
    import rakhshai_graph_nlp.lm.trainer as trainer_module

    texts = ["الف ب پ"]
    tokenizer = PersianTokenizer().fit(texts)
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=5,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="gcn",
            graph_hidden_dim=8,
            pad_token_id=tokenizer.pad_id,
        )
    )
    trainer = LMTrainer(
        model,
        tokenizer,
        None,
        None,
        config=LMTrainingConfig(dynamic_graph=True, block_size=5, device="cpu"),
        graph_config={"dynamic_graph": True},
    )
    original_builder = trainer_module.build_graph_lm_graph_from_token_ids
    calls = []

    def recording_builder(token_id_sequences, *args, **kwargs):
        calls.append([list(sequence) for sequence in token_id_sequences])
        return original_builder(token_id_sequences, *args, **kwargs)

    monkeypatch.setattr(
        trainer_module,
        "build_graph_lm_graph_from_token_ids",
        recording_builder,
    )
    input_ids = torch.tensor([tokenizer.encode(texts[0])], dtype=torch.long)

    trainer._causal_dynamic_graph_embeddings(input_ids)

    assert calls[0][0] == input_ids[0, :1].tolist()
    assert calls[1][0] == input_ids[0, :2].tolist()
    assert input_ids[0, 2].item() not in calls[1][0]


def test_graph_encoders_are_edge_weight_aware():
    gat = RakhshaiGraphEncoder(
        GraphLMConfig(vocab_size=8, d_model=8, graph_encoder="gat", graph_hidden_dim=4)
    )
    sage = RakhshaiGraphEncoder(
        GraphLMConfig(vocab_size=8, d_model=8, graph_encoder="graphsage", graph_hidden_dim=4)
    )

    assert gat.conv1.edge_dim == 1
    assert isinstance(sage.conv1, WeightedSAGEConv)


def test_graph_lm_context_fusion_all_layers_forward_shape():
    texts = ["مجلس قانون را تصویب کرد", "دولت لایحه جدید آورد"]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(texts, tokenizer, window_size=2, directed=True)
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=6,
            d_model=16,
            n_heads=2,
            n_layers=2,
            dim_feedforward=32,
            graph_encoder="gcn",
            graph_hidden_dim=16,
            fusion="context_gated",
            fusion_layers="all",
            pad_token_id=tokenizer.pad_id,
        )
    )
    batch = torch.tensor([tokenizer.encode(texts[0])[:6]], dtype=torch.long)

    output = model(
        batch,
        graph_data=graph.to_pyg_data(),
        token_node_ids=graph.token_node_ids(tokenizer.vocab_size),
    )

    assert output["logits"].shape == (1, batch.size(1), tokenizer.vocab_size)


def test_lm_cli_train_writes_complete_checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "امروز در تهران باران آمد\n"
        "مجلس قانون جدید را تصویب کرد\n"
        "دولت لایحه تازه‌ای ارائه کرد\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "graph-lm"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--graph-encoder",
            "gcn",
            "--fusion",
            "context_gated",
            "--fusion-layers",
            "all",
            "--tokenizer-type",
            "subword",
            "--graph-weighting",
            "ppmi",
            "--graph-directed",
            "--graph-scope",
            "sentence",
            "--context-node-type",
            "sentence",
            "--dynamic-graph",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    for filename in [
        "model.pt",
        "config.json",
        "tokenizer.json",
        "graph_config.json",
        "generation_config.json",
    ]:
        assert (output_dir / filename).exists()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["history"][0]["perplexity"] > 0

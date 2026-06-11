import json

import torch

from rakhshai_graph_nlp.cli import main
from rakhshai_graph_nlp.lm import (
    GraphCausalLM,
    GraphLMConfig,
    GraphMemoryArtifact,
    GraphMemoryConfig,
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


def test_persian_tokenizer_phase2_normalizer_and_digits():
    tokenizer = PersianTokenizer().fit(["کلاس 123"])

    assert tokenizer.normalize("كلاس ۱۲۳\u0640\u064E") == "کلاس 123"
    assert tokenizer.tokenize("مي\u200c روم") == ["می\u200cروم"]


def test_persian_tokenizer_morphology_and_compound_verbs():
    tokenizer = PersianTokenizer(
        morph_splitting=True,
        compound_verb_mode="join",
    ).fit(["می\u200cروم کتاب‌ها تصمیم گرفت"])

    tokens = tokenizer.tokenize("نمی\u200cروم کتاب‌ها تصمیم گرفت")

    assert "نمی" in tokens
    assert "روم" in tokens
    assert "کتاب" in tokens
    assert "##ها" in tokens
    assert "تصمیم\u200cگرفت" in tokens


def test_bpe_tokenizer_learns_merges_and_round_trips(tmp_path):
    tokenizer = PersianTokenizer(tokenizer_type="bpe", bpe_num_merges=20).fit(
        ["دانشگاه دانشجو دانشمند", "دانشگاه تهران"]
    )

    tokens = tokenizer.tokenize("دانشگاه")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = PersianTokenizer.load(path)

    assert tokenizer.bpe_merges
    assert tokens
    assert loaded.tokenizer_type == "bpe"
    assert loaded.bpe_merges == tokenizer.bpe_merges
    assert loaded.decode(loaded.encode("دانشگاه")) == "دانشگاه"


def test_persian_tokenizer_loads_legacy_missing_type(tmp_path):
    path = tmp_path / "legacy-tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "token_to_id": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3},
                "tokenizer_type": None,
                "special_tokens": {
                    "pad_token": "<pad>",
                    "unk_token": "<unk>",
                    "bos_token": "<bos>",
                    "eos_token": "<eos>",
                },
            }
        ),
        encoding="utf-8",
    )

    tokenizer = PersianTokenizer.load(path)

    assert tokenizer.tokenizer_type == "word"


def test_lm_dataset_creates_next_token_targets():
    tokenizer = PersianTokenizer().fit(["من امروز به مدرسه رفتم"])
    dataset = LMDataset(["من امروز به مدرسه رفتم"], tokenizer, block_size=4)
    input_ids, target_ids = dataset[0]

    assert input_ids.shape == target_ids.shape == (4,)
    assert torch.equal(input_ids[1:], target_ids[:-1])


def test_untrained_graph_lm_starts_near_uniform_loss():
    texts = ["مجلس قانون را تصویب کرد", "دولت لایحه جدید آورد"]
    tokenizer = PersianTokenizer().fit(texts)
    torch.manual_seed(0)
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=6,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            graph_encoder="none",
            pad_token_id=tokenizer.pad_id,
        )
    )
    model.eval()
    batch = torch.tensor([tokenizer.encode(texts[0])[:6]], dtype=torch.long)

    output = model(batch, labels=batch)

    uniform_loss = torch.log(torch.tensor(float(tokenizer.vocab_size)))
    assert float(output["loss"]) < float(uniform_loss) + 1.0
    assert torch.equal(
        model.token_embedding.weight[tokenizer.pad_id],
        torch.zeros(model.config.d_model),
    )


def test_zero_init_gating_matches_baseline_at_init():
    texts = ["مجلس قانون را تصویب کرد", "دولت لایحه جدید آورد"]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(texts, tokenizer, window_size=2)
    torch.manual_seed(0)
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=6,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            graph_encoder="gcn",
            graph_hidden_dim=16,
            fusion="gated",
            fusion_levels="all",
            pad_token_id=tokenizer.pad_id,
        )
    )
    model.eval()
    batch = torch.tensor([tokenizer.encode(texts[0])[:6]], dtype=torch.long)

    with_graph = model(
        batch,
        graph_data=graph.to_pyg_data(),
        token_node_ids=graph.token_node_ids(tokenizer.vocab_size),
    )
    without_graph = model(batch)

    # alpha starts at zero, so the untrained graph model must be exactly the
    # no-graph baseline; graph information only enters once alpha is trained.
    assert torch.allclose(with_graph["logits"], without_graph["logits"])


def test_perplexity_uses_next_token_loss_only(tmp_path):
    corpus = [
        "مجلس قانون جدید را تصویب کرد",
        "دولت لایحه تازه را به مجلس فرستاد",
        "نمایندگان درباره بودجه گفتگو کردند",
        "وزیر گزارش اقتصادی را ارائه داد",
    ]
    metrics = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(tmp_path / "run"),
            epochs=1,
            batch_size=2,
            validation_ratio=0.25,
            block_size=8,
            device="cpu",
            seed=0,
        ),
        graph_encoder="none",
    )

    import math

    row = metrics["history"][0]
    assert row["perplexity"] == math.exp(row["validation_next_token_loss"])
    assert metrics["best_perplexity"] == math.exp(metrics["best_next_token_loss"])
    # The old implementation capped every run at exp(20) computed from the
    # total multi-task loss.
    assert metrics["best_perplexity"] != math.exp(20.0)


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


def test_graph_builder_builds_phase3_multi_relation_graph():
    texts = [
        "کتاب‌ها و کتابخانه برای دانشجویان مفید هستند",
        "کتاب و دانشجو در دانشگاه تهران دیده شدند",
        "دانشگاه و دانشجویان درباره کتابخانه گفتند",
    ]
    tokenizer = PersianTokenizer(
        tokenizer_type="char_chunk",
        morph_splitting=True,
    ).fit(texts)
    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        window_size=2,
        graph_relations=[
            "cooccurrence",
            "pmi",
            "stem",
            "subword",
            "semantic_similarity",
            "word_document",
            "topic_document",
        ],
        relation_weights={"pmi": 0.5, "stem": 0.75},
        semantic_similarity_threshold=0.2,
        semantic_top_k=2,
        topic_top_k=2,
    )
    data = graph.to_pyg_data()
    edge_types = graph.graph_config["edge_types"]

    assert set(graph.graph_config["enabled_relations"]) == {
        "cooccurrence",
        "pmi",
        "stem",
        "subword",
        "semantic_similarity",
        "word_document",
        "topic_document",
    }
    assert edge_types["cooccurrence"] == 0
    assert edge_types["topic_document"] > edge_types["cooccurrence"]
    assert graph.graph_config["relation_weights"]["pmi"] == 0.5
    assert graph.graph_config["node_type_counts"]["document"] == 3
    assert graph.graph_config["node_type_counts"]["topic"] == 2
    assert graph.graph_config["relation_edge_counts"]["word_document"] > 0
    assert data.edge_type.numel() == data.edge_weight.numel()
    assert int(data.edge_type.max()) < len(edge_types)


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


def test_phase9_batched_graph_build_matches_default():
    texts = [
        "کتاب در کتابخانه بود",
        "دانشجو کتاب تازه خواند",
        "دانشگاه کتابخانه بزرگی دارد",
    ]
    tokenizer = PersianTokenizer().fit(texts)
    full = build_graph_lm_graph(
        texts,
        tokenizer,
        graph_relations=["cooccurrence", "pmi", "stem"],
        top_k=3,
    )
    batched = build_graph_lm_graph(
        texts,
        tokenizer,
        graph_relations=["cooccurrence", "pmi", "stem"],
        top_k=3,
        build_batch_size=1,
    )

    assert batched.graph_config["build_batch_size"] == 1
    assert batched.graph_config["graph_build_batches"] == len(texts)
    assert batched.nodes == full.nodes
    assert batched.token_to_node == full.token_to_node
    assert set(map(tuple, batched.edge_index.T.tolist())) == set(
        map(tuple, full.edge_index.T.tolist())
    )


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
    assert (tmp_path / "graph_memory.pt").exists()
    assert (tmp_path / "graph_memory_config.json").exists()
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


def test_phase8_graph_memory_retrieves_prompt_subgraph(tmp_path):
    texts = [
        "مجلس قانون تازه را تصویب کرد",
        "دولت درباره لایحه اقتصادی گزارش داد",
        "دانشگاه کتابخانه جدیدی ساخت",
    ]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        graph_relations=["cooccurrence", "word_document", "topic_document"],
        topic_top_k=2,
    )
    memory = GraphMemoryArtifact.from_graph(graph)
    context = memory.retrieve(
        "مجلس قانون",
        tokenizer,
        config=GraphMemoryConfig(top_k_nodes=4, depth=1, max_edges=8),
    )

    assert context.graph_data is not None
    assert context.token_node_ids is not None
    assert context.graph_data.num_nodes <= 4
    assert context.graph_data.edge_index.size(1) <= 8
    assert context.report["retrieved_nodes"] <= 4
    assert context.report["coverage"] > 0

    memory.save(tmp_path)
    loaded, config = GraphMemoryArtifact.load(tmp_path)
    assert loaded is not None
    assert config.enabled is True
    assert loaded.nodes == memory.nodes


def test_phase8_cli_generate_uses_graph_memory_by_default(tmp_path):
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
        graph_memory=GraphMemoryArtifact.from_graph(graph),
    )
    report_path = tmp_path / "memory-report.json"

    result = main(
        [
            "generate",
            "--model",
            str(tmp_path),
            "--prompt",
            "مجلس",
            "--max-new-tokens",
            "1",
            "--graph-memory-report-path",
            str(report_path),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["enabled"] is True
    assert report["retrieved_nodes"] > 0


def test_phase8_cli_generate_can_disable_graph_memory(tmp_path):
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
        graph_memory=GraphMemoryArtifact.from_graph(graph),
    )

    result = main(
        [
            "generate",
            "--model",
            str(tmp_path),
            "--prompt",
            "مجلس",
            "--max-new-tokens",
            "1",
            "--graph-memory",
            "off",
            "--device",
            "cpu",
        ]
    )

    assert result == 0


def test_phase3_graph_checkpoint_preserves_edge_types(tmp_path):
    texts = [
        "کتاب‌ها در کتابخانه ماندند",
        "کتاب و کتابخانه برای دانشجویان مهم است",
    ]
    tokenizer = PersianTokenizer(morph_splitting=True).fit(texts)
    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        graph_relations=["cooccurrence", "stem", "word_document"],
    )
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            graph_encoder="gcn",
            graph_hidden_dim=16,
            graph_edge_types=len(graph.graph_config["edge_types"]),
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

    assert graph_data is not None
    assert token_node_ids is not None
    assert hasattr(graph_data, "edge_type")
    assert graph_data.edge_type.numel() == graph_data.edge_weight.numel()
    saved_config = json.loads((tmp_path / "graph_config.json").read_text(encoding="utf-8"))
    assert saved_config["edge_types"]["word_document"] == graph.graph_config["edge_types"]["word_document"]


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


def test_phase4_relation_embedding_uses_edge_type_signal():
    torch.manual_seed(0)
    encoder = RakhshaiGraphEncoder(
        GraphLMConfig(
            vocab_size=8,
            d_model=8,
            graph_encoder="graphsage",
            graph_hidden_dim=8,
            graph_edge_types=2,
            graph_relation_mode="embedding",
            dropout=0.0,
        )
    )
    encoder.eval()
    node_features = torch.eye(3, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    from torch_geometric.data import Data

    graph_a = Data(edge_index=edge_index, num_nodes=3)
    graph_a.edge_weight = torch.ones(3)
    graph_a.edge_type = torch.zeros(3, dtype=torch.long)
    graph_b = Data(edge_index=edge_index, num_nodes=3)
    graph_b.edge_weight = torch.ones(3)
    graph_b.edge_type = torch.ones(3, dtype=torch.long)

    out_a = encoder(node_features, graph_a)
    out_b = encoder(node_features, graph_b)

    assert not torch.allclose(out_a, out_b)


def test_phase4_rgcn_handles_missing_edge_type():
    from torch_geometric.data import Data

    encoder = RakhshaiGraphEncoder(
        GraphLMConfig(
            vocab_size=8,
            d_model=8,
            graph_encoder="rgcn",
            graph_hidden_dim=8,
            graph_edge_types=3,
            dropout=0.0,
        )
    )
    graph = Data(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        num_nodes=3,
    )
    graph.edge_weight = torch.ones(3)

    output = encoder(torch.eye(3, 8), graph)

    assert output.shape == (3, 8)


def test_phase4_node_type_metadata_survives_checkpoint(tmp_path):
    texts = [
        "کتاب‌ها در کتابخانه ماندند",
        "دانشجویان کتاب تازه خواندند",
    ]
    tokenizer = PersianTokenizer(morph_splitting=True).fit(texts)
    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        graph_relations=["cooccurrence", "word_document"],
    )
    data = graph.to_pyg_data()
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            graph_encoder="graphsage",
            graph_hidden_dim=16,
            graph_edge_types=len(graph.graph_config["edge_types"]),
            graph_relation_mode="embedding",
            graph_pooling="mean",
            graph_node_importance=True,
            pad_token_id=tokenizer.pad_id,
        )
    )

    model.save_pretrained(
        tmp_path,
        tokenizer=tokenizer,
        graph_config=graph.graph_config,
        graph_data=data,
        token_node_ids=graph.token_node_ids(tokenizer.vocab_size),
    )
    loaded_graph, _ = GraphCausalLM.load_graph_artifacts(tmp_path)

    assert hasattr(data, "node_type_id")
    assert loaded_graph is not None
    assert hasattr(loaded_graph, "node_type_id")
    assert torch.equal(loaded_graph.node_type_id, data.node_type_id)


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


def test_phase5_adaptive_fusion_reports_multi_level_stats():
    texts = [
        "مجلس قانون را تصویب کرد",
        "دولت لایحه جدید آورد",
        "دانشگاه درباره کتابخانه گزارش داد",
    ]
    tokenizer = PersianTokenizer().fit(texts)
    graph = build_graph_lm_graph(
        texts,
        tokenizer,
        window_size=2,
        graph_relations=["cooccurrence", "word_document", "topic_document"],
        topic_top_k=2,
    )
    model = GraphCausalLM(
        GraphLMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dim_feedforward=32,
            graph_encoder="gcn",
            graph_hidden_dim=16,
            fusion="context_gated",
            fusion_levels="token,sentence,subgraph",
            graph_fusion_scale=0.5,
            graph_edge_types=len(graph.graph_config["edge_types"]),
            pad_token_id=tokenizer.pad_id,
        )
    )
    batch = torch.tensor([tokenizer.encode(texts[0])[:8]], dtype=torch.long)

    output = model(
        batch,
        graph_data=graph.to_pyg_data(),
        token_node_ids=graph.token_node_ids(tokenizer.vocab_size),
    )

    assert output["logits"].shape == (1, batch.size(1), tokenizer.vocab_size)
    assert "fusion_stats" in output
    assert "token_graph_share_mean" in output["fusion_stats"]
    assert "sentence_graph_share_mean" in output["fusion_stats"]
    assert "subgraph_graph_share_mean" in output["fusion_stats"]


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
            "--tokenizer-morph-splitting",
            "--tokenizer-compound-verb-mode",
            "join",
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
    assert metrics["tokenizer_stats"]["morph_splitting"] is True


def test_lm_cli_train_accepts_phase5_adaptive_fusion_options(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "کتاب‌ها در کتابخانه ماندند\n"
        "دانشجویان کتاب تازه خواندند\n"
        "دانشگاه درباره کتابخانه گزارش داد\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase5-fusion-lm"

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
            "--fusion-levels",
            "token,sentence,subgraph",
            "--graph-fusion-scale",
            "0.75",
            "--graph-fusion-dropout",
            "0.1",
            "--graph-relations",
            "cooccurrence",
            "word_document",
            "topic_document",
            "--topic-top-k",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
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
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    model_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert model_config["fusion_levels"] == "token,sentence,subgraph"
    assert model_config["graph_fusion_scale"] == 0.75
    assert model_config["graph_fusion_dropout"] == 0.1
    assert "fusion_stats" in metrics
    assert "subgraph_graph_share_mean" in metrics["fusion_stats"]


def test_lm_cli_train_accepts_phase3_graph_relations(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "کتاب‌ها در کتابخانه ماندند\n"
        "دانشجویان کتاب تازه خواندند\n"
        "دانشگاه درباره کتابخانه گزارش داد\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase3-graph-lm"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--graph-encoder",
            "gcn",
            "--graph-relations",
            "cooccurrence",
            "pmi",
            "stem",
            "word_document",
            "topic_document",
            "--relation-weights",
            "pmi=0.5,stem=0.8",
            "--topic-top-k",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
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
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    graph_config = json.loads((output_dir / "graph_config.json").read_text(encoding="utf-8"))
    model_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert "topic_document" in graph_config["edge_types"]
    assert graph_config["relation_weights"]["stem"] == 0.8
    assert model_config["graph_edge_types"] == len(graph_config["edge_types"])


def test_lm_cli_train_accepts_phase4_graph_reasoning_options(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "کتاب‌ها در کتابخانه ماندند\n"
        "دانشجویان کتاب تازه خواندند\n"
        "دانشگاه درباره کتابخانه گزارش داد\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase4-graph-lm"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--graph-encoder",
            "graphsage",
            "--graph-relation-mode",
            "embedding",
            "--graph-pooling",
            "attention",
            "--graph-node-importance",
            "--graph-relations",
            "cooccurrence",
            "stem",
            "word_document",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
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
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    model_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    graph_data, _ = GraphCausalLM.load_graph_artifacts(output_dir)
    assert model_config["graph_relation_mode"] == "embedding"
    assert model_config["graph_pooling"] == "attention"
    assert model_config["graph_node_importance"] is True
    assert graph_data is not None
    assert hasattr(graph_data, "node_type_id")


def test_phase6_multitask_losses_are_enabled_by_default(tmp_path):
    corpus = [
        "مجلس قانون تازه را بررسی کرد",
        "دولت درباره اقتصاد گزارش داد",
        "دانشگاه برای دانشجویان کلاس برگزار کرد",
    ]
    output_dir = tmp_path / "phase6"

    metrics = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            graph_relations=["cooccurrence", "stem", "word_document"],
            graph_relation_mode="embedding",
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="gcn",
            graph_hidden_dim=8,
        ),
        graph_encoder="gcn",
    )

    row = metrics["history"][0]

    assert "masked_token" in row["train_task_losses"]
    assert "edge" in row["train_task_losses"]
    assert "node_relation" in row["train_task_losses"]
    assert row["task_status"]["masked_token"] == "active"
    assert row["task_status"]["edge"] == "active"
    assert metrics["training_config"]["task_losses"].startswith("next_token")


def test_phase6_graph_losses_skip_without_graph(tmp_path):
    corpus = [
        "مدل زبانی متن فارسی را می‌خواند",
        "آموزش چندوظیفه‌ای سیگنال بیشتری می‌دهد",
    ]
    output_dir = tmp_path / "phase6-no-graph"

    metrics = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="none",
        ),
        graph_encoder="none",
    )

    row = metrics["history"][0]

    assert row["task_status"]["masked_token"] == "active"
    assert row["task_status"]["edge"] == "skipped"
    assert "edge" not in row["train_task_losses"]


def test_lm_cli_train_accepts_phase6_multitask_options(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "کتاب‌ها در کتابخانه ماندند\n"
        "دانشجویان کتاب تازه خواندند\n"
        "دانشگاه درباره کتابخانه گزارش داد\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase6-cli"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--graph-encoder",
            "gcn",
            "--task-losses",
            "next_token,masked_token,edge,node_relation,graph_text,sentence_graph",
            "--masked-token-weight",
            "0.2",
            "--edge-prediction-weight",
            "0.05",
            "--node-relation-weight",
            "0.05",
            "--mask-probability",
            "0.2",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--dim-feedforward",
            "16",
            "--graph-hidden-dim",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["training_config"]["masked_token_weight"] == 0.2
    assert metrics["history"][0]["task_status"]["masked_token"] == "active"


def test_phase7_low_data_training_defaults_are_active(tmp_path):
    corpus = [
        "مدل رخشای با داده کم بهتر آموزش می‌بیند",
        "گراف فارسی رابطه واژه‌ها را نگه می‌دارد",
        "آموزش مقاوم جلوی حفظ کردن را می‌گیرد",
    ]
    output_dir = tmp_path / "phase7"

    metrics = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            graph_relations=["cooccurrence", "stem", "word_document"],
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="gcn",
            graph_hidden_dim=8,
        ),
        graph_encoder="gcn",
    )

    row = metrics["history"][0]

    assert metrics["training_config"]["text_augmentation"] is True
    assert metrics["training_config"]["edge_dropout"] > 0
    assert metrics["training_config"]["node_dropout"] > 0
    assert metrics["training_config"]["contrastive_weight"] > 0
    assert metrics["training_config"]["curriculum_learning"] is True
    assert metrics["low_data_training"]["augmented_train_examples"] >= len(corpus)
    assert "generalization_gap" in row
    assert row["task_status"]["contrastive"] in {"active", "skipped"}
    assert metrics["epochs_ran"] == 1
    assert metrics["best_epoch"] == 1


def test_lm_cli_train_accepts_phase7_low_data_options(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "کتاب‌ها در کتابخانه ماندند\n"
        "دانشجویان کتاب تازه خواندند\n"
        "دانشگاه درباره کتابخانه گزارش داد\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase7-cli"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--graph-encoder",
            "gcn",
            "--augmentation-ratio",
            "1.0",
            "--token-dropout",
            "0.1",
            "--punctuation-dropout",
            "1.0",
            "--node-dropout",
            "0.1",
            "--edge-dropout",
            "0.2",
            "--subgraph-sampling-ratio",
            "0.8",
            "--contrastive-weight",
            "0.07",
            "--early-stopping-patience",
            "2",
            "--early-stopping-min-delta",
            "0.001",
            "--max-grad-norm",
            "0.5",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--dim-feedforward",
            "16",
            "--graph-hidden-dim",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    graph_config = json.loads((output_dir / "graph_config.json").read_text(encoding="utf-8"))
    assert metrics["training_config"]["augmentation_ratio"] == 1.0
    assert metrics["training_config"]["edge_dropout"] == 0.2
    assert metrics["training_config"]["contrastive_weight"] == 0.07
    assert metrics["training_config"]["early_stopping_patience"] == 2
    assert metrics["training_config"]["max_grad_norm"] == 0.5
    assert metrics["low_data_training"]["text_augmentation"] is True
    assert graph_config["low_data_training"]["subgraph_sampling_ratio"] == 0.8


def test_phase7_low_data_controls_can_be_disabled(tmp_path):
    corpus = [
        "رخشای متن فارسی را پردازش می‌کند",
        "گراف چندرابطه‌ای کمک آموزشی می‌دهد",
    ]
    output_dir = tmp_path / "phase7-off"

    metrics = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            text_augmentation=False,
            augmentation_ratio=0.0,
            edge_dropout=0.0,
            node_dropout=0.0,
            subgraph_sampling_ratio=1.0,
            contrastive_weight=0.0,
            curriculum_learning=False,
            early_stopping_patience=0,
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="none",
        ),
        graph_encoder="none",
    )

    assert metrics["training_config"]["text_augmentation"] is False
    assert metrics["low_data_training"]["augmented_train_examples"] == len(corpus)
    assert "contrastive" not in metrics["history"][0].get("train_task_losses", {})
    assert metrics["stopped_early"] is False


def test_phase9_graph_cache_reuses_artifact(tmp_path):
    corpus = [
        "رخشای گراف فارسی را می‌سازد",
        "مدل زبانی از گراف کمک می‌گیرد",
        "حافظه گرافی هنگام تولید استفاده می‌شود",
    ]
    cache_dir = tmp_path / "graph-cache"

    first = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(tmp_path / "phase9-cache-first"),
            epochs=1,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            graph_relations=["cooccurrence", "stem"],
            graph_build_batch_size=1,
            graph_cache_dir=str(cache_dir),
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="gcn",
            graph_hidden_dim=8,
        ),
        graph_encoder="gcn",
    )
    second = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(tmp_path / "phase9-cache-second"),
            epochs=1,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            graph_relations=["cooccurrence", "stem"],
            graph_build_batch_size=1,
            graph_cache_dir=str(cache_dir),
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="gcn",
            graph_hidden_dim=8,
        ),
        graph_encoder="gcn",
    )

    assert first["graph_scalability"]["cache_enabled"] is True
    assert first["graph_scalability"]["cache_hit"] is False
    assert second["graph_scalability"]["cache_hit"] is True
    assert (tmp_path / "phase9-cache-first" / "training_state.pt").exists()


def test_phase9_resume_continues_training_state(tmp_path):
    corpus = [
        "رخشای مدل زبانی فارسی است",
        "ادامه آموزش از checkpoint انجام می‌شود",
        "فاز مقیاس پذیری برای داده بزرگ است",
    ]
    output_dir = tmp_path / "phase9-resume"

    first = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="none",
        ),
        graph_encoder="none",
    )
    resumed = train_graph_lm(
        corpus,
        training_config=LMTrainingConfig(
            output_dir=str(output_dir),
            epochs=2,
            batch_size=1,
            validation_ratio=0.0,
            block_size=8,
            resume_from=str(output_dir),
            device="cpu",
            seed=0,
        ),
        model_config=GraphLMConfig(
            vocab_size=1,
            max_seq_len=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dim_feedforward=16,
            graph_encoder="none",
        ),
        graph_encoder="none",
    )

    assert first["epochs_ran"] == 1
    assert resumed["resumed_from"] == str(output_dir)
    assert resumed["history"][0]["epoch"] == 1
    assert resumed["history"][-1]["epoch"] == 2
    assert resumed["epochs_ran"] == 2


def test_phase9_cli_accepts_scalability_options(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "کتاب در کتابخانه ماند\n"
        "دانشجو کتاب تازه خواند\n"
        "گراف فارسی برای مدل مفید است\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "phase9-cli"
    cache_dir = tmp_path / "cli-cache"

    result = main(
        [
            "lm-train",
            "--corpus",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--graph-encoder",
            "gcn",
            "--graph-relations",
            "cooccurrence",
            "stem",
            "--graph-build-batch-size",
            "1",
            "--graph-cache-dir",
            str(cache_dir),
            "--dataloader-num-workers",
            "0",
            "--dataloader-pin-memory",
            "--amp",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--block-size",
            "8",
            "--d-model",
            "8",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--dim-feedforward",
            "16",
            "--graph-hidden-dim",
            "8",
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["training_config"]["graph_build_batch_size"] == 1
    assert metrics["training_config"]["dataloader_pin_memory"] is True
    assert metrics["training_config"]["amp"] is True
    assert metrics["graph_scalability"]["cache_enabled"] is True
    assert metrics["graph_scalability"]["graph_build_batches"] >= 1

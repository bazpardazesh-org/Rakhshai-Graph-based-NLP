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


def test_persian_tokenizer_normalizes_and_encodes():
    tokenizer = PersianTokenizer().fit(["مي روم كلاس", "می\u200cروم کلاس"])

    tokens = tokenizer.tokenize("مي\u200cروم به كلاس")
    ids = tokenizer.encode("مي\u200cروم به كلاس")

    assert "می\u200cروم" in tokens
    assert "كلاس" not in tokens
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id


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
            "gated",
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

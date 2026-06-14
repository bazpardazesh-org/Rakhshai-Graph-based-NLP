"""Regression tests for the six Persian Graph-LM improvements.

Covers: unigram default tokenizer, next-token checkpoint metric, the real
``<mask>`` token, true multi-relation (parallel) edges, non-token node-type
embeddings, and the distributional semantic relation. None require Stanza
(the dependency/lemma backend degrades gracefully when it is absent).
"""

from __future__ import annotations

from collections import Counter

import torch

from rakhshai_graph_nlp.lm.graph_builder import build_graph_lm_graph
from rakhshai_graph_nlp.lm.model import GraphCausalLM, GraphLMConfig
from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer
from rakhshai_graph_nlp.lm.trainer import LMTrainingConfig

TEXTS = [
    "کتاب‌ها در کتابخانه دانشگاه نگهداری می‌شوند",
    "دانشجو کتاب تازه‌ای از کتابخانه گرفت",
    "استاد به دانشجویان درباره کتاب توضیح داد",
    "دانشگاه کتابخانه بزرگی برای پژوهش دارد",
]


# Phase 1 -------------------------------------------------------------------
def test_phase1_operational_tokenizer_default_is_unigram():
    assert LMTrainingConfig().tokenizer_type == "unigram"
    assert LMTrainingConfig().tokenizer_unigram_num_pieces > 0


def test_phase1_unigram_lowers_oov_vs_word():
    train, held_out = TEXTS[:3], [TEXTS[3]]

    def oov(tok_type: str) -> float:
        tok = PersianTokenizer(tokenizer_type=tok_type, unigram_num_pieces=80).fit(train)
        total = unk = 0
        for text in held_out:
            ids = tok.encode(text, add_special_tokens=False)
            total += len(ids)
            unk += sum(1 for i in ids if i == tok.unk_id)
        return unk / max(1, total)

    assert oov("unigram") < oov("word")


# Phase 2 -------------------------------------------------------------------
def test_phase2_checkpoint_metric_defaults_to_next_token():
    assert LMTrainingConfig().checkpoint_metric == "next_token"


# Phase 3 -------------------------------------------------------------------
def test_phase3_real_mask_token():
    tok = PersianTokenizer(tokenizer_type="unigram", unigram_num_pieces=60).fit(TEXTS)
    assert tok.mask_token == "<mask>"
    assert tok.mask_id == 4
    assert tok.id_to_token[4] == "<mask>"
    assert tok.mask_id != tok.unk_id
    restored = PersianTokenizer.from_dict(tok.to_dict())
    assert restored.mask_id == tok.mask_id


def test_phase3_mask_falls_back_to_unk_for_legacy_tokenizer():
    legacy = {
        "tokenizer_type": "word",
        "token_to_id": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "سلام": 4},
        "special_tokens": {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        },
    }
    tok = PersianTokenizer.from_dict(legacy)
    assert tok.mask_id == tok.unk_id == 1


def test_phase3_masked_loss_uses_config_mask_id():
    from rakhshai_graph_nlp.lm.multitask import _masked_token_loss

    tok = PersianTokenizer(tokenizer_type="unigram", unigram_num_pieces=60).fit(TEXTS)
    cfg = GraphLMConfig(
        vocab_size=tok.vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        graph_encoder="none",
        pad_token_id=tok.pad_id,
        mask_token_id=tok.mask_id,
    )
    model = GraphCausalLM(cfg)
    ids = torch.tensor([tok.encode("کتاب در کتابخانه", add_special_tokens=False)[:6]])
    labels = ids.clone()
    loss = _masked_token_loss(
        model,
        ids,
        labels,
        graph_data=None,
        token_node_ids=None,
        graph_embeddings=None,
        mask_probability=1.0,
    )
    assert loss is not None and torch.isfinite(loss)


# Phase 4 -------------------------------------------------------------------
def test_phase4_edges_are_truly_multi_relational():
    tok = PersianTokenizer(tokenizer_type="char_chunk", morph_splitting=True).fit(TEXTS)
    graph = build_graph_lm_graph(
        TEXTS,
        tok,
        window_size=2,
        graph_relations=["cooccurrence", "pmi", "stem", "subword"],
    )
    data = graph.to_pyg_data()
    # Parallel-edge invariant: one type/weight per edge column.
    assert data.edge_type.numel() == data.edge_weight.numel() == data.edge_index.shape[1]
    # At least one node pair must carry more than one relation id.
    pairs = list(map(tuple, data.edge_index.t().tolist()))
    counts = Counter(pairs)
    assert max(counts.values()) >= 2


# Phase 5 -------------------------------------------------------------------
def _non_token_node(data) -> int:
    node_type_id = data.node_type_id
    nonzero = (node_type_id != 0).nonzero(as_tuple=False)
    assert nonzero.numel() > 0, "expected at least one non-token node"
    return int(nonzero[0])


def test_phase5_node_type_embedding_initialises_non_token_nodes():
    tok = PersianTokenizer(tokenizer_type="unigram", unigram_num_pieces=80).fit(TEXTS)
    graph = build_graph_lm_graph(
        TEXTS,
        tok,
        window_size=2,
        graph_relations=["cooccurrence", "word_document"],
    )
    data = graph.to_pyg_data()
    token_node_ids = graph.token_node_ids(tok.vocab_size)
    valid = token_node_ids >= 0
    doc_node = _non_token_node(data)

    base = GraphLMConfig(
        vocab_size=tok.vocab_size,
        d_model=16,
        graph_edge_types=len(graph.graph_config["edge_types"]),
        pad_token_id=tok.pad_id,
    )
    enabled = GraphCausalLM(base)
    assert enabled.node_type_embedding is not None
    with torch.no_grad():
        feats = enabled._node_input_features(data, token_node_ids, valid)
    assert float(feats[doc_node].norm()) > 0  # document node is no longer zero

    disabled_cfg = GraphLMConfig(
        vocab_size=tok.vocab_size,
        d_model=16,
        graph_edge_types=len(graph.graph_config["edge_types"]),
        pad_token_id=tok.pad_id,
        graph_node_type_embedding=False,
    )
    disabled = GraphCausalLM(disabled_cfg)
    assert disabled.node_type_embedding is None
    with torch.no_grad():
        feats0 = disabled._node_input_features(data, token_node_ids, valid)
    assert float(feats0[doc_node].norm()) == 0  # original zero-init behaviour


# Phase 6 -------------------------------------------------------------------
def test_phase6_distributional_is_default_semantic_method():
    tok = PersianTokenizer(tokenizer_type="unigram", unigram_num_pieces=80).fit(TEXTS)
    graph = build_graph_lm_graph(
        TEXTS,
        tok,
        window_size=3,
        graph_relations=["cooccurrence", "semantic_similarity"],
        semantic_similarity_threshold=0.1,
        semantic_top_k=3,
    )
    assert graph.graph_config["semantic_method"] == "distributional"
    assert graph.graph_config["relation_edge_counts"]["semantic_similarity"] > 0


def test_phase6_dependency_in_defaults_with_graceful_backend():
    tok = PersianTokenizer(tokenizer_type="unigram", unigram_num_pieces=80).fit(TEXTS)
    graph = build_graph_lm_graph(TEXTS, tok, window_size=3)
    assert "dependency" in graph.graph_config["enabled_relations"]
    # Stanza may be absent in CI; the backend must still resolve to a value.
    assert graph.graph_config["dependency_backend"] in {"stanza", "heuristic"}

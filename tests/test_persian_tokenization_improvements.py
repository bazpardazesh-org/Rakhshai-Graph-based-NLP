"""Tests for the Persian-specific tokenisation / normalisation improvements.

Covers: punctuation as standalone tokens (no gluing, no silent drops),
Persian numeric separators, configurable hamza folding and ezafe handling,
decode spacing around punctuation, config round-trips, and the Pre-LN
Transformer layer.
"""

import torch

from rakhshai_graph_nlp.features.preprocessing import (
    PersianNormalizer,
    PersianNormalizerConfig,
    normalize_persian_text,
)
from rakhshai_graph_nlp.lm.model import (
    GELUFeedForward,
    GenerationConfig,
    GraphCausalLM,
    GraphLMConfig,
    RMSNorm,
    SwiGLU,
)
from rakhshai_graph_nlp.lm.tokenizer import PersianTokenizer

ZWNJ = "‌"
HAMZA_ABOVE = "ٔ"  # combining hamza above (ه + ٔ form of ezafe)


# --- Punctuation: standalone tokens, no gluing, no silent drops ------------


def test_persian_punctuation_is_not_glued_to_words():
    tokens = PersianTokenizer().tokenize("سلام، حال شما چطور است؟ همین است؛")

    # The Persian comma / question mark / semicolon are their own tokens ...
    assert "،" in tokens
    assert "؟" in tokens
    assert "؛" in tokens
    # ... and "است" is no longer fragmented into "است؟" / "است؛".
    assert "است" in tokens
    assert "است؟" not in tokens
    assert "است؛" not in tokens


def test_ascii_punctuation_is_preserved_not_dropped():
    tokens = PersianTokenizer().tokenize("خوب. عالی! واقعا: بله")

    assert "." in tokens
    assert "!" in tokens
    assert ":" in tokens
    # Sentence-final marks survive, so sentence boundaries are learnable.
    assert tokens.count(".") == 1


def test_parentheses_and_guillemets_are_tokens():
    tokens = PersianTokenizer().tokenize("او گفت «سلام» (با لبخند)")

    for symbol in ("«", "»", "(", ")"):
        assert symbol in tokens


# --- Persian numeric separators --------------------------------------------


def test_persian_decimal_separator_keeps_number_intact():
    tokens = PersianTokenizer().tokenize("قیمت ۱۲٫۵ دلار شد")

    assert "12.5" in tokens
    assert "٫" not in tokens


def test_persian_thousands_separator_keeps_number_intact():
    tokens = PersianTokenizer().tokenize("مبلغ ۱٬۰۰۰ تومان است")

    assert "1,000" in tokens
    assert "٬" not in tokens


# --- Configurable hamza folding --------------------------------------------


def test_hamza_folding_default_on():
    assert normalize_persian_text("مسئول مسائل مؤثر") == "مسیول مسایل موثر"


def test_hamza_folding_can_be_disabled():
    cfg = PersianNormalizerConfig(normalize_hamza=False)
    out = PersianNormalizer(cfg).normalize("مسئول مسائل مؤثر")

    assert out == "مسئول مسائل مؤثر"


# --- Configurable ezafe handling -------------------------------------------


def test_ezafe_marker_is_the_default():
    # Both the precomposed ۀ and the ه + combining-hamza forms are preserved
    # as the explicit ه‌ی marker by default.
    assert PersianNormalizer().normalize("خانهٔ او") == f"خانه{ZWNJ}ی او"
    assert PersianNormalizer().normalize("کافۀ شهر") == f"کافه{ZWNJ}ی شهر"


def test_ezafe_collapse_can_be_selected():
    cfg = PersianNormalizerConfig(ezafe_mode="collapse")
    normalizer = PersianNormalizer(cfg)

    assert normalizer.normalize("خانهٔ او") == "خانه او"
    assert normalizer.normalize("کافۀ شهر") == "کافه شهر"


def test_legacy_config_without_ezafe_field_falls_back_to_collapse():
    # A config serialised before ezafe_mode existed must reproduce the old
    # collapse behaviour so saved artifacts stay faithful.
    cfg = PersianNormalizerConfig.from_dict({"half_space": "preserve"})

    assert cfg.ezafe_mode == "collapse"
    assert PersianNormalizer(cfg).normalize("خانهٔ او") == "خانه او"


def test_ezafe_marker_runs_before_diacritic_removal():
    # The ه + U+0654 form must be rewritten before diacritics are stripped,
    # otherwise the ezafe is silently lost.
    text = "خانه" + HAMZA_ABOVE
    cfg = PersianNormalizerConfig(ezafe_mode="marker")
    assert PersianNormalizer(cfg).normalize(text) == f"خانه{ZWNJ}ی"


def test_invalid_ezafe_mode_raises():
    cfg = PersianNormalizerConfig(ezafe_mode="nonsense")
    try:
        PersianNormalizer(cfg)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for invalid ezafe_mode")


# --- Decode re-attaches punctuation ----------------------------------------


def test_decode_reattaches_punctuation():
    text = "سلام، دنیا؟ (خوب) است."
    tokenizer = PersianTokenizer().fit([text])

    assert tokenizer.decode(tokenizer.encode(text)) == text


# --- Config serialisation round-trips --------------------------------------


def test_normalizer_config_round_trips_new_fields():
    cfg = PersianNormalizerConfig(normalize_hamza=False, ezafe_mode="marker")
    restored = PersianNormalizerConfig.from_dict(cfg.to_dict())

    assert restored.normalize_hamza is False
    assert restored.ezafe_mode == "marker"


def test_tokenizer_save_load_preserves_normalizer_config(tmp_path):
    tokenizer = PersianTokenizer(
        normalizer_config=PersianNormalizerConfig(
            normalize_hamza=False,
            ezafe_mode="marker",
        )
    ).fit(["متن نمونه برای آزمایش"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = PersianTokenizer.load(path)

    assert loaded.normalizer_config.normalize_hamza is False
    assert loaded.normalizer_config.ezafe_mode == "marker"
    # Behaviour survives the round-trip.
    assert loaded.normalize("مسئول") == "مسئول"


# --- Modern decoder stack: RoPE + SwiGLU + RMSNorm (P4) --------------------


def test_modern_architecture_is_the_default():
    config = GraphLMConfig(vocab_size=32, graph_encoder="none")

    assert config.position_encoding == "rope"
    assert config.ffn_type == "swiglu"
    assert config.norm_type == "rmsnorm"


def test_default_model_uses_rope_swiglu_rmsnorm():
    model = GraphCausalLM(GraphLMConfig(vocab_size=32, graph_encoder="none"))

    # RoPE => no learned position table, a rotary module present instead.
    assert model.position_embedding is None
    assert model.rope is not None
    assert isinstance(model.final_norm, RMSNorm)
    for layer in model.transformer_layers:
        assert isinstance(layer.ffn, SwiGLU)
        assert isinstance(layer.norm1, RMSNorm)


def test_legacy_architecture_is_selectable():
    config = GraphLMConfig(
        vocab_size=32,
        graph_encoder="none",
        position_encoding="learned",
        ffn_type="gelu",
        norm_type="layernorm",
    )
    model = GraphCausalLM(config)

    assert model.position_embedding is not None
    assert model.rope is None
    assert isinstance(model.final_norm, torch.nn.LayerNorm)
    assert isinstance(model.transformer_layers[0].ffn, GELUFeedForward)


def test_invalid_architecture_options_raise():
    for kwargs in (
        {"position_encoding": "sinusoidal"},
        {"ffn_type": "relu"},
        {"norm_type": "groupnorm"},
    ):
        try:
            GraphCausalLM(GraphLMConfig(vocab_size=16, graph_encoder="none", **kwargs))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_rope_requires_even_head_dimension():
    # d_model / n_heads must be even for the rotary pairing.
    try:
        GraphCausalLM(
            GraphLMConfig(vocab_size=16, graph_encoder="none", d_model=12, n_heads=4)
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for odd head dimension")


def test_rope_forward_shape():
    config = GraphLMConfig(
        vocab_size=24,
        graph_encoder="none",
        max_seq_len=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
    )
    model = GraphCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    output = model(input_ids)

    assert output["logits"].shape == (2, 8, config.vocab_size)


def test_model_forward_runs_with_modern_stack():
    config = GraphLMConfig(vocab_size=32, graph_encoder="none", max_seq_len=8)
    model = GraphCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 6))

    output = model(input_ids, labels=input_ids)

    assert output["logits"].shape == (2, 6, config.vocab_size)
    assert torch.isfinite(output["loss"])


def test_rmsnorm_matches_reference():
    norm = RMSNorm(8)
    x = torch.randn(4, 8)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)

    assert torch.allclose(norm(x), expected, atol=1e-6)


# --- KV-cache correctness (P4) ---------------------------------------------


def test_kv_cache_matches_full_forward():
    config = GraphLMConfig(vocab_size=40, graph_encoder="none", max_seq_len=32, n_layers=3)
    model = GraphCausalLM(config).eval()
    input_ids = torch.randint(4, 40, (1, 7))

    with torch.no_grad():
        full = model(input_ids)["logits"]
        past = None
        offset = 0
        steps = []
        for position in range(input_ids.size(1)):
            logits, past = model._decode_forward(
                input_ids[:, position : position + 1],
                offset=offset,
                past_key_values=past,
                graph_table=None,
                subgraph=None,
            )
            offset += 1
            steps.append(logits[:, -1, :])
        cached = torch.stack(steps, dim=1)

    assert torch.allclose(full, cached, atol=1e-5)


def test_generate_with_cache_produces_tokens():
    config = GraphLMConfig(vocab_size=24, graph_encoder="none", max_seq_len=16)
    model = GraphCausalLM(config)
    tokenizer = PersianTokenizer().fit(["من به مدرسه رفتم و درس خواندم"])

    text = model.generate(
        "من",
        tokenizer,
        generation_config=GenerationConfig(max_new_tokens=5, eos_token_id=tokenizer.eos_id),
    )

    assert isinstance(text, str)


# --- Real Unigram tokenizer (P3) -------------------------------------------

UNIGRAM_CORPUS = [
    "دانشگاه دانشجو دانشمند دانش",
    "دانشگاه تهران دانشکده دانشجویان",
    "کتاب کتابخانه کتابدار کتابفروشی",
    "خواندن خواندم خواند خوانده می‌خوانم",
]


def test_unigram_trains_a_real_model_not_bpe():
    tokenizer = PersianTokenizer(
        tokenizer_type="unigram", unigram_num_pieces=40
    ).fit(UNIGRAM_CORPUS)

    assert tokenizer.tokenizer_type == "unigram"
    assert tokenizer.unigram_pieces  # learned a piece inventory ...
    assert tokenizer.bpe_merges == []  # ... and did NOT fall back to BPE


def test_unigram_segments_into_subword_pieces():
    tokenizer = PersianTokenizer(
        tokenizer_type="unigram", unigram_num_pieces=40
    ).fit(UNIGRAM_CORPUS)

    pieces = tokenizer.tokenize("دانشگاه")

    assert len(pieces) >= 2
    assert any(piece.startswith("##") for piece in pieces[1:])


def test_unigram_round_trips_through_decode():
    tokenizer = PersianTokenizer(
        tokenizer_type="unigram", unigram_num_pieces=40
    ).fit(UNIGRAM_CORPUS)

    assert tokenizer.decode(tokenizer.encode("دانشگاه تهران")) == "دانشگاه تهران"


def test_unigram_handles_unseen_words_with_char_fallback():
    tokenizer = PersianTokenizer(
        tokenizer_type="unigram", unigram_num_pieces=40
    ).fit(UNIGRAM_CORPUS)

    # An unseen word must still be segmentable (single-character coverage).
    pieces = tokenizer.tokenize("جدیدترین")

    assert pieces  # no crash, non-empty


def test_unigram_save_load_preserves_pieces(tmp_path):
    tokenizer = PersianTokenizer(
        tokenizer_type="unigram", unigram_num_pieces=40
    ).fit(UNIGRAM_CORPUS)
    path = tmp_path / "unigram.json"
    tokenizer.save(path)
    loaded = PersianTokenizer.load(path)

    assert loaded.tokenizer_type == "unigram"
    assert loaded.unigram_pieces == tokenizer.unigram_pieces
    assert loaded.tokenize("دانشگاه") == tokenizer.tokenize("دانشگاه")

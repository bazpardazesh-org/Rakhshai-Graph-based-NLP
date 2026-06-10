from rakhshai_graph_nlp.features.datasets import load_dummy_classification_dataset
from rakhshai_graph_nlp.features.preprocessing import (
    PersianNormalizer,
    PersianNormalizerConfig,
    normalize_persian_text,
    normalise_characters,
    remove_diacritics,
    normalise_whitespace,
    preprocess,
)
from rakhshai_graph_nlp.features.tokenizer import tokenize


def test_load_dummy_classification_dataset():
    docs, labels = load_dummy_classification_dataset()
    assert isinstance(docs, list) and isinstance(labels, list)
    assert len(docs) == len(labels) == 3
    assert labels == [0, 1, 2]
    assert all(isinstance(doc, str) for doc in docs)


def test_preprocessing_steps():
    text = "كبير\u200c\u064E يدرس  \u064Fدرس"
    assert normalise_characters(text) == "کبیر \u064E یدرس  \u064Fدرس"
    assert remove_diacritics(text) == "كبير\u200c يدرس  درس"
    assert normalise_whitespace("a   b\t c\n") == "a b c"
    processed = preprocess(text)
    assert processed == "کبیر یدرس درس"


def test_phase2_shared_persian_normalizer_modes():
    text = "كلاس\u200cها ۱۲۳"

    assert normalize_persian_text(text) == "کلاس\u200cها 123"
    assert normalize_persian_text(text, half_space="split") == "کلاس ها 123"
    normalizer = PersianNormalizer(PersianNormalizerConfig(half_space="remove"))
    assert normalizer.normalize(text) == "کلاسها 123"


def test_tokenize_fallback():
    tokens = tokenize("سلام دنیا!")
    assert tokens == ["سلام", "دنیا"]

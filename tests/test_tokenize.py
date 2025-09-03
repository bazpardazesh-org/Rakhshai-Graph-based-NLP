from rakhshai_graph_nlp.features.tokenize import split_sentences, tokenize


def test_unicode_tokenization():
    text = "سلام دنیا! Hello world؟ خوب؛"
    tokens = tokenize(text)
    assert "سلام" in tokens and "Hello" in tokens


def test_sentence_split():
    text = "سلام دنیا! چه خبر؟ خوبه."
    sents = split_sentences(text)
    assert len(sents) == 3

import numpy as np

from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.tasks.classification import train_gcn_classifier
from rakhshai_graph_nlp.tasks.hate_speech import contains_hate_speech
from rakhshai_graph_nlp.tasks.recommendation import recommend_similar
from rakhshai_graph_nlp.tasks.social_analysis import compute_social_embeddings
from rakhshai_graph_nlp.tasks.summarization import (
    GATSummarizer,
    _pagerank,
    split_sentences,
    textrank_summarise,
)

def test_train_gcn_classifier():
    g = Graph(nodes=["a", "b", "c"], adjacency=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    X = np.eye(3)
    labels = np.array([0, 1, 2])
    clf, losses = train_gcn_classifier(g, X, labels, num_epochs=5)
    assert len(losses) == 5
    preds = clf.predict(g, X)
    assert preds.shape == (3,)

def test_contains_hate_speech():
    text = "این متن بد است"
    assert contains_hate_speech(text, ["بد"])
    assert not contains_hate_speech(text, ["خوب"])

def test_recommend_similar():
    docs = ["این یک تست است", "تست دیگری", "سلام دنیا"]
    recs = recommend_similar("تست", docs, top_k=2)
    assert len(recs) == 2
    assert recs[0][0] in (0, 1)

def test_compute_social_embeddings():
    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    feats = np.random.rand(2, 4)
    emb = compute_social_embeddings(g, feats, hidden_dims=[3, 2])
    assert emb.shape == (2, 2)

def test_summarization_functions():
    text = "جمله اول. جمله دوم! جمله سوم؟"
    sentences = split_sentences(text)
    assert sentences == ["جمله اول", "جمله دوم", "جمله سوم"]

    adjacency = np.array([[0, 1], [1, 0]], dtype=float)
    pr = _pagerank(adjacency)
    assert np.isclose(pr.sum(), 1.0)

    summary = textrank_summarise("این جمله اول است. این جمله دوم است. این جمله سوم است.", top_k=2)
    assert len(summary.splitlines()) == 2

    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    X = np.random.rand(2, 4)
    summarizer = GATSummarizer(4, 3, 2)
    idx = summarizer.summarise(g, X, top_k=1)
    assert idx.shape == (1,)

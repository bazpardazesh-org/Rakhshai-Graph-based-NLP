import numpy as np
import pytest

from rakhshai_graph_nlp.features.pyg_data import graph_to_data
from rakhshai_graph_nlp.graphs.graph import Graph
from rakhshai_graph_nlp.tasks.classification import (
    TextGraphClassifier,
    _select_device,
    train_gcn_classifier,
    train_node_classifier,
)
from rakhshai_graph_nlp.tasks.hate_speech import (
    HateSpeechDetector,
    contains_hate_speech,
)
from rakhshai_graph_nlp.tasks.recommendation import recommend_similar
from rakhshai_graph_nlp.tasks.social_analysis import compute_social_embeddings
from rakhshai_graph_nlp.tasks.summarization import (
    GATSummarizer,
    _pagerank,
    _build_sentence_graph,
    gat_summarise,
    split_sentences,
    textrank_summarise,
)

def test_train_gcn_classifier():
    g = Graph(nodes=["a", "b", "c"], adjacency=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    X = np.eye(3)
    labels = np.array([0, 1, 2])
    clf, losses = train_gcn_classifier(g, X, labels, num_epochs=5)
    assert len(losses) == 5
    data = graph_to_data(g, features=X, labels=labels)
    preds = clf.predict(data)
    assert preds.shape == (3,)


def test_train_node_classifier_supports_feature_fallback_and_mask():
    g = Graph(
        nodes=["a", "b", "c"],
        adjacency=np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=float,
        ),
    )
    labels = np.array([0, 1, 0])
    mask = np.array([True, True, False])

    clf, losses = train_node_classifier(
        g,
        labels,
        mask=mask,
        model_type="graphsage",
        hidden_dim=4,
        num_epochs=2,
        dropout=0.0,
    )

    assert len(losses) == 2
    assert all(np.isfinite(loss) for loss in losses)
    data = graph_to_data(g, features=np.eye(3), labels=labels)
    assert clf.predict(data).shape == (3,)


def test_train_node_classifier_validates_label_and_mask_lengths():
    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))

    with pytest.raises(ValueError, match="labels must match"):
        train_node_classifier(g, np.array([0]), num_epochs=1)

    with pytest.raises(ValueError, match="mask must match"):
        train_node_classifier(g, np.array([0, 1]), mask=np.array([True]), num_epochs=1)


def test_text_graph_classifier_fit_predict_evaluate_and_save_load(tmp_path):
    texts = [
        "انتخابات دولت مجلس",
        "قانون دولت مجلس",
        "فوتبال تیم گل",
        "مسابقه تیم فوتبال",
    ]
    labels = ["politics", "politics", "sports", "sports"]
    clf = TextGraphClassifier(
        hidden_dim=4,
        num_epochs=3,
        dropout=0.0,
        learning_rate=0.01,
        seed=0,
    )

    clf.fit(texts, labels)
    preds = clf.predict(["دولت و مجلس", "تیم فوتبال"])
    report = clf.evaluate(texts, labels)

    assert len(preds) == 2
    assert set(preds) <= {"politics", "sports"}
    assert report["count"] == 4
    assert 0.0 <= report["accuracy"] <= 1.0
    assert 0.0 <= report["macro_f1"] <= 1.0

    model_dir = tmp_path / "text-graph-model"
    clf.save(model_dir)

    loaded = TextGraphClassifier.load(model_dir)

    assert loaded.predict(["دولت و مجلس"]) == clf.predict(
        ["دولت و مجلس"]
    )


def test_select_device_falls_back_when_cuda_is_unavailable():
    if _select_device("cuda").type == "cpu":
        assert _select_device("cuda").type == "cpu"


def test_contains_hate_speech():
    text = "این متن بد است"
    assert contains_hate_speech(text, ["بد"])
    assert not contains_hate_speech(text, ["خوب"])


def test_contains_hate_speech_checks_terms_lazily():
    terms_checked = []

    def terms():
        for term in ["بد", "خوب"]:
            terms_checked.append(term)
            yield term

    assert contains_hate_speech("این متن بد است", terms())
    assert terms_checked == ["بد"]
    assert not contains_hate_speech("متن معمولی", ["بد", "زشت"])


def test_hate_speech_detector_fit_predict_save_load(tmp_path):
    texts = [
        "متن آرام و محترمانه",
        "گفتگوی خوب و عادی",
        "توهین بد و نفرت",
        "پیام بد و سمی",
    ]
    labels = [False, False, True, True]
    detector = HateSpeechDetector(
        hidden_dim=4,
        num_epochs=3,
        learning_rate=0.01,
        seed=0,
    )

    detector.fit(texts, labels)
    preds = detector.predict(["متن محترمانه", "پیام نفرت"])
    report = detector.evaluate(texts, labels)

    assert len(preds) == 2
    assert all(isinstance(pred, bool) for pred in preds)
    assert report["count"] == 4

    model_dir = tmp_path / "hate-detector"
    detector.save(model_dir)
    loaded = HateSpeechDetector.load(model_dir)

    assert loaded.predict_labels(["متن محترمانه"]) == detector.predict_labels(
        ["متن محترمانه"]
    )


def test_recommend_similar():
    docs = ["این یک تست است", "تست دیگری", "سلام دنیا"]
    recs = recommend_similar("تست", docs, top_k=2)
    assert len(recs) == 2
    assert recs[0][0] in (0, 1)


def test_recommend_similar_returns_sorted_document_indices_and_scores():
    pytest.importorskip("sklearn")

    docs = [
        "apple banana",
        "car bus",
        "apple apple banana",
    ]
    recs = recommend_similar("apple banana", docs, top_k=3)

    assert [idx for idx, _ in recs] == [0, 2, 1]
    assert recs[0][1] >= recs[1][1] >= recs[2][1]
    assert all(isinstance(idx, int) for idx, _ in recs)
    assert all(isinstance(score, float) for _, score in recs)


def test_recommend_similar_clamps_top_k_to_available_documents():
    pytest.importorskip("sklearn")

    recs = recommend_similar("apple", ["apple", "banana"], top_k=10)

    assert len(recs) == 2


def test_compute_social_embeddings():
    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    feats = np.random.rand(2, 4)
    emb = compute_social_embeddings(g, feats, hidden_dims=[3, 2])
    assert emb.shape == (2, 2)


def test_compute_social_embeddings_supports_sampled_neighbors():
    g = Graph(
        nodes=["a", "b", "c", "d"],
        adjacency=np.array(
            [
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=float,
        ),
    )
    feats = np.arange(16, dtype=float).reshape(4, 4)

    emb = compute_social_embeddings(g, feats, hidden_dims=[5, 2], num_samples=1)

    assert emb.shape == (4, 2)
    assert np.isfinite(emb).all()


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


def test_build_sentence_graph_rejects_empty_sentence_list():
    pytest.importorskip("sklearn")

    with pytest.raises(ValueError, match="No sentences provided"):
        _build_sentence_graph([])


def test_textrank_summary_preserves_original_sentence_order():
    pytest.importorskip("sklearn")

    text = (
        "alpha beta beta. "
        "alpha beta. "
        "gamma delta."
    )

    summary = textrank_summarise(text, top_k=2)

    assert summary.splitlines() == ["alpha beta beta ", "alpha beta"]


def test_pagerank_handles_dangling_rows():
    adjacency = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )

    scores = _pagerank(adjacency)

    assert scores.shape == (3,)
    assert np.isclose(scores.sum(), 1.0)
    assert np.isfinite(scores).all()


def test_gat_summarizer_limits_selection_to_available_nodes():
    g = Graph(nodes=["a", "b"], adjacency=np.array([[0, 1], [1, 0]], dtype=float))
    X = np.ones((2, 3), dtype=float)
    summarizer = GATSummarizer(3, 2, 2)

    idx = summarizer.summarise(g, X, top_k=5)

    assert idx.shape == (2,)
    assert set(idx.tolist()) == {0, 1}


def test_gat_summarise_returns_requested_sentence_count():
    pytest.importorskip("sklearn")

    summary = gat_summarise(
        "alpha beta beta. alpha beta. gamma delta.",
        top_k=2,
        hidden_dim=4,
        output_dim=2,
        random_state=0,
    )

    assert len(summary.splitlines()) == 2

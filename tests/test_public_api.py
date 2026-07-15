import rakhshai_graph_nlp as rgnn
from rakhshai_graph_nlp import api


def test_stable_api_metadata_and_exports():
    assert rgnn.__version__ == "2.2.1"
    assert rgnn.API_STATUS == "stable"
    assert rgnn.__api_version__ == "2.2"
    assert api.API_STATUS == "stable"
    assert api.stable_api() == rgnn.stable_api()

    expected = {
        "Graph",
        "build_text_graph",
        "build_cooccurrence_graph",
        "TextGraphClassifier",
        "PersianTokenizer",
        "GraphCausalLM",
        "LMTrainingConfig",
        "train_graph_lm",
        "GraphMemoryConfig",
        "PoemRecommender",
        "ArticleCorpusConfig",
        "ArticleAuditConfig",
        "ArticleAblationConfig",
        "ArticleTrainingConfig",
        "ArticleGenerationConfig",
        "PersianArticle",
        "audit_article_corpus",
        "prepare_article_corpus",
        "train_article_llm",
        "run_article_ablation",
        "generate_persian_article",
        "tokenize_persian",
        "graph_to_data",
        "accuracy",
        "macro_f1",
    }

    assert expected <= set(api.stable_api())
    assert expected <= set(rgnn.__all__)
    for name in expected:
        assert getattr(rgnn, name) is getattr(api, name)


def test_stable_api_keeps_subpackage_shortcuts():
    assert rgnn.graphs.Graph is rgnn.Graph
    assert rgnn.graphs.build_text_graph is rgnn.build_text_graph
    assert hasattr(rgnn.graphs, "sparse")
    assert rgnn.features.tokenize_persian is rgnn.tokenize_persian
    assert rgnn.models.GCNClassifier is rgnn.GCNClassifier
    assert rgnn.tasks.TextGraphClassifier is rgnn.TextGraphClassifier
    assert rgnn.lm.GraphCausalLM is rgnn.GraphCausalLM
    assert rgnn.llm.article.PersianArticle is rgnn.PersianArticle
    assert rgnn.article_llm.PersianArticle is rgnn.PersianArticle

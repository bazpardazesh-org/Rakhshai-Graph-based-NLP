"""Task‑specific utilities.

The ``rakhshai_graph_nlp.tasks`` subpackage contains higher‑level functions and
classes for performing specific analyses on text graphs. These include
classification, summarisation, recommendation and social network
analysis.  The implementation of these tasks is deliberately kept
lightweight to avoid heavy dependencies; where possible, algorithms
are implemented using NumPy.  You can plug in your own models by
subclassing the provided classes.
"""

from .classification import (
    TextGraphClassifier,
    TextGraphClassifierConfig,
    train_gcn_classifier,
    train_node_classifier,
)
from .hate_speech import HateSpeechDetector, contains_hate_speech
from .recommendation import recommend_similar
from .social_analysis import compute_social_embeddings
from .summarization import GATSummarizer, gat_summarise, textrank_summarise

__all__ = [
    "TextGraphClassifier",
    "TextGraphClassifierConfig",
    "train_node_classifier",
    "train_gcn_classifier",
    "textrank_summarise",
    "gat_summarise",
    "GATSummarizer",
    "recommend_similar",
    "contains_hate_speech",
    "HateSpeechDetector",
    "compute_social_embeddings",
]

"""Task‑specific utilities.

The ``rakhshai_graph_nlp.tasks`` subpackage contains higher‑level functions and
classes for performing specific analyses on text graphs. These include
classification, summarisation, recommendation and social network
analysis.  The implementation of these tasks is deliberately kept
lightweight to avoid heavy dependencies; where possible, algorithms
are implemented using NumPy.  You can plug in your own models by
subclassing the provided classes.
"""

__all__ = [
    "classification",
    "summarization",
    "recommendation",
    "hate_speech",
    "social_analysis",
]
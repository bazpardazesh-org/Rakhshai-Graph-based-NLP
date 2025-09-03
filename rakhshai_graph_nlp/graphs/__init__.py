"""Graph construction utilities.

The ``rakhshai_graph_nlp.graphs`` subpackage collects functions for converting
raw Persian text into graph structures. These graphs can be used as
input to graph neural networks or other graph‑based algorithms. Each
module corresponds to a particular graph construction strategy, for
example co‑occurrence graphs or document similarity graphs.

The primary entry points are:

* :func:`rakhshai_graph_nlp.graphs.co_occurrence.build_cooccurrence_graph` –
  Build a word co‑occurrence graph where nodes represent words and
  weighted edges represent the frequency with which two words appear
  together within a sliding window. The TextGCN model uses a similar
  strategy, assigning TF‑IDF weights to document–word edges and PMI
  weights to word–word edges【136383271440271†L794-L812】.

* :func:`rakhshai_graph_nlp.graphs.document.build_document_graph` – Construct a
  graph of documents in which nodes are documents and edges encode
  similarity between documents measured via TF‑IDF or sentence
  embeddings. Document graphs are useful for tasks like recommendation
  or clustering.

* :func:`rakhshai_graph_nlp.graphs.dependency.build_dependency_graph` – Use a
  dependency parser (e.g. Stanza) to derive a syntactic graph
  connecting tokens according to their grammatical relations
  【534291576090157†L11-L33】.

* :func:`rakhshai_graph_nlp.graphs.semantic.build_semantic_graph` – Construct
  a semantic graph from a list of words using an optional mapping of
  semantic relations (for example synonyms). This offers a lightweight
  alternative until a comprehensive Persian WordNet becomes available.

See the individual modules for further details and usage examples.
"""

from .graph import Graph

# Expose constructors
from .co_occurrence import build_cooccurrence_graph  # noqa: F401
from .document import build_document_graph  # noqa: F401
from .dependency import build_dependency_graph  # noqa: F401
from .semantic import build_semantic_graph  # noqa: F401
from .text_graph import build_text_graph  # noqa: F401

__all__ = ["Graph"]
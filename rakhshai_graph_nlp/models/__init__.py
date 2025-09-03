"""Neural models for graph representation learning.

This subpackage contains simple, dependency‑free implementations of
graph neural network layers and models. They are designed to be
sufficient for small to medium scale experiments but are not
optimised for speed.  If you require more advanced or scalable
implementations, consider integrating external frameworks such as
PyTorch Geometric or DGL.
"""

__all__ = ["gcn", "gat", "graphsage"]
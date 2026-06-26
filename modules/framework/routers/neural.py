"""Backward-compat re-export.  Canonical: kalman.filters.routers.neural.

NeuralRouter is now a concrete class that accepts a ``model=`` parameter
instead of requiring subclassing::

    from kalman.filters.routers import NeuralRouter
    from kalman.filters.routers.nn import MLP

    router = NeuralRouter(model=MLP(hidden=32), window=5)
"""
from ...filters.routers.neural import NeuralRouter  # noqa: F401

__all__ = ["NeuralRouter"]

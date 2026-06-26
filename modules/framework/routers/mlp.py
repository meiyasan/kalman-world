"""MLPRouter — backward-compat alias for NeuralRouter(model=MLP(...)).

Prefer the explicit form::

    from kalman.filters.routers import NeuralRouter
    from kalman.filters.routers.nn import MLP

    router = NeuralRouter(model=MLP(hidden=16), window=5)
"""
from ...filters.routers.neural import NeuralRouter
from ...filters.routers.nn.mlp import MLP


class MLPRouter(NeuralRouter):
    """Backward-compat alias: ``NeuralRouter(model=MLP(hidden, window))``."""

    def __init__(self, hidden: int = 16, window: int = 5):
        super().__init__(model=MLP(hidden=hidden), window=window)


__all__ = ["MLPRouter"]

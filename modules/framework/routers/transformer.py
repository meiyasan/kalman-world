"""Backward-compat: TransformerRouter → NeuralRouter(model=TransformerNN()).

The old subclassing pattern is superseded by the pluggable model= parameter::

    from kalman.filters.routers import NeuralRouter
    from kalman.filters.routers.nn import TransformerNN

    class MyArch(TransformerNN):
        def build(self, d_in, d_out, seed=None): ...
        def fit(self, X, y, **kw): ...
        def __call__(self, x): ...

    router = NeuralRouter(model=MyArch(), window=16)

TransformerRouter remains importable as a thin alias.
"""
from ...filters.routers.neural import NeuralRouter
from ...filters.routers.nn.transformer import TransformerNN


class TransformerRouter(NeuralRouter):
    """Backward-compat alias.  Prefer ``NeuralRouter(model=MyTransformerNN())``."""

    def __init__(self, window: int = 5):
        super().__init__(model=TransformerNN(), window=window)


__all__ = ["TransformerRouter"]

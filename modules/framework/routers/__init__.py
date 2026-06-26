"""Backward-compat re-exports.  Canonical location: kalman.filters.routers.

Hierarchy (canonical)::

    KalmanRouter          filters/routers/base.py
    ├── BayesianRouter    filters/routers/bayesian.py
    ├── ThresholdRouter   filters/routers/threshold.py
    ├── NeuralRouter      filters/routers/neural.py  (model= parameter)
    └── HybridRouter      filters/routers/hybrid.py

Neural architectures (filters/routers/nn/)::

    MLP             working pure-numpy baseline
    TransformerNN   stub — override 3 methods
    LSTM            stub — override 3 methods
    CNN             stub — override 3 methods

Backward-compat aliases defined here::

    MLPRouter        → NeuralRouter(model=MLP())
    TransformerRouter → NeuralRouter(model=TransformerNN())
"""
from ...filters.routers.base      import KalmanRouter
from ...filters.routers.bayesian  import BayesianRouter
from ...filters.routers.threshold import ThresholdRouter
from ...filters.routers.neural    import NeuralRouter
from ...filters.routers.hybrid    import HybridRouter
from .mlp         import MLPRouter
from .transformer import TransformerRouter

__all__ = [
    "KalmanRouter",
    "BayesianRouter",
    "ThresholdRouter",
    "NeuralRouter",
    "HybridRouter",
    "MLPRouter",
    "TransformerRouter",
]

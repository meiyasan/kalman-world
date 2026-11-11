"""IKF model routers — pluggable mode-probability update strategies.

Hierarchy::

    KalmanRouter          abstract contract  (framework/routers/base.py)
    ├── BayesianRouter    analytical: μ_j ∝ c_j · N(ν_j; 0, S_j)
    ├── ThresholdRouter   rule-based: closest NIS to χ²(m) mean
    ├── NeuralRouter      model=callable, owns feature extraction
    └── HybridRouter      geometric blend of any two routers

Neural architectures (plug into NeuralRouter as model=...)::

    from kalman.framework.routers.nn import MLP, TransformerNN, LSTM, CNN
    router = NeuralRouter(model=MLP(hidden=32), window=5)
"""
from .base      import KalmanRouter
from .bayesian  import BayesianRouter
from .threshold import ThresholdRouter
from .neural    import NeuralRouter
from .hybrid    import HybridRouter

__all__ = [
    "KalmanRouter",
    "BayesianRouter",
    "ThresholdRouter",
    "NeuralRouter",
    "HybridRouter",
]

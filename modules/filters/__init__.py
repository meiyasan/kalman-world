"""
Public Kalman filter terminology (KF / EKF / UKF / PF / IKF).

Each filter is a thin :class:`~kalman.framework.filter.KalmanFilter` leaf that wraps a model
with the matching engine (selected by ``KIND``), exposing ``forward()`` / ``backward()``.
Construct by wrapping a model, e.g. ``LinearKalmanFilter(OscillatorModel("s", fs=100))``.

IMM routing classes are exported here for convenience::

    from kalman.filters import (
        InteractiveKalmanFilter,
        NeuralRouter, BayesianRouter, ThresholdRouter, HybridRouter,
        MLP,
    )
"""
from .kf  import LinearKalmanFilter
from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter
from .pf  import ParticleFilter
from .ckf import CombinatorialKalmanFilter
from .ikf import InteractiveKalmanFilter, NeuralInteractiveKalmanFilter

from .routers          import (KalmanRouter, BayesianRouter,
                                ThresholdRouter, NeuralRouter, HybridRouter)
from .routers.nn       import MLP, TransformerNN, LSTM, CNN

# Backward-compat: MLPRouter and TransformerRouter still importable
from ..framework.routers import MLPRouter, TransformerRouter

__all__ = [
    "LinearKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "CombinatorialKalmanFilter",
    "InteractiveKalmanFilter",
    "NeuralInteractiveKalmanFilter",
    # routers
    "KalmanRouter",
    "BayesianRouter",
    "ThresholdRouter",
    "NeuralRouter",
    "HybridRouter",
    # nn architectures
    "MLP",
    "TransformerNN",
    "LSTM",
    "CNN",
    # backward-compat
    "MLPRouter",
    "TransformerRouter",
]

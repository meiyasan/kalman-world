"""
Public Kalman filter terminology (KF / EKF / UKF / PF / IKF).

Each filter is a thin :class:`~kalman.framework.filter.KalmanFilter` leaf that wraps a model
with the matching engine (selected by ``KIND``), exposing ``forward()`` / ``backward()``.
Construct by wrapping a model, e.g. ``LinearKalmanFilter(OscillatorModel("s", fs=100))``.

IMM routing classes live in ``kalman.framework.routers``::

    from kalman.filters import InteractiveKalmanFilter
    from kalman.framework.routers import NeuralRouter, BayesianRouter
    from kalman.framework.routers.nn import MLP
"""
from .kf  import LinearKalmanFilter
from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter
from .pf  import ParticleFilter
from .ckf import CombinatorialKalmanFilter
from .ikf import InteractiveKalmanFilter

__all__ = [
    "LinearKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "CombinatorialKalmanFilter",
    "InteractiveKalmanFilter",
]

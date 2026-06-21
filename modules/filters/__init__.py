"""
Public Kalman filter terminology (KF / EKF / UKF / PF).

Each is a thin :class:`~kalman.framework.filter.KalmanFilter` leaf that wraps a model
with the matching engine (selected by ``KIND``), exposing ``forward()`` / ``backward()``.
Construct by wrapping a model, e.g. ``LinearKalmanFilter(OscillatorModel("s", fs=100))``.
"""
from .kf import LinearKalmanFilter
from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter
from .pf import ParticleFilter
from .ckf import CombinatorialKalmanFilter

__all__ = [
    "LinearKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "CombinatorialKalmanFilter",
]

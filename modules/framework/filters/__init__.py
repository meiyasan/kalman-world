"""
Public Kalman filter terminology (KF / EKF / UKF / PF / EnKF / CKF).

Concretes for the **Filter** role, standing to ``framework/filter.py`` exactly as
``engines/`` stands to ``engine.py`` and ``models/`` to ``model.py``.

Each filter is a thin :class:`~kalman.framework.filter.KalmanFilter` leaf that wraps a model
with the matching engine (selected by ``KIND``), exposing ``forward()`` / ``backward()``.
Construct by wrapping a model, e.g. ``LinearKalmanFilter(OscillatorModel("s", fs=100))``.

A system that changes regime has no filter of its own: ``route(models, Pi)``
makes the model, which the ordinary filter runs on the interacting engine. Its
decision layers live in ``kalman.framework.routers``.
"""
from .kf  import LinearKalmanFilter
from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter
from .pf  import ParticleFilter
from .ckf import CombinatorialKalmanFilter
from .enkf import EnsembleKalmanFilter

__all__ = [
    "LinearKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "CombinatorialKalmanFilter",
    "EnsembleKalmanFilter",
]

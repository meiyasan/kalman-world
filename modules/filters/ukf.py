"""Unscented Kalman Filter (UKF) — wraps a model with the sigma-point engine."""
from __future__ import annotations

from ..framework.filter import KalmanFilter
from ..framework.constants import KalmanEngineKind


class UnscentedKalmanFilter(KalmanFilter):
    """
    Unscented (sigma-point) Kalman filter. Non-additive noise (``f(...,w)`` /
    ``h(...,v)``) is handled automatically by the engine via augmented-state sigma
    points::

        ukf = UnscentedKalmanFilter(my_model, alpha=1e-3)
        rec = ukf.forward(zk)
        rec_s = ukf.backward()
    """

    NAME = "Unscented Kalman Filter"
    DESCRIPTION = "Unscented Kalman filter (UKF): sigma-point engine over a model."
    KIND = KalmanEngineKind.UNSCENTED

    def __init__(self, model=None, *, alpha=1e-3, beta=2.0, kappa=0.0, **kwargs):
        from ..framework.engines.unscented import UnscentedKalmanEngine
        super().__init__(model, engine=UnscentedKalmanEngine(alpha=alpha, beta=beta, kappa=kappa), **kwargs)

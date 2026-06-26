"""Extended Kalman engine (EKF) — Jacobian linearisation around the current state."""
from __future__ import annotations

from ..engine import KalmanEngine
from ..constants import KalmanEngineKind


class ExtendedKalmanEngine(KalmanEngine):
    """
    EKF step: propagate with the true (nonlinear) ``model.f(x,u,t)`` and linearise the
    covariance with the Jacobian ``model.F(x=x,u=u,t)``; likewise ``model.h(x,t)`` /
    ``model.H(x=x,t)`` on update.

    On a matrix (linear) model this reduces exactly to the linear computation
    (``f(x)=Fx``, ``F(x)=F``), so an Extended filter runs correctly on a linear model.
    """

    NAME = "Extended Kalman Engine"
    DESCRIPTION = "EKF step: nonlinear f/h with Jacobian linearisation (falls back to linear)."
    VERSION = "1.0"
    KIND = KalmanEngineKind.EXTENDED

    def _propagate(self, model, x, u, t):
        F = model.F(x=x, u=u, t=t)
        xp = model.f(x, u=u, t=t)
        return xp, F

    def _measure(self, model, x, t):
        H = model.H(x=x, t=t)
        return H, model.h(x, t=t)

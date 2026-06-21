"""Linear Kalman engine (KF) — constant-matrix predict/update."""
from __future__ import annotations

from ..engine import KalmanEngine
from ..constants import EngineKind


class LinearKalmanEngine(KalmanEngine):
    """
    Standard linear Kalman step: ``x⁻ = F x + B u``, ``z_pred = H x`` using the model's
    **constant** matrices (``model.F()/H()/B()``). For a nonlinear model this would
    ignore the nonlinearity, which is why nonlinear models disallow this engine.
    """

    NAME = "Linear Kalman Engine"
    DESCRIPTION = "Standard linear Kalman step (constant F/H matrices)."
    VERSION = "1.0"
    KIND = EngineKind.LINEAR

    def _propagate(self, model, x, u, t):
        F = model.F(t=t)
        xp = F @ x
        if u is not None:
            B = model.B(t=t)
            xp = xp + (u if B is None else B @ u)
        return xp, F

    def _measure(self, model, x, t):
        H = model.H(t=t)
        return H, H @ x

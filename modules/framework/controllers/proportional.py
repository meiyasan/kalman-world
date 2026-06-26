"""ProportionalKalmanController — u = −Kp · (x̂ − x_ref)."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class ProportionalKalmanController(KalmanController):
    """Proportional controller: ``u = −Kp · (x̂ − x_ref)``."""

    def __init__(self, model, Kp, **kw):
        super().__init__(model, **kw)
        self.Kp = np.asarray(Kp, dtype=float)

    def _policy(self, x_hat, P, t, *, nu=None):
        e = x_hat - self.x_ref
        return -self.Kp @ e

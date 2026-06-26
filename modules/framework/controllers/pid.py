"""PIDKalmanController — u = −Kp·e − Ki·∫e dt − Kd·ė."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class PIDKalmanController(KalmanController):
    """PID controller on the state error ``e = x̂ − x_ref``.

    Set any gain to ``0`` (or ``None``) for P-only, PD, PI, etc.
    """

    def __init__(self, model, Kp, Ki=None, Kd=None, **kw):
        super().__init__(model, **kw)
        self.Kp = np.asarray(Kp, dtype=float)
        self.Ki = np.zeros_like(self.Kp) if Ki is None else np.asarray(Ki, dtype=float)
        self.Kd = np.zeros_like(self.Kp) if Kd is None else np.asarray(Kd, dtype=float)
        self._integral = np.zeros((model.n, 1))
        self._e_prev = np.zeros((model.n, 1))
        self._dt = 1.0 / model.fs if model.fs else 1.0

    def _policy(self, x_hat, P, t, *, nu=None):
        e = x_hat - self.x_ref
        self._integral += e * self._dt
        de = (e - self._e_prev) / self._dt
        self._e_prev = e.copy()
        return -(self.Kp @ e + self.Ki @ self._integral + self.Kd @ de)

    def reset(self):
        super().reset()
        n = self.model.n
        self._integral = np.zeros((n, 1))
        self._e_prev = np.zeros((n, 1))

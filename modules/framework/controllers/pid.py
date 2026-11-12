"""PIDKalmanController — u = −Kp·e − Ki·∫e dt − Kd·ė."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class PIDKalmanController(KalmanController):
    """PID controller on the state error ``e = x̂ − x_ref``.

    Set any gain to ``0`` (or ``None``) for P-only, PD, PI, etc.

    Computed by ``Kalman::Controllers::PID`` when libKalman is installed, which then
    holds the integral and the previous error (``reset()`` clears both sides);
    :meth:`_reference_policy` is the Python reference. Changing a gain or the set
    point rebuilds the C++ controller, whose integral starts again from zero.
    """

    NATIVE = "pid"

    def __init__(self, model, Kp, Ki=None, Kd=None, **kw):
        super().__init__(model, **kw)
        self.Kp = np.asarray(Kp, dtype=float)
        self.Ki = np.zeros_like(self.Kp) if Ki is None else np.asarray(Ki, dtype=float)
        self.Kd = np.zeros_like(self.Kp) if Kd is None else np.asarray(Kd, dtype=float)
        self._integral = np.zeros((model.n, 1))
        self._e_prev = np.zeros((model.n, 1))
        self._dt = 1.0 / model.fs if model.fs else 1.0

    def _policy(self, x_hat, P, t, *, nu=None):
        u = self._native_policy(x_hat, P, t, nu)
        return u if u is not None else self._reference_policy(x_hat, P, t, nu=nu)

    def _reference_policy(self, x_hat, P, t, *, nu=None):
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

    def _native_key(self):
        # The C++ PID takes its step from the model's sample rate, as _dt does.
        if self._dt != (1.0 / self.model.fs if self.model.fs else 1.0):
            return None
        gains, costs = self._gains_key(self.Kp, self.Ki, self.Kd), self._cost_key()
        return None if gains is None or costs is None else gains + costs

    def _native_create(self, model_handle):
        from ..native import core
        return core.create_controller("pid", model_handle, int(self.model.n), max(int(self.model.l), 1),
                                      Kp=self.Kp, Ki=self.Ki, Kd=self.Kd, **self._cost_args())

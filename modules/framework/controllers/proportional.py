"""ProportionalKalmanController — u = −Kp · (x̂ − x_ref)."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class ProportionalKalmanController(KalmanController):
    """Proportional controller: ``u = −Kp · (x̂ − x_ref)``.

    Computed by ``Kalman::Controllers::Proportional`` when libKalman is installed;
    :meth:`_reference_policy` is the Python reference.
    """

    NATIVE = "proportional"

    def __init__(self, model, Kp, **kw):
        super().__init__(model, **kw)
        self.Kp = np.asarray(Kp, dtype=float)

    def _policy(self, x_hat, P, t, *, nu=None):
        u = self._native_policy(x_hat, P, t, nu)
        return u if u is not None else self._reference_policy(x_hat, P, t, nu=nu)

    def _reference_policy(self, x_hat, P, t, *, nu=None):
        e = x_hat - self.x_ref
        return -self.Kp @ e

    def _native_key(self):
        gains, costs = self._gains_key(self.Kp), self._cost_key()
        return None if gains is None or costs is None else gains + costs

    def _native_create(self, model_handle):
        from ..native import core
        return core.create_controller("proportional", model_handle, int(self.model.n), max(int(self.model.l), 1),
                                      Kp=self.Kp, **self._cost_args())

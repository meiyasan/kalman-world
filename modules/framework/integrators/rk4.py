"""Classic 4-stage Runge–Kutta integrator (4th order).

The single home for the RK4 math that ``runtime.astro.rk4`` and the old
``vehicles._rk4`` / ``tracking._rk4`` each duplicated.  The control ``u`` is held
constant across the step (zero-order hold), which is the right assumption inside
one filter cycle.
"""
from __future__ import annotations
import numpy as np

from ..integrator import KalmanIntegrator
from ..constants import KalmanIntegratorKind


class RK4Integrator(KalmanIntegrator):
    """``x(t+dt)`` from the classic RK4 tableau of ``ẋ = f(x, u, t)``."""

    NAME = "RK4 Integrator"
    DESCRIPTION = "Classic 4-stage Runge–Kutta (fourth order); ZOH on the control."
    VERSION = "1.0"
    KIND = KalmanIntegratorKind.RK4
    tableau = ([0.0, 0.5, 0.5, 1.0],
               [[], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]],
               [1/6, 1/3, 1/3, 1/6])

    NATIVE = "rk4"

    def _step(self, f, x, u=None, t=None, dt=1.0):
        x = np.asarray(x)                      # dtype-preserving (real or spectral/complex)
        t = 0.0 if t is None else float(t)
        k1 = np.asarray(f(x, u, t))
        k2 = np.asarray(f(x + 0.5 * dt * k1, u, t + 0.5 * dt))
        k3 = np.asarray(f(x + 0.5 * dt * k2, u, t + 0.5 * dt))
        k4 = np.asarray(f(x + dt * k3, u, t + dt))
        return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


from ..integrator import _BUILTIN_STEPS
_BUILTIN_STEPS["rk4"] = RK4Integrator._step

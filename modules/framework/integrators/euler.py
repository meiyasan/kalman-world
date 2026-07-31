"""Explicit forward-Euler integrator (1st order)."""
from __future__ import annotations
import numpy as np

from ..integrator import KalmanIntegrator
from ..constants import KalmanIntegratorKind


class EulerIntegrator(KalmanIntegrator):
    """``x(t+dt) = x(t) + dt · f(x, u, t)`` — cheapest, first-order."""

    NAME = "Euler Integrator"
    DESCRIPTION = "Explicit forward Euler: x + dt·f(x,u,t) (first order)."
    VERSION = "1.0"
    KIND = KalmanIntegratorKind.EULER
    tableau = ([0.0], [[]], [1.0])          # 1-stage explicit

    def step(self, f, x, u=None, t=None, dt=1.0):
        x = np.asarray(x)                      # dtype-preserving (real or spectral/complex)
        return x + dt * np.asarray(f(x, u, t))

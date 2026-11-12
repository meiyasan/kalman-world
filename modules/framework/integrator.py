# kalman/framework/integrator.py
"""
Abstract **integrator** base — the time-stepping scheme (WHEN-in-dt).

An integrator advances a state one step ``x(t) → x(t+dt)`` given a *continuous*
derivative ``ẋ = f(x, u, t)``.  It is the counterpart of the estimation
:class:`~kalman.framework.engine.KalmanEngine` for the *deterministic mean*
propagation, and the piece a :class:`~kalman.framework.solver.KalmanSolver`
composes for its time march (for a PDE) — for an ODE the model binds an
integrator directly.

Concrete integrators (``integrators/``) tag their
:class:`~kalman.framework.constants.KalmanIntegratorKind` and override ``step``;
``integrate`` (n steps) has a default loop here.  Integrators own no run state.
"""
from __future__ import annotations

import numpy as np

from .abstract.metadata import KalmanIntegratorMetadata


class KalmanIntegrator(KalmanIntegratorMetadata):

    NAME = "Kalman Integrator"
    DESCRIPTION = "Abstract time integrator: advance x(t) -> x(t+dt) for xdot = f(x,u,t)."
    VERSION = "1.0"
    KIND = None

    # Explicit Runge–Kutta Butcher tableau ``(c, A, b)`` for methods that have
    # one, so field solvers can run *any* explicit integrator generically (in
    # numpy or torch) instead of hard-coding a scheme.  ``None`` for methods
    # without a fixed tableau (adaptive/implicit).
    tableau = None

    # The name of the C++ integrator that implements this class
    # (Kalman::Integrators), or None for a scheme written in Python.
    NATIVE = None

    # ------------------------------------------------------------------
    def step(self, f, x, u=None, t=None, dt=1.0):
        """Advance one step of size ``dt``; ``f(x, u, t) → ẋ``.

        The built-in schemes run in C++ (``NATIVE``). A scheme written in
        Python overrides this method."""
        out = self._native(f, x, u, t, dt, 1)
        if out is not None:
            return out
        return self._step(f, x, u=u, t=t, dt=dt)

    def _step(self, f, x, u=None, t=None, dt=1.0):
        """The scheme in Python: the reference the C++ one is tested against,
        used for complex states and when libKalman is not there."""
        raise NotImplementedError

    def _native_options(self) -> dict:
        return {}

    def _native(self, f, x, u, t, dt, n):
        """The C++ integrator's result, or ``None`` when this call is not one it takes:
        a Python-written scheme, a state that is not real double precision, the
        reference path, or no library."""
        if self.NATIVE is None or type(self)._step is not _BUILTIN_STEPS.get(self.NATIVE):
            return None
        from .models import native
        if not native.native_loop_enabled():
            return None
        state = np.asarray(x)
        if state.dtype != np.float64 or state.size == 0:
            return None
        from .native import core
        return core.integrate(self.NATIVE, f, state, u=u, t=0.0 if t is None else float(t),
                              dt=float(dt), n=int(n), **self._native_options())

    def integrate(self, f, x, u=None, t=None, dt=1.0, n=1):
        """Advance ``n`` steps of ``dt`` (holding ``u`` constant across each)."""
        if type(self).step is KalmanIntegrator.step:
            out = self._native(f, x, u, t, dt, n)
            if out is not None:
                return out
        t = 0.0 if t is None else float(t)
        for _ in range(int(n)):
            x = self.step(f, x, u=u, t=t, dt=dt)
            t += dt
        return x


#: ``NATIVE`` name -> the ``_step`` of the class that the C++ scheme mirrors. A
#: subclass that changes ``_step`` is a different scheme and runs in Python.
_BUILTIN_STEPS: dict = {}

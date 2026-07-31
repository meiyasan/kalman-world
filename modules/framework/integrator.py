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

    # ------------------------------------------------------------------
    def step(self, f, x, u=None, t=None, dt=1.0):
        """Advance one step of size ``dt``; ``f(x, u, t) → ẋ``.  Override."""
        raise NotImplementedError

    def integrate(self, f, x, u=None, t=None, dt=1.0, n=1):
        """Advance ``n`` steps of ``dt`` (holding ``u`` constant across each)."""
        t = 0.0 if t is None else float(t)
        for _ in range(int(n)):
            x = self.step(f, x, u=u, t=t, dt=dt)
            t += dt
        return x

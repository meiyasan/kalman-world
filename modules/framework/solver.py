# kalman/framework/solver.py
"""
Abstract **solver** base — the spatial-discretization strategy.

A solver turns a continuous field problem into something an integrator can march
in time.  Contract:

    - ``discretize(field)`` — the solver's own job: form the semi-discrete RHS
      (spatial part), e.g. the pseudo-spectral RHS for a PDE;
    - ``integrate(...)`` — the *time* march, **delegated to a composed**
      :class:`~kalman.framework.integrator.KalmanIntegrator` (one home for the
      stepping math);
    - ``solve(field, steps, u, t)`` — the public entry: discretize then
      integrate over the interval.

For an ODE ``discretize`` is the identity, so ``solve`` ≡ ``integrate`` — which
is why "solver ≈ integrator" for ODEs.  **Composition, not inheritance**: a
solver *has-a* integrator (different solvers pair with different integrators).

A tensor **backend** (``"numpy"`` default, ``"torch"`` optional) with a
``device`` (``"cpu"/"cuda"/"mps"``) lets field solvers run on GPU/MPS without any
per-model change; concrete solvers honour it where it pays (see
``solvers/spectral.py``).
"""
from __future__ import annotations
import os

from .abstract.metadata import KalmanSolverMetadata
from .constants import KalmanIntegratorKind


class KalmanSolver(KalmanSolverMetadata):

    NAME = "Kalman Solver"
    DESCRIPTION = "Abstract spatial-discretization solver: discretize + integrate (composes an integrator)."
    VERSION = "1.0"
    KIND = None
    DEFAULT_INTEGRATOR = KalmanIntegratorKind.RK4

    def __init__(self, *, integrator=None, backend=None, device=None):
        self.backend = (backend or os.environ.get("KALMAN_BACKEND", "numpy")).lower()
        self.device = (device or os.environ.get("KALMAN_DEVICE", "cpu")).lower()
        self._integrator = integrator          # resolved lazily (avoids import cycle)

    # ------------------------------------------------------------------
    @property
    def integrator(self):
        if self._integrator is None:
            from .integrators import resolve_integrator
            self._integrator = resolve_integrator(self.DEFAULT_INTEGRATOR)
        elif isinstance(self._integrator, KalmanIntegratorKind):
            from .integrators import resolve_integrator
            self._integrator = resolve_integrator(self._integrator)
        return self._integrator

    # ------------------------------------------------------------------
    def discretize(self, field):
        """Spatial discretization → a semi-discrete representation.  Identity by
        default (ODE case); PDE solvers override (e.g. FFT to spectral space)."""
        return field

    def integrate(self, rhs, state, u=None, t=None, dt=1.0, n=1):
        """Time march via the composed integrator (``rhs(state,u,t) → d/dt``)."""
        return self.integrator.integrate(rhs, state, u=u, t=t, dt=dt, n=int(n))

    def solve(self, field, steps=1, u=None, t=None):
        """Public entry: ``discretize`` → ``integrate`` → return the evolved field.
        Concrete field solvers override for their representation."""
        raise NotImplementedError

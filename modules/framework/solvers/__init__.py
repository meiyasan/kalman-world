"""
Concrete spatial-discretization solvers (the *how in space*).

Each solver tags its :class:`~kalman.framework.constants.KalmanSolverKind` and
composes a :class:`~kalman.framework.integrator.KalmanIntegrator` for the time
march.  ``SOLVER_BY_KIND`` lets a model resolve its declared ``DEFAULT_SOLVER``
without importing a solver.  FD/FV/FEM are reserved (raise on use).
"""
from ..solver import KalmanSolver
from ..constants import KalmanSolverKind

from .spectral import SpectralSolver
from .finite_difference import FiniteDifferenceSolver


def _reserved(kind, need):
    class _Reserved(KalmanSolver):
        NAME = f"{kind.name.title()} Solver"
        DESCRIPTION = f"{kind.name} discretization — reserved ({need})."
        KIND = kind
        def solve(self, field, steps=1, u=None, t=None):
            raise NotImplementedError(f"{kind.name} solver is reserved ({need}).")
    return _Reserved


_FiniteVolumeReserved = _reserved(KalmanSolverKind.FINITE_VOLUME, "flux discretization")
_FiniteElementReserved = _reserved(KalmanSolverKind.FINITE_ELEMENT, "mesh/basis discretization")


SOLVER_BY_KIND = {
    KalmanSolverKind.SPECTRAL:          SpectralSolver,
    KalmanSolverKind.FINITE_DIFFERENCE: FiniteDifferenceSolver,
    KalmanSolverKind.FINITE_VOLUME:     _FiniteVolumeReserved,
    KalmanSolverKind.FINITE_ELEMENT:    _FiniteElementReserved,
}


def resolve_solver(kind: KalmanSolverKind, **kwargs) -> KalmanSolver:
    """Instantiate the solver for ``kind`` (defaults if unknown -> Spectral).

    Robust to identity changes from ``%autoreload`` via a ``name`` fallback (same
    discipline as ``resolve_engine``).  ``kwargs`` forward to the solver ctor."""
    cls = SOLVER_BY_KIND.get(kind)
    if cls is None and kind is not None:
        name = getattr(kind, "name", None)
        cls = next((c for k, c in SOLVER_BY_KIND.items() if getattr(k, "name", None) == name), None)
    return (cls or SpectralSolver)(**kwargs)


__all__ = [
    "KalmanSolver",
    "SpectralSolver",
    "FiniteDifferenceSolver",
    "SOLVER_BY_KIND",
    "resolve_solver",
]

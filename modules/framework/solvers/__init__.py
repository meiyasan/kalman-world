"""
Concrete spatial-discretization solvers (the *how in space*).

Each shipped solver tags its
:class:`~kalman.framework.constants.KalmanSolverKind` and composes a
:class:`~kalman.framework.integrator.KalmanIntegrator` for the time march;
``SOLVER_BY_KIND`` lets a model resolve its declared ``DEFAULT_SOLVER`` without
importing a solver.  FD/FV/FEM entries are reserved (raise on use).

As with the integrators, the enum is the authority for **shipped** solvers only:
:func:`register_solver` binds a custom discretization to a name that
:func:`resolve_solver` accepts, and an unknown kind raises rather than silently
degrading to the spectral default.
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

# Custom discretizations registered at runtime: lowercase name -> class.  Kept
# apart from SOLVER_BY_KIND so the "registry covers its whole enum" invariant holds.
SOLVER_BY_NAME: dict[str, type] = {}

DEFAULT_SOLVER = SpectralSolver


def _builtin_names() -> set:
    return {k.name.lower() for k in SOLVER_BY_KIND}


def available_solvers() -> list:
    """Every resolvable solver name — shipped kinds plus registered ones."""
    return sorted(_builtin_names() | set(SOLVER_BY_NAME))


def register_solver(name, cls):
    """Register a custom :class:`KalmanSolver` subclass under ``name``.

    ``name`` may be a string (a new kind, resolvable everywhere a kind name is
    accepted — including a ``KalmanSpec``'s ``bindings.solver``) or a
    :class:`KalmanSolverKind` member, which **rebinds** that shipped kind.
    Returns ``cls``, so it also works as a decorator.
    """
    if not (isinstance(cls, type) and issubclass(cls, KalmanSolver)):
        raise TypeError(f"{cls!r} is not a KalmanSolver subclass.")
    if isinstance(name, KalmanSolverKind):
        SOLVER_BY_KIND[name] = cls
        return cls
    key = str(name).strip().lower()
    if not key:
        raise ValueError("a solver name must be a non-empty string.")
    if key in _builtin_names():
        raise ValueError(f"{key!r} is a shipped kind; pass KalmanSolverKind.{key.upper()} "
                         f"to rebind it deliberately.")
    SOLVER_BY_NAME[key] = cls
    return cls


def unregister_solver(name) -> bool:
    """Drop a custom registration (no-op for shipped kinds).  Returns whether it existed."""
    return SOLVER_BY_NAME.pop(str(name).strip().lower(), None) is not None


def resolve_solver(kind=None, **kwargs) -> KalmanSolver:
    """Instantiate the solver for ``kind``; ``kwargs`` forward to its constructor.

    Accepts a :class:`KalmanSolverKind`, a registered name, a ``KalmanSolver``
    subclass, or a ready instance (returned as-is, ``kwargs`` ignored).  ``None``
    gives the default (spectral).  An unrecognised kind raises
    :class:`ValueError` instead of silently falling back.  Enum lookup keeps the
    ``name`` fallback that makes it robust to ``%autoreload``.
    """
    if isinstance(kind, KalmanSolver):
        return kind
    if isinstance(kind, type) and issubclass(kind, KalmanSolver):
        return kind(**kwargs)
    if kind is None:
        return DEFAULT_SOLVER(**kwargs)

    if isinstance(kind, str):
        key = kind.strip().lower()
        cls = SOLVER_BY_NAME.get(key) or next(
            (c for k, c in SOLVER_BY_KIND.items() if k.name.lower() == key), None)
    else:
        cls = SOLVER_BY_KIND.get(kind)
        if cls is None:                              # %autoreload identity fallback
            name = getattr(kind, "name", None)
            cls = next((c for k, c in SOLVER_BY_KIND.items()
                        if getattr(k, "name", None) == name), None)

    if cls is None:
        raise ValueError(f"unknown solver kind {kind!r}; "
                         f"choose from {available_solvers()} or register_solver(...).")
    return cls(**kwargs)


__all__ = [
    "KalmanSolver",
    "SpectralSolver",
    "FiniteDifferenceSolver",
    "SOLVER_BY_KIND",
    "SOLVER_BY_NAME",
    "available_solvers",
    "register_solver",
    "unregister_solver",
    "resolve_solver",
]

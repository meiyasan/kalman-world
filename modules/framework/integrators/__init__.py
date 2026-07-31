"""
Concrete time integrators (the *time-stepping* — advance x(t) -> x(t+dt)).

Each shipped integrator tags its
:class:`~kalman.framework.constants.KalmanIntegratorKind`; ``INTEGRATOR_BY_KIND``
lets a model resolve its declared ``DEFAULT_INTEGRATOR`` to a concrete class
without importing an integrator.

The enum is the authority for **shipped** schemes only.  A custom stepper does
not need a member: :func:`register_integrator` binds it to a name, and
:func:`resolve_integrator` accepts that name, a class, or a ready instance — so
a downstream package can add a scheme without editing ``constants.py``.  An
unknown kind raises rather than silently degrading to the default.
"""
from ..integrator import KalmanIntegrator
from ..constants import KalmanIntegratorKind

from .euler import EulerIntegrator
from .rk4 import RK4Integrator
from .dormand_prince import DormandPrinceIntegrator


class _ImplicitEulerReserved(KalmanIntegrator):
    """Reserved: backward Euler needs a per-step nonlinear (Newton) solve — deferred."""
    NAME = "Implicit Euler Integrator"
    DESCRIPTION = "Backward Euler (stiff) — reserved; needs a nonlinear solve."
    KIND = KalmanIntegratorKind.IMPLICIT_EULER

    def step(self, f, x, u=None, t=None, dt=1.0):
        raise NotImplementedError(
            "IMPLICIT_EULER is reserved (needs a Newton solve); use RK4/DORMAND_PRINCE.")


INTEGRATOR_BY_KIND = {
    KalmanIntegratorKind.EULER:          EulerIntegrator,
    KalmanIntegratorKind.RK4:            RK4Integrator,
    KalmanIntegratorKind.DORMAND_PRINCE: DormandPrinceIntegrator,
    KalmanIntegratorKind.IMPLICIT_EULER: _ImplicitEulerReserved,
}

# Custom schemes registered at runtime: lowercase name -> class.  Kept apart from
# INTEGRATOR_BY_KIND so the "registry covers its whole enum" invariant still holds.
INTEGRATOR_BY_NAME: dict[str, type] = {}

DEFAULT_INTEGRATOR = RK4Integrator


def _builtin_names() -> set:
    return {k.name.lower() for k in INTEGRATOR_BY_KIND}


def available_integrators() -> list:
    """Every resolvable integrator name — shipped kinds plus registered ones."""
    return sorted(_builtin_names() | set(INTEGRATOR_BY_NAME))


def register_integrator(name, cls):
    """Register a custom :class:`KalmanIntegrator` subclass under ``name``.

    ``name`` may be a string (a new kind, resolvable everywhere a kind name is
    accepted — including a ``KalmanSpec``'s ``bindings.integrator``) or a
    :class:`KalmanIntegratorKind` member, which **rebinds** that shipped kind to
    your class.  Returns ``cls``, so it also works as a decorator.
    """
    if not (isinstance(cls, type) and issubclass(cls, KalmanIntegrator)):
        raise TypeError(f"{cls!r} is not a KalmanIntegrator subclass.")
    if isinstance(name, KalmanIntegratorKind):
        INTEGRATOR_BY_KIND[name] = cls
        return cls
    key = str(name).strip().lower()
    if not key:
        raise ValueError("an integrator name must be a non-empty string.")
    if key in _builtin_names():
        raise ValueError(f"{key!r} is a shipped kind; pass KalmanIntegratorKind.{key.upper()} "
                         f"to rebind it deliberately.")
    INTEGRATOR_BY_NAME[key] = cls
    return cls


def unregister_integrator(name) -> bool:
    """Drop a custom registration (no-op for shipped kinds).  Returns whether it existed."""
    return INTEGRATOR_BY_NAME.pop(str(name).strip().lower(), None) is not None


def resolve_integrator(kind=None, **kwargs) -> KalmanIntegrator:
    """Instantiate the integrator for ``kind``.

    Accepts a :class:`KalmanIntegratorKind`, a registered name, a
    ``KalmanIntegrator`` subclass, or a ready instance (returned as-is).
    ``None`` gives the default (RK4).  An unrecognised kind raises
    :class:`ValueError` — it is never silently replaced by the default, because a
    typo that quietly runs a different scheme is worse than a failure.

    Enum lookup falls back to matching by ``name`` so the registry survives
    ``KalmanIntegratorKind`` identity changes from ``%autoreload`` or duplicate
    imports (same discipline as ``resolve_engine``).
    """
    if isinstance(kind, KalmanIntegrator):
        return kind
    if isinstance(kind, type) and issubclass(kind, KalmanIntegrator):
        return kind(**kwargs)
    if kind is None:
        return DEFAULT_INTEGRATOR(**kwargs)

    if isinstance(kind, str):
        key = kind.strip().lower()
        cls = INTEGRATOR_BY_NAME.get(key) or next(
            (c for k, c in INTEGRATOR_BY_KIND.items() if k.name.lower() == key), None)
    else:
        cls = INTEGRATOR_BY_KIND.get(kind)
        if cls is None:                              # %autoreload identity fallback
            name = getattr(kind, "name", None)
            cls = next((c for k, c in INTEGRATOR_BY_KIND.items()
                        if getattr(k, "name", None) == name), None)

    if cls is None:
        raise ValueError(f"unknown integrator kind {kind!r}; "
                         f"choose from {available_integrators()} or register_integrator(...).")
    return cls(**kwargs)


__all__ = [
    "KalmanIntegrator",
    "EulerIntegrator",
    "RK4Integrator",
    "DormandPrinceIntegrator",
    "INTEGRATOR_BY_KIND",
    "INTEGRATOR_BY_NAME",
    "available_integrators",
    "register_integrator",
    "unregister_integrator",
    "resolve_integrator",
]

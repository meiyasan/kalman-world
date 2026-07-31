"""
Concrete time integrators (the *time-stepping* — advance x(t) -> x(t+dt)).

Each integrator tags its :class:`~kalman.framework.constants.KalmanIntegratorKind`.
``INTEGRATOR_BY_KIND`` lets a model resolve its declared ``DEFAULT_INTEGRATOR``
to a concrete class without importing an integrator.
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


def resolve_integrator(kind: KalmanIntegratorKind) -> KalmanIntegrator:
    """Instantiate the integrator for ``kind`` (defaults if unknown -> RK4).

    Robust to ``KalmanIntegratorKind`` identity changes from ``%autoreload`` /
    duplicate imports: fall back to matching by ``name`` (same discipline as
    ``resolve_engine``)."""
    cls = INTEGRATOR_BY_KIND.get(kind)
    if cls is None and kind is not None:
        name = getattr(kind, "name", None)
        cls = next((c for k, c in INTEGRATOR_BY_KIND.items() if getattr(k, "name", None) == name), None)
    return (cls or RK4Integrator)()


__all__ = [
    "KalmanIntegrator",
    "EulerIntegrator",
    "RK4Integrator",
    "DormandPrinceIntegrator",
    "INTEGRATOR_BY_KIND",
    "resolve_integrator",
]

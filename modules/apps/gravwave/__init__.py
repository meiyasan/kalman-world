"""
``kalman.apps.gravwave`` — gravitational-wave polarization domain.

- ``polarization`` — :class:`PolarizationModel` (Jones-formalism EKF on the
  Poincare sphere) and the :func:`jones_signal` generator.
"""
from .polarization import (PolarizationModel,
                   PolarizationState,
                   PolarizationObservable,
                   jones_signal)

VARIANTS = {
    "extended": PolarizationModel,
}

__all__ = [
    "PolarizationModel",
    "PolarizationState",
    "PolarizationObservable",
    "jones_signal",
    "VARIANTS",
]

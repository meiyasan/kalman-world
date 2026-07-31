"""
``kalman.apps.suspensions`` — mechanical suspension domain.

- ``model`` — :class:`SuspensionModel` (multi-stage pendulum, exact ``expm``
  discretisation of the continuous generator).
"""
from .model import (SuspensionModel,
                   SuspensionState,
                   SuspensionObservable)

VARIANTS = {
    "linear": SuspensionModel,
}

__all__ = [
    "SuspensionModel",
    "SuspensionState",
    "SuspensionObservable",
    "VARIANTS",
]

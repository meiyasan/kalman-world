"""
``kalman.apps.fusion`` — heterogeneous sensor-fusion domain.

- ``model`` — :class:`SensorFusionModel` and the :class:`SensorSpec` records
  describing each fused sensor (rate, noise, observed states).
"""
from .model import (SensorFusionModel,
                   SensorSpec,
                   FusionState,
                   FusionObservable)

VARIANTS = {
    "linear": SensorFusionModel,
}

__all__ = [
    "SensorFusionModel",
    "SensorSpec",
    "FusionState",
    "FusionObservable",
    "VARIANTS",
]

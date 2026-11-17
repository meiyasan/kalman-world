"""
``kalman.apps.fusion`` — heterogeneous sensor-fusion domain.

- ``model`` — :class:`SensorFusionModel` and the :class:`SensorSpec` records
  describing each fused sensor (rate, noise, observed states).
- ``sensors`` — ready-made sensors for any model: :class:`PositionSensor`,
  :class:`DopplerRadar`, :class:`Camera` (``Kalman::Apps::Fusion`` in C++).
"""
from .model import (SensorFusionModel,
                   SensorSpec,
                   FusionState,
                   FusionObservable)
from .sensors import PositionSensor, DopplerRadar, Camera

VARIANTS = {
    "linear": SensorFusionModel,
}

__all__ = [
    "SensorFusionModel",
    "SensorSpec",
    "FusionState",
    "FusionObservable",
    "PositionSensor",
    "DopplerRadar",
    "Camera",
    "VARIANTS",
]

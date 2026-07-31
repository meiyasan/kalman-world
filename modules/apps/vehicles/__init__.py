"""
kalman.apps.vehicles — road-vehicle domain.

- speed — :class:`CarSpeedModel` (longitudinal force balance, RK4);
- lane  — :class:`CarLaneModel` (lateral bicycle kinematics, Euler).
"""
from .speed import (CarSpeedModel, CarSpeedState, CarSpeedObservable)
from .lane import (CarLaneModel, CarLaneState, CarLaneObservable)

VARIANTS = {
    "speed": CarSpeedModel,
    "lane": CarLaneModel,
}

__all__ = [
    "CarSpeedModel",
    "CarSpeedState",
    "CarSpeedObservable",
    "CarLaneModel",
    "CarLaneState",
    "CarLaneObservable",
    "VARIANTS",
]

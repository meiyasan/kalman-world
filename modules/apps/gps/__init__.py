"""
``kalman.apps.gps`` — GNSS positioning domain.

- ``model`` — :class:`GPSModel` (pseudorange EKF, relativistic corrections)
  and the :class:`GPSConstellation` almanac it observes.
"""
from .model import (GPSModel,
                   GPSConstellation,
                   GPSState,
                   GPSObservable)

VARIANTS = {
    "extended": GPSModel,
}

__all__ = [
    "GPSModel",
    "GPSConstellation",
    "GPSState",
    "GPSObservable",
    "VARIANTS",
]

"""
``kalman.apps.tracking`` — particle-tracking (drift chamber) domain.

- ``driftchamber`` — :class:`DriftChamberModel` (arc-length propagation in a
  magnetic field) and the :class:`TrackingEvent` hit container.
"""
from .driftchamber import (DriftChamberModel,
                   TrackingEvent,
                   TrackState,
                   TrackObservable)

VARIANTS = {
    "extended": DriftChamberModel,
}

__all__ = [
    "DriftChamberModel",
    "TrackingEvent",
    "TrackState",
    "TrackObservable",
    "VARIANTS",
]

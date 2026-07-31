"""
kalman.apps.rocket — launch-vehicle domain.

Flight phases as variants: ascent, liftoff, landing, descent.
"""
from .ascent import (RocketAscentModel, RocketAscentState, RocketAscentObservable)
from .liftoff import (RocketLiftoffModel, RocketLiftoffState, RocketLiftoffObservable)
from .landing import (RocketLandingModel, RocketLandingState, RocketLandingObservable)
from .descent import (RocketDescentModel, RocketDescentState, RocketDescentObservable)

# Legacy aliases kept from the flat module.
RocketLaunchModel = RocketAscentModel
RocketTakeoffModel = RocketLiftoffModel

VARIANTS = {
    "ascent": RocketAscentModel,
    "liftoff": RocketLiftoffModel,
    "landing": RocketLandingModel,
    "descent": RocketDescentModel,
}

__all__ = [
    "RocketAscentModel",
    "RocketAscentState",
    "RocketAscentObservable",
    "RocketLiftoffModel",
    "RocketLiftoffState",
    "RocketLiftoffObservable",
    "RocketLandingModel",
    "RocketLandingState",
    "RocketLandingObservable",
    "RocketDescentModel",
    "RocketDescentState",
    "RocketDescentObservable",
    "RocketLaunchModel",
    "RocketTakeoffModel",
    "VARIANTS",
]

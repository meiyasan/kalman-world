"""
``kalman.apps.ballistics`` — the ballistics domain package.

One domain, its model **variants** as sibling submodules (the same axis the
framework uses for ``models/{linear,extended,unscented}.py``):

- ``base``     — shared state/observable enums, noise presets, estimate records;
- ``linear``   — :class:`BallisticLinearModel` (constant-acceleration KF, exact
  ``expm`` discretization, no drag);
- ``extended`` — :class:`BallisticModel` (nonlinear gravity + drag + thrust EKF).

``VARIANTS`` maps the variant key to its class so a catalog/agent can offer
"ballistics: linear | extended" without importing each submodule.
"""
from .base import (BallisticState, BallisticObservable, BallisticNoise,
                   ApexEstimate, ImpactEstimate)
from .linear import BallisticLinearModel
from .extended import BallisticModel

VARIANTS = {
    "linear": BallisticLinearModel,
    "extended": BallisticModel,
}

__all__ = [
    "BallisticState", "BallisticObservable", "BallisticNoise",
    "ApexEstimate", "ImpactEstimate",
    "BallisticLinearModel", "BallisticModel",
    "VARIANTS",
]

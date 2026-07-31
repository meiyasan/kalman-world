"""
``kalman.apps.astro`` — flight and orbital dynamics.

Domains sharing the ``runtime.astro`` foundation (gravity models,
ephemerides, propagation):

- ``rocket`` — launch-vehicle phases (ascent, liftoff, landing, descent);
- ``space``  — cislunar/interplanetary cruise and the transfer toolkit.
"""
from .rocket import *          # noqa: F401,F403
from .space import *          # noqa: F401,F403

from . import rocket, space

DOMAINS = {
    "rocket": rocket,
    "space": space,
}

__all__ = [d for d in DOMAINS] + [
    n for _m in DOMAINS.values() for n in getattr(_m, '__all__', [])
] + ['DOMAINS']

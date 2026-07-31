"""
``kalman.apps.gravitational`` — gravitational-wave physics.

The source and the instrument that observes it:

- ``gravwave``       — GW polarization (Jones formalism, Poincare sphere);
- ``interferometer`` — detector violin-mode tracking (K1/H1/L1/V1).

``interferometer`` builds on ``apps.resonators.oscillators`` — an
implementation dependency across families, which is expected.
"""
from .gravwave import *          # noqa: F401,F403
from .interferometer import *          # noqa: F401,F403

from . import gravwave, interferometer

DOMAINS = {
    "gravwave": gravwave,
    "interferometer": interferometer,
}

__all__ = [d for d in DOMAINS] + [
    n for _m in DOMAINS.values() for n in getattr(_m, '__all__', [])
] + ['DOMAINS']

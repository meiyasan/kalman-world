"""
``kalman.apps.resonators`` — harmonic-oscillator physics and its applications.

- ``oscillators`` — the shared oscillator library (analytic, phasor,
  windowed composite, bank, harmonic/inharmonic variants);
- ``powergrids``  — mains-harmonic tracking (a PhasorOscillatorModel
  specialisation);
- ``suspensions`` — multi-stage pendulum chains.
"""
from .oscillators import *          # noqa: F401,F403
from .powergrids import *          # noqa: F401,F403
from .suspensions import *          # noqa: F401,F403

from . import oscillators, powergrids, suspensions

DOMAINS = {
    "oscillators": oscillators,
    "powergrids": powergrids,
    "suspensions": suspensions,
}

__all__ = [d for d in DOMAINS] + [
    n for _m in DOMAINS.values() for n in getattr(_m, '__all__', [])
] + ['DOMAINS']

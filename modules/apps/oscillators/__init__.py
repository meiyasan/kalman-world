"""
``kalman.apps.oscillators`` — the oscillator domain package.

One domain, its model **variants** as sibling submodules:

- ``base``      — shared state/observable enums + wire/partial-frequency helpers;
- ``analytic``  — :class:`AnalyticOscillatorModel` (single decoupled oscillator on
  the Hilbert-demodulated analytic signal — standalone illustration);
- ``phasor``    — :class:`PhasorOscillatorModel` (lab-frame **joint** phasor
  estimator; all partials coupled through one scalar measurement);
- ``composite`` — :class:`OscillatorModel` (windowed rotating-frame composite —
  the KAGRA violin-mode workhorse);
- ``bank``      — :class:`OscillatorBankModel` (independent partial bank);
- ``harmonic``  — :class:`InharmonicOscillatorModel` / :class:`HarmonicOscillatorModel`.

``VARIANTS`` maps each variant key to its class for catalog/agent discovery.
"""
from .base import (OscillatorState, OscillatorObservable, wire_inharmonicity,
                   wire_fundamental, partial_frequencies)
from .analytic import AnalyticOscillatorModel
from .phasor import PhasorState, PhasorObservable, PhasorOscillatorModel
from .composite import OscillatorModel
from .bank import OscillatorBankModel
from .harmonic import InharmonicOscillatorModel, HarmonicOscillatorModel

# Legacy alias kept from the flat module.
OscillatorCompositeModel = OscillatorBankModel

VARIANTS = {
    "analytic": AnalyticOscillatorModel,
    "phasor": PhasorOscillatorModel,
    "composite": OscillatorModel,
    "bank": OscillatorBankModel,
    "inharmonic": InharmonicOscillatorModel,
    "harmonic": HarmonicOscillatorModel,
}

__all__ = [
    "OscillatorState", "OscillatorObservable",
    "wire_inharmonicity", "wire_fundamental", "partial_frequencies",
    "AnalyticOscillatorModel", "PhasorState", "PhasorObservable",
    "PhasorOscillatorModel", "OscillatorModel", "OscillatorBankModel",
    "OscillatorCompositeModel", "InharmonicOscillatorModel", "HarmonicOscillatorModel",
    "VARIANTS",
]

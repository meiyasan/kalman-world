"""Inharmonic / harmonic partial series — variants layered on OscillatorModel."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from ....framework.models.linear import LinearKalmanModel
from ....framework.models.composite import CompositeKalmanModel
from ....framework.constants import KalmanState, KalmanObservable, KalmanObservableReadout
from .base import (OscillatorState, OscillatorObservable, wire_inharmonicity,
                   wire_fundamental, partial_frequencies)

from .composite import OscillatorModel
# ======================================================================
# (In)harmonic ladders — an OscillatorModel window whose candidate list is a ladder
# ======================================================================
class InharmonicOscillatorModel(OscillatorModel):
    """
    One :class:`OscillatorModel` window whose candidate frequency list is an **inharmonic
    ladder**: ``f_n = n·f0·√(1+B·n²)`` for ``n = 1 … n_harmonics``, generated from
    ``(f0, B)``.

    As any :class:`OscillatorModel` window it **masks** to ``[fc ± dfc]``: only partials
    inside the window are tracked, even though the full ladder is generated.  To track
    the whole ladder, place one :class:`InharmonicOscillatorModel` per partial and
    compose them in an :class:`OscillatorBankModel` (INDEPENDENT).

    Physical parameters (Young's modulus, wire diameter, length, tension) can be
    converted to ``(f0, B)`` via :func:`wire_fundamental` and :func:`wire_inharmonicity`.

    Parameters
    ----------
    sensor_id : str
    f0 : float
        Fundamental frequency [Hz].
    fc : float
        Window center frequency [Hz] — which partial to track.
    dfc : float
        Window half-width [Hz].
    n_harmonics : int
        Number of partials to include in the candidate ladder.
    B : float
        Inharmonicity coefficient (``B = 0`` → harmonic, use :class:`HarmonicOscillatorModel`).
    """

    NAME = "Inharmonic Oscillator (windowed)"
    DESCRIPTION = "Windowed oscillator over an inharmonic ladder f_n = n·f0·√(1+B·n²)."
    VERSION = "3.1"

    def __init__(self, sensor_id, f0, fc, dfc, *, n_harmonics=3, B=0.0,
                 Q=1e4, fs=4096.0, measurement_uncertainty=1.0,
                 process_uncertainty=1e-3, p0=None, **kwargs):
        self.f0 = float(f0)
        self.inharmonicity = float(B)
        self.n_harmonics = int(n_harmonics)
        self.partial_freqs = partial_frequencies(f0, B, n_harmonics)
        super().__init__(sensor_id, self.partial_freqs, fc, dfc, Q=Q, fs=fs,
                         measurement_uncertainty=measurement_uncertainty,
                         process_uncertainty=process_uncertainty, p0=p0, **kwargs)


class HarmonicOscillatorModel(InharmonicOscillatorModel):
    """
    Perfectly harmonic ladder — :class:`InharmonicOscillatorModel` with ``B = 0``
    (partials at exact integer multiples ``f_n = n·f0``).

    All windowing and masking semantics of :class:`OscillatorModel` apply: passing
    ``fc`` and ``dfc`` selects which harmonic(s) this window tracks.  Only partials
    inside ``[fc ± dfc]`` enter the state; the rest are generated but silently dropped.
    """

    NAME = "Harmonic Oscillator (windowed, B=0)"
    DESCRIPTION = "Windowed oscillator over a harmonic ladder f_n = n·f0."
    VERSION = "3.1"

    def __init__(self, sensor_id, f0, fc, dfc, *, n_harmonics=3, **kwargs):
        if kwargs.pop("B", 0.0):
            raise TypeError("HarmonicOscillatorModel forces B=0; use InharmonicOscillatorModel for B>0.")
        super().__init__(sensor_id, f0, fc, dfc, n_harmonics=n_harmonics, B=0.0, **kwargs)



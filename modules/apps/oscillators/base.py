"""Shared oscillator-domain definitions — state/observable enums and the
wire/partial-frequency helpers every oscillator variant builds on."""
import numpy as np

from ...framework.constants import KalmanState, KalmanObservable, KalmanObservableReadout


# ======================================================================
# State / observable enums
# ======================================================================
class OscillatorState(KalmanState):
    REAL = "re"   # rotating-frame complex amplitude — real (in-phase) part
    IMAG = "im"   # rotating-frame complex amplitude — imaginary (quadrature) part


class OscillatorObservable(KalmanObservable):
    PSI = ""      # demodulated narrowband signal (complex envelope ψ)

# ======================================================================
# Physical parameter utilities
# ======================================================================
def wire_inharmonicity(E: float, d: float, L: float, T: float) -> float:
    """
    Inharmonicity coefficient B for a stiff wire under tension (Fletcher 1964).

    Predicts partial frequencies: ``f_n = n·f0·√(1 + B·n²)``.

    Parameters
    ----------
    E : Young's modulus [Pa]  (steel ≈ 2e11 Pa, tungsten ≈ 4e11 Pa)
    d : wire diameter [m]
    L : wire length [m]
    T : wire tension [N]

    Returns
    -------
    B : inharmonicity coefficient (dimensionless)
    """
    return np.pi**3 * E * d**4 / (64.0 * T * L**2)


def wire_fundamental(d: float, L: float, T: float, rho: float) -> float:
    """
    Fundamental transverse vibration frequency of a wire under tension.

    ``f0 = (1/2L) · √(T/μ)``,  ``μ = ρ·π·d²/4``  (linear mass density).

    Parameters
    ----------
    d   : wire diameter [m]
    L   : wire length [m]
    T   : wire tension [N]
    rho : material density [kg/m³]  (steel ≈ 7800, tungsten ≈ 19300)

    Returns
    -------
    f0 : fundamental frequency [Hz]
    """
    mu = rho * np.pi * d**2 / 4.0      # linear mass density [kg/m]
    return (1.0 / (2.0 * L)) * np.sqrt(T / mu)


# ======================================================================
# Inharmonic frequency helpers (shared)
# ======================================================================
def partial_frequencies(f0: float, B: float, n_harmonics: int) -> np.ndarray:
    """
    Fletcher (in)harmonic partial series: ``f_n = n·f0·√(1 + B·n²)``.

    ``B = 0`` gives the perfectly harmonic series ``f_n = n·f0``.  For a stiff wire
    under tension, B can be derived from physical parameters via
    :func:`wire_inharmonicity`.  A typical KAGRA suspension wire has B ≈ 0.1.

    Parameters
    ----------
    f0         : fundamental frequency [Hz]
    B          : inharmonicity coefficient (dimensionless, ≥ 0)
    n_harmonics: number of partials to compute

    Returns
    -------
    freqs : (n_harmonics,) array of partial frequencies [Hz]
    """
    n = np.arange(1, n_harmonics + 1, dtype=float)
    return n * f0 * np.sqrt(1.0 + B * n ** 2)



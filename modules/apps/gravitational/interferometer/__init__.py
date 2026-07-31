"""
kalman.apps.interferometer — gravitational-wave detector domain.

- spectral  — PSD / peak-fitting helpers;
- model     — :class:`InterferometerModel` (violin-mode oscillator bank);
- detectors — :class:`InterferometerK1`/`H1`/`L1`/`V1` configurations;
- control   — the violin-bank damping controller.
"""
from .spectral import (welch_psd, find_peaks_in_band, fit_lorentzian_peak, group_peaks_into_windows, lorentzian)
from .model import (InterferometerModel)
from .detectors import (InterferometerK1, InterferometerH1, InterferometerL1, InterferometerV1)
from .control import (ViolinBankController)

VARIANTS = {
    "k1": InterferometerK1,
    "h1": InterferometerH1,
    "l1": InterferometerL1,
    "v1": InterferometerV1,
}

__all__ = [
    "welch_psd",
    "find_peaks_in_band",
    "fit_lorentzian_peak",
    "group_peaks_into_windows",
    "lorentzian",
    "InterferometerModel",
    "InterferometerK1",
    "InterferometerH1",
    "InterferometerL1",
    "InterferometerV1",
    "ViolinBankController",
    "VARIANTS",
]

"""Per-detector configurations (K1 / H1 / L1 / V1)."""
from __future__ import annotations
"""GW interferometer noise subtraction via spectral-line Kalman tracking.

Two-stage workflow
------------------
Stage 1 -- peak fitting
    ``InterferometerModel.fit(sensor_id, strain, fs, bands)``
    Estimates PSD via Welch, detects spectral peaks per band with
    ``scipy.signal.find_peaks``, fits each peak to a Lorentzian profile to
    extract (f0, Q, amplitude), and groups nearby peaks into analysis windows.

Stage 2 -- Kalman tracking and subtraction
    The returned ``InterferometerModel`` (a specialised ``OscillatorBankModel``)
    tracks each window's phasor in the rotating frame -- the joint Kalman filter
    of Moisset et al. 2025 (arXiv:2509.20658).  After running the filter::

        record = KalmanFilter(model).forward(strain, uk=model.make_controller())
        clean  = model.subtract(strain, record)

    ``make_controller()`` returns a ``ViolinBankController`` that attaches one
    ``AdaptiveQRKalmanController`` per window, self-adjusting each window's
    process-noise level (Q) from empirical innovation statistics.

Noise sources
-------------
``NOISE_SOURCES = ("violin",)`` -- violin-mode suspension noise only.
Future additions:

* ``"powergrid"`` -- use ``PowerGridStateModel`` (already implemented in
  ``modules/apps/powergrids.py``) as a **separate parallel filter** since its
  PhasorOscillatorModel architecture (lab-frame, time-varying H, scalar voltage)
  is incompatible with ``OscillatorBankModel._broadcast_preprocess``.
* ``"omc_pcal"`` -- fixed-frequency PCal calibration lines can be added as
  additional ``OscillatorModel`` windows inside this bank (Q approx inf, known fc).

Detector-specific subclasses
-----------------------------
Pre-loaded per-run violin-mode parameters are available via detector subclasses::

    from kalman.apps.interferometer import InterferometerK1

    # Compute band limits from the built-in catalog and fit from synthetic data:
    bands = InterferometerK1.violin_bands("O3GK")
    model = InterferometerK1.fit("K1", strain, fs=4096, bands=bands)

    # Or download O3GK strain directly (requires gwpy):
    model = InterferometerK1.from_gps(1271267328, 1271281728, run="O3GK")
    ctrl  = model.make_controller()
    rec   = model.filter(strain, uk=ctrl, smooth=True)
    clean = model.subtract(strain, rec)

Available: InterferometerK1, InterferometerH1, InterferometerL1, InterferometerV1.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch as scipy_welch, find_peaks as scipy_find_peaks

from ...resonators.oscillators import OscillatorModel, OscillatorBankModel
from ....framework.controllers.adaptive_qr import AdaptiveQRKalmanController
from ....framework.control import KalmanController
from .model import InterferometerModel, _DetectorBase


class InterferometerK1(_DetectorBase):
    """KAGRA (K1) interferometer — Type-A intermediate-mass steel wire suspension.

    O3GK catalog: 88 measured peak positions from Moisset et al. 2025
    (arXiv:2509.20658, ROOT/Minuit2 fiQi.txt output):
    27 wires Mode 1 (173-185 Hz), 30 wires Mode 2 (405-430 Hz),
    31 wires Mode 3 (721-767 Hz).

    Example
    -------
    ::

        # Build directly from measured peak catalog (no strain needed):
        model = InterferometerK1.from_peaks("O3GK")

        # Or fit from raw strain / GWOSC download:
        bands = InterferometerK1.violin_bands("O3GK")
        model = InterferometerK1.fit("K1", strain, fs=4096, bands=bands)
        model = InterferometerK1.from_gps(1271267328, 1271281728, run="O3GK")
    """
    _IFO = "K1"
    _CATALOG: dict = {
        "O3GK": {
            # KAGRA Type-A steel wire -- Moisset et al. 2025 (arXiv:2509.20658)
            # 88 peaks from fiQi.txt (Minuit2 Lorentzian pre-fits on K1 O3GK PSD).
            # Q values are Minuit2 seeds; entries at 1.6e4 / 8.7e4 / 6.5e4 hit fit bounds.
            "bands": [(171.0, 186.0), (403.0, 432.0), (718.0, 770.0)],
            "peaks": [
                # Mode 1 (n=1): 27 wires, 173-185 Hz
                [(173.021, 16000), (174.035,  5549), (176.510, 16000), (177.929, 14300),
                 (178.295, 11678), (178.387, 24826), (178.425, 16000), (178.539,  7767),
                 (179.052, 14703), (179.247, 23496), (179.762, 12596), (179.833,  5385),
                 (180.061, 16000), (180.093, 86831), (180.253,  9935), (180.625,  5948),
                 (180.902,  7207), (180.962, 13543), (181.615, 23600), (181.661, 11721),
                 (182.093, 16000), (182.139, 26419), (182.543, 15517), (182.959, 12950),
                 (183.427, 16000), (184.314,  5767), (184.387,  8634)],
                # Mode 2 (n=2): 30 wires, 405-430 Hz
                [(404.961, 16000), (408.562, 16000), (414.901, 16000), (415.482, 16000),
                 (416.433, 32344), (416.998, 14305), (418.844, 12826), (419.270, 16000),
                 (419.512, 26866), (419.721, 15823), (419.906,  6990), (420.296, 16000),
                 (420.644, 12534), (421.157, 14321), (421.269, 10323), (421.325,  9972),
                 (421.659,  8486), (421.744, 13238), (422.415,  6173), (422.540, 44973),
                 (422.642,  6393), (422.949, 16000), (424.255, 18801), (424.658, 10359),
                 (425.184, 10192), (425.798,  7000), (427.256, 13959), (427.454, 16000),
                 (427.662, 16000), (429.796,  7510)],
                # Mode 3 (n=3): 31 wires, 721-767 Hz
                [(721.271, 13576), (728.291, 16000), (739.319, 10121), (743.367, 21125),
                 (743.762,  8904), (746.132, 11740), (746.455, 26279), (746.707, 10475),
                 (747.163, 22779), (749.812, 16000), (750.900,  9715), (751.171, 21157),
                 (751.249, 44711), (753.344, 16000), (753.651, 29196), (754.724, 22800),
                 (755.729, 11747), (756.340,  9638), (756.560, 15582), (756.750, 19724),
                 (756.786, 14823), (756.885,  6688), (758.686, 27985), (759.405, 19940),
                 (760.548, 17498), (762.359, 17498), (762.611, 64955), (762.930, 17228),
                 (764.139, 16000), (766.052, 23220), (767.439, 13156)],
            ],
            "Q_typical": 1.3e4, "fs": 4096.0,
        },
        "O4": {
            # Placeholder -- same wire geometry as O3GK; peaks not yet characterised
            "bands": [(171.0, 186.0), (403.0, 432.0), (718.0, 770.0)],
            "peaks": None,   # use from_gps(..., run="O4") + fit() instead
            "Q_typical": 1.3e4, "fs": 4096.0,
        },
    }


class InterferometerH1(_DetectorBase):
    """LIGO Hanford (H1) — fused-silica test-mass fiber suspension.

    Approximate Fletcher inharmonicity parameters for O3 / O4.
    Update ``_CATALOG`` entries with measured values for precise tracking.
    """
    _IFO = "H1"
    _CATALOG: dict = {
        "O3": {
            "f0": 500.0, "B": 5e-4, "n_harmonics": 3,
            "band_margin_hz": 15.0, "Q_typical": 1e6, "fs": 4096.0,
        },
        "O4": {
            "f0": 500.0, "B": 5e-4, "n_harmonics": 3,
            "band_margin_hz": 15.0, "Q_typical": 1e6, "fs": 4096.0,
        },
    }


class InterferometerL1(_DetectorBase):
    """LIGO Livingston (L1) — fused-silica test-mass fiber suspension.

    Approximate Fletcher inharmonicity parameters for O3 / O4.
    Update ``_CATALOG`` entries with measured values for precise tracking.
    """
    _IFO = "L1"
    _CATALOG: dict = {
        "O3": {
            # Slightly lower fundamental than H1 due to wire geometry differences
            "f0": 497.3, "B": 5e-4, "n_harmonics": 3,
            "band_margin_hz": 15.0, "Q_typical": 1e6, "fs": 4096.0,
        },
        "O4": {
            "f0": 497.3, "B": 5e-4, "n_harmonics": 3,
            "band_margin_hz": 15.0, "Q_typical": 1e6, "fs": 4096.0,
        },
    }


class InterferometerV1(_DetectorBase):
    """Virgo (V1) — last-stage monolithic fused-silica fiber suspension.

    Approximate Fletcher inharmonicity parameters for O3 / O4.
    Update ``_CATALOG`` entries with measured values for precise tracking.
    """
    _IFO = "V1"
    _CATALOG: dict = {
        "O3": {
            "f0": 380.0, "B": 3e-4, "n_harmonics": 3,
            "band_margin_hz": 15.0, "Q_typical": 5e5, "fs": 4096.0,
        },
        "O4": {
            "f0": 380.0, "B": 3e-4, "n_harmonics": 3,
            "band_margin_hz": 15.0, "Q_typical": 5e5, "fs": 4096.0,
        },
    }

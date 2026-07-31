"""Violin-mode bank controller (adaptive-QR leaves)."""
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


# ======================================================================
# Per-window leaf-level AdaptiveQR wrapper
# ======================================================================

class _LeafAdaptiveQR:
    """Drives one ``AdaptiveQRKalmanController`` per leaf inside an ``OscillatorModel``.

    ``OscillatorModel`` is a ``CompositeKalmanModel`` whose ``Q()`` is bound as
    a closure over its leaf submodels — mutating ``window._Q`` has no effect.
    This class targets each ``AnalyticOscillatorModel`` leaf directly so that the
    ``model._Q`` mutations performed by ``AdaptiveQRKalmanController._update``
    actually propagate through ``CompositeKalmanModel.Q()``.
    """

    def __init__(self, window: OscillatorModel, *, window_len: int = 30,
                 q_min: float = 0.1, q_max: float = 10.0):
        self._leaves = window.models  # list[AnalyticOscillatorModel]
        self._ctrls  = [
            AdaptiveQRKalmanController(leaf, window=window_len,
                                       q_min=q_min, q_max=q_max)
            for leaf in self._leaves
        ]

    def update(self, x_hat_win, P_win, t, nu_win):
        """Scale each leaf's Q from the window-level innovation ``nu_win`` (shape (2,1)).

        All leaves in a FUSED window share the same 2-element [Re, Im] innovation
        channel, so each leaf receives the same ``nu_win``.  State/covariance are
        sliced to match each leaf's own ``n=2`` state block.
        """
        offset = 0
        for ctrl in self._ctrls:
            n = ctrl.model.n
            ctrl._update(
                x_hat_win[offset:offset + n],
                P_win[offset:offset + n, offset:offset + n],
                t,
                nu=nu_win,
            )
            offset += n

    def reset(self):
        for ctrl in self._ctrls:
            ctrl.reset()


# ======================================================================
# Bank-level controller (one dispatcher for all windows)
# ======================================================================

class ViolinBankController(KalmanController):
    """Bank-level controller: dispatches per-window ``AdaptiveQR`` from bank-level nu/P.

    Intended to be passed as ``uk`` to ``KalmanFilter.forward()``::

        ctrl   = model.make_controller()
        record = KalmanFilter(model).forward(strain, uk=ctrl)

    Each call slices the bank-level state/covariance/innovation into per-window
    sub-vectors and drives the corresponding ``_LeafAdaptiveQR``.  No actuation
    is produced (``_policy`` returns zeros).
    """

    def __init__(self, bank_model: "InterferometerModel",
                 window_aqrs: list[_LeafAdaptiveQR],
                 state_offsets: list[int]):
        # Initialize before super().__init__() since that calls reset()
        self._window_aqrs  = window_aqrs
        self._state_offsets = state_offsets
        super().__init__(bank_model)

    def _policy(self, x_hat, P, t, *, nu=None):
        return np.zeros((max(self.model.l, 1), 1))

    def _update(self, x_hat, P, t, *, nu=None):
        if nu is None:
            return None
        for i, (win, aqr) in enumerate(zip(self.model.windows, self._window_aqrs)):
            o    = self._state_offsets[i]
            x_i  = x_hat[o:o + win.n]
            P_i  = P[o:o + win.n, o:o + win.n]
            # OscillatorBankModel uses INDEPENDENT composition →
            # bank nu = [nu_win0; nu_win1; ...], each window contributes 2 elements
            # (FUSED internally → single 2-D [Re,Im] observation channel per window).
            nu_i = nu[2 * i: 2 * i + 2]
            aqr.update(x_i, P_i, t, nu_i)
        return None

    def reset(self):
        super().reset()
        for aqr in self._window_aqrs:
            aqr.reset()


# ======================================================================
# InterferometerModel
# ======================================================================


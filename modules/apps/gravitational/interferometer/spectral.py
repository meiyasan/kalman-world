"""Spectral analysis helpers — PSD, Lorentzian peak fits, window grouping."""
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
# Spectral analysis utilities
# ======================================================================

def welch_psd(strain, fs: float, nperseg: int = None, **kwargs):
    """PSD estimate via Welch's method.

    Parameters
    ----------
    strain : array-like, shape (N,)
    fs : float, sample rate [Hz]
    nperseg : int, optional
        FFT segment length.  Default: ``fs`` (1-second segments).

    Returns
    -------
    freqs : (M,) ndarray [Hz]
    psd   : (M,) ndarray [strain^2/Hz]
    """
    x = np.asarray(strain, dtype=float).ravel()
    if nperseg is None:
        nperseg = int(fs)
    return scipy_welch(x, fs=fs, nperseg=nperseg, **kwargs)


def lorentzian(f, f0: float, gamma: float, amplitude: float):
    """Single Lorentzian peak: ``A · (γ/2)² / ((f−f0)² + (γ/2)²)``.

    At resonance (f = f0) the function evaluates to ``A``.
    Half-power bandwidth = ``γ`` [Hz];  Q = f0 / γ.
    """
    return amplitude * (gamma / 2.0) ** 2 / ((f - f0) ** 2 + (gamma / 2.0) ** 2)


def fit_lorentzian_peak(freqs, psd, f_center: float, f_half_width: float) -> dict:
    """Fit one Lorentzian to PSD within ``[f_center ± f_half_width]``.

    Uses ``scipy.optimize.curve_fit`` with physically motivated bounds.
    Pattern mirrors ``OscillatorModel.fit_inharmonicity`` (bounded, maxfev=10000).

    Parameters
    ----------
    freqs, psd : array-like
        Full PSD from :func:`welch_psd`.
    f_center, f_half_width : float
        Search band centre and half-width [Hz].

    Returns
    -------
    dict with keys ``f0``, ``gamma``, ``Q``, ``amplitude``.
    ``Q = f0 / gamma``.  Raises ``RuntimeError`` if the fit fails.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd   = np.asarray(psd,   dtype=float)
    mask  = np.abs(freqs - f_center) <= f_half_width
    if mask.sum() < 4:
        raise RuntimeError(
            f"Too few PSD points in [{f_center - f_half_width:.1f}, "
            f"{f_center + f_half_width:.1f}] Hz for Lorentzian fit."
        )
    f_band = freqs[mask]
    p_band = psd[mask]

    p0_amp   = float(p_band.max())
    p0_gamma = f_half_width / 5.0   # rough initial linewidth estimate
    p0       = [f_center, p0_gamma, p0_amp]
    bounds   = (
        [f_center - f_half_width, 1e-6, 0.0],
        [f_center + f_half_width, f_half_width * 2, p0_amp * 1e4],
    )
    try:
        popt, _ = curve_fit(lorentzian, f_band, p_band, p0=p0,
                            bounds=bounds, maxfev=10000)
    except RuntimeError as e:
        raise RuntimeError(f"Lorentzian fit failed near {f_center:.2f} Hz: {e}") from e

    f0_fit, gamma_fit, amp_fit = float(popt[0]), float(popt[1]), float(popt[2])
    return {
        "f0":        f0_fit,
        "gamma":     gamma_fit,
        "Q":         f0_fit / max(gamma_fit, 1e-12),
        "amplitude": amp_fit,
    }


def find_peaks_in_band(freqs, psd, f_low: float, f_high: float,
                       *, min_prominence=None, **find_peaks_kwargs) -> np.ndarray:
    """Detect PSD peaks inside ``[f_low, f_high]`` Hz.

    Parameters
    ----------
    freqs, psd : array-like
        Full PSD from :func:`welch_psd`.
    f_low, f_high : float
        Frequency band [Hz].
    min_prominence : float, optional
        Minimum peak prominence in PSD units.  Defaults to
        ``0.3 × (band max − band median)``.

    Returns
    -------
    peak_freqs : ndarray of peak frequencies [Hz], sorted ascending.
    """
    freqs = np.asarray(freqs, dtype=float)
    psd   = np.asarray(psd,   dtype=float)
    mask  = (freqs >= f_low) & (freqs <= f_high)
    f_band = freqs[mask]
    p_band = psd[mask]
    if f_band.size == 0:
        return np.array([])

    if min_prominence is None:
        # The floor guards only the degenerate flat-band case (max == median);
        # it must scale with the PSD's own magnitude, not a fixed absolute value —
        # GW strain PSDs run ~1e-36–1e-42, far below a unit-scale constant like 1e-30,
        # which would silently reject every genuine peak.
        #
        # The 0.3 coefficient (not 0.1) matters once a band spans many frequency
        # bins: with hundreds of bins, pure-noise fluctuations are likely to exceed
        # a lax 10%-of-peak threshold *somewhere* by chance, flooding a weak-SNR
        # band with spurious "peaks" that then get merged into one huge window by
        # group_peaks_into_windows — which breaks OscillatorBankModel's requirement
        # that every window share one sample rate (each window's dfc sets its own
        # decimated fs). 0.3 keeps the true peak (which sits far above the noise
        # floor) while rejecting single-bin statistical noise fluctuations.
        floor = 1e-12 * max(abs(p_band.max()), 1e-300)
        min_prominence = 0.3 * max(p_band.max() - np.median(p_band), floor)
    idx, _ = scipy_find_peaks(p_band, prominence=min_prominence, **find_peaks_kwargs)
    return np.sort(f_band[idx])


def _estimate_half_max_width(freqs, psd, f_peak: float, f_low: float, f_high: float,
                             min_bins: int = 3) -> float:
    """Estimate a Lorentzian fit half-width from the peak's own half-power points.

    Walks outward from the peak bin until the PSD drops to the local background
    plus half the peak's excess above it (the standard half-power/FWHM
    definition), instead of using an arbitrary fraction of the whole search
    band. A search band is deliberately wide (it only has to bracket the true
    line, which may sit anywhere inside), so sizing the *fit* window from the
    band width — as opposed to the observed peak shape — makes the Lorentzian
    fit degenerate for any line much narrower than the band: with only 1-2
    bins actually elevated out of dozens across a wide, mostly-flat window,
    ``curve_fit`` has little to constrain it and returns close to its own
    initial guess rather than a real measurement of the line.

    Returns a half-width no larger than half the search band (a fallback for
    peaks too close to the band edge to bracket on both sides).
    """
    freqs = np.asarray(freqs, dtype=float)
    psd   = np.asarray(psd,   dtype=float)
    mask  = (freqs >= f_low) & (freqs <= f_high)
    f_band = freqs[mask]
    p_band = psd[mask]
    band_half_width = (f_high - f_low) / 2.0
    if f_band.size < 2 * min_bins + 1:
        return band_half_width

    df = f_band[1] - f_band[0]
    i_peak = int(np.argmin(np.abs(f_band - f_peak)))
    background = np.median(p_band)
    half_level = background + 0.5 * (p_band[i_peak] - background)

    i_left = i_peak
    while i_left > 0 and p_band[i_left] > half_level:
        i_left -= 1
    i_right = i_peak
    while i_right < len(p_band) - 1 and p_band[i_right] > half_level:
        i_right += 1

    half_w = max(f_band[i_peak] - f_band[i_left], f_band[i_right] - f_band[i_peak],
                min_bins * df)
    return min(half_w, band_half_width)


def group_peaks_into_windows(peak_fits: list[dict], *,
                             merge_gap_hz: float = 2.0,
                             margin_hz: float = 1.0) -> list[dict]:
    """Group nearby fitted peaks into analysis windows.

    Peaks separated by less than ``merge_gap_hz`` are placed in the same
    ``OscillatorModel`` window (FUSED — they share one demodulated channel,
    matching the Moisset et al. joint filter design).  Each window is
    characterised by:

    * ``fc``    — window centre [Hz]
    * ``dfc``   — window half-width [Hz]
    * ``peaks`` — list of peak-fit dicts that fall inside the window

    Parameters
    ----------
    peak_fits : list of dicts (output of :func:`fit_lorentzian_peak`)
    merge_gap_hz : float
        Peaks within this distance [Hz] are merged into one window.
    margin_hz : float
        Extra padding added to ``dfc`` beyond the outermost peak offsets.

    Returns
    -------
    windows : list of dicts with keys ``fc``, ``dfc``, ``peaks``.
    """
    if not peak_fits:
        return []

    sorted_fits = sorted(peak_fits, key=lambda d: d["f0"])
    groups: list[list[dict]] = []
    current = [sorted_fits[0]]
    for fit in sorted_fits[1:]:
        if fit["f0"] - current[-1]["f0"] <= merge_gap_hz:
            current.append(fit)
        else:
            groups.append(current)
            current = [fit]
    groups.append(current)

    windows = []
    for grp in groups:
        freqs = [d["f0"] for d in grp]
        fc  = float(np.mean(freqs))
        dfc = max(abs(f - fc) for f in freqs) + margin_hz
        windows.append({"fc": fc, "dfc": dfc, "peaks": grp})
    return windows



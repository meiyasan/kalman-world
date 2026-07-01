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
from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch as scipy_welch, find_peaks as scipy_find_peaks

from .oscillators import OscillatorModel, OscillatorBankModel
from ..framework.controllers.adaptive_qr import AdaptiveQRKalmanController
from ..framework.control import KalmanController


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
        ``0.1 × (band max − band median)``.

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
        min_prominence = 0.1 * max(p_band.max() - np.median(p_band), 1e-30)
    idx, _ = scipy_find_peaks(p_band, prominence=min_prominence, **find_peaks_kwargs)
    return np.sort(f_band[idx])


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

class InterferometerModel(OscillatorBankModel):
    """Noise-subtraction model for GW interferometer spectral lines.

    Inherits ``OscillatorBankModel`` (bank of windowed rotating-frame Kalman
    filters) and adds:

    * A classmethod :meth:`fit` that auto-builds the window bank from raw strain
      data by fitting Lorentzian profiles to detected PSD peaks.
    * :meth:`make_controller` returning a ``ViolinBankController`` with one
      ``AdaptiveQRKalmanController`` per window for closed-loop Q adaptation.
    * :meth:`subtract` convenience method (thin wrapper over inherited
      ``reconstruct``).

    Typical usage::

        model  = InterferometerModel.fit(
            "H1", strain, fs=4096,
            bands=[(170, 186), (402, 432), (718, 770)],   # KAGRA violin bands
        )
        ctrl   = model.make_controller()
        record = KalmanFilter(model).forward(strain, uk=ctrl)
        clean  = model.subtract(strain, record)
        print(model.peak_summary())

    Pre-fitted parameters can be passed directly::

        win1   = OscillatorModel("H1", [175.3, 177.1], fc=176.2, dfc=4.0,
                                 Q=[1e5, 8e4], fs=4096)
        model  = InterferometerModel("H1", [win1])

    Noise sources
    -------------
    ``NOISE_SOURCES = ("violin",)``

    Future: ``"powergrid"`` via ``PowerGridStateModel`` (separate parallel
    filter — incompatible with this bank's broadcast-preprocess architecture);
    ``"omc_pcal"`` as additional fixed-frequency windows inside this bank.
    """

    NAME        = "Interferometer Noise Subtraction"
    DESCRIPTION = ("OscillatorBankModel with Lorentzian peak-fitting and per-window "
                   "AdaptiveQR closed-loop Q adaptation.  Violin suspension only.")
    VERSION     = "1.0"

    NOISE_SOURCES = ("violin",)

    def __init__(self, sensor_id: str, windows: list[OscillatorModel], *,
                 aqr_window: int = 30,
                 aqr_q_min: float = 0.1,
                 aqr_q_max: float = 10.0,
                 **kwargs):
        """
        Parameters
        ----------
        sensor_id : str
            Channel / detector identifier (e.g. ``"H1"``, ``"K1"``).
        windows : list of OscillatorModel
            One window per violin-mode group.  Build via :meth:`fit` or supply
            manually for pre-known peak parameters.
        aqr_window : int
            Innovation buffer length for each ``AdaptiveQRKalmanController``.
        aqr_q_min, aqr_q_max : float
            Clipping bounds for the Q-scaling factor per window.
        """
        self.sensor_id = sensor_id
        super().__init__(*windows, **kwargs)

        self._window_aqr: list[_LeafAdaptiveQR] = [
            _LeafAdaptiveQR(win, window_len=aqr_window,
                            q_min=aqr_q_min, q_max=aqr_q_max)
            for win in self.windows
        ]
        # Cumulative state offsets for bank-level nu/P slicing in ViolinBankController.
        self._state_offsets: list[int] = list(
            np.cumsum([0] + [w.n for w in self.windows[:-1]])
        )

    # ------------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------------

    def make_controller(self) -> ViolinBankController:
        """Return a ``ViolinBankController`` for ``KalmanFilter.forward(uk=...)``.

        The controller self-adjusts each window's process-noise Q from empirical
        innovation statistics (one noise level per ``wc``/window centre).
        """
        return ViolinBankController(self, self._window_aqr, self._state_offsets)

    # ------------------------------------------------------------------
    # Classmethod constructor: fit from raw strain
    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, sensor_id: str,
            strain,
            fs: float,
            bands: list[tuple[float, float]],
            *,
            Q_init: float = 1e4,
            merge_gap_hz: float = 2.0,
            window_margin_hz: float = 1.0,
            min_prominence=None,
            measurement_uncertainty: float = 1.0,
            process_uncertainty: float = 1e-3,
            nperseg: int = None,
            **kwargs) -> "InterferometerModel":
        """Fit Lorentzian peaks in each band and build an ``InterferometerModel``.

        Parameters
        ----------
        sensor_id : str
        strain : array-like, shape (N,)
            Raw strain time series.
        fs : float
            Sample rate [Hz].
        bands : list of (f_low, f_high) tuples
            Frequency bands to search for peaks, e.g.
            ``[(170, 186), (402, 432), (718, 770)]`` for KAGRA violin modes.
        Q_init : float
            Fallback Q when Lorentzian fit is unavailable for a peak.
        merge_gap_hz : float
            Peaks within this Hz are merged into one FUSED window.
        window_margin_hz : float
            Extra padding on each side of the outermost peak in a window.
        min_prominence : float, optional
            Minimum PSD prominence for ``find_peaks_in_band``.  Auto-scaled
            per band if ``None``.
        measurement_uncertainty, process_uncertainty : float
            Passed through to each ``OscillatorModel``.
        nperseg : int, optional
            Welch segment length.  Default: ``int(fs)`` (1-second segments).

        Returns
        -------
        InterferometerModel

        Workflow
        --------
        1. ``welch_psd`` → (freqs, psd)
        2. For each band: ``find_peaks_in_band`` → candidate peak frequencies
        3. For each candidate: ``fit_lorentzian_peak`` → (f0, Q, amplitude)
        4. ``group_peaks_into_windows`` → [(fc, dfc, peaks), …]
        5. Build one ``OscillatorModel`` per window (Q per mode from fit)
        6. Construct ``InterferometerModel``
        """
        freqs, psd = welch_psd(strain, fs, nperseg=nperseg)

        all_fits: list[dict] = []
        for f_low, f_high in bands:
            peak_freqs = find_peaks_in_band(freqs, psd, f_low, f_high,
                                            min_prominence=min_prominence)
            for fp in peak_freqs:
                half_w = (f_high - f_low) / (2.0 * max(len(peak_freqs), 1))
                half_w = max(half_w, 0.5)
                try:
                    fit = fit_lorentzian_peak(freqs, psd, fp, half_w)
                except RuntimeError:
                    fit = {"f0": fp, "gamma": fp / Q_init, "Q": Q_init,
                           "amplitude": float(psd[np.argmin(np.abs(freqs - fp))])}
                all_fits.append(fit)

        if not all_fits:
            raise ValueError(
                "No peaks found in any of the supplied bands. "
                "Check that `strain` contains the expected spectral lines and "
                "that the band limits are correct."
            )

        win_defs = group_peaks_into_windows(all_fits,
                                            merge_gap_hz=merge_gap_hz,
                                            margin_hz=window_margin_hz)

        windows: list[OscillatorModel] = []
        for wd in win_defs:
            peak_freqs_w = [p["f0"] for p in wd["peaks"]]
            Qs_w         = [p["Q"]  for p in wd["peaks"]]
            win = OscillatorModel(
                sensor_id,
                peak_freqs_w,
                fc=wd["fc"],
                dfc=wd["dfc"],
                Q=Qs_w if len(Qs_w) > 1 else Qs_w[0],
                fs=fs,
                measurement_uncertainty=measurement_uncertainty,
                process_uncertainty=process_uncertainty,
            )
            win._fit_results = wd["peaks"]  # store for peak_summary()
            windows.append(win)

        return cls(sensor_id, windows, **kwargs)

    # ------------------------------------------------------------------
    # Post-filter convenience
    # ------------------------------------------------------------------

    def subtract(self, strain, record) -> np.ndarray:
        """Return ``strain − reconstruct(record)``: the noise-subtracted residual."""
        return np.asarray(strain, dtype=float).ravel() - self.reconstruct(record)

    def peak_summary(self) -> list[dict]:
        """Fitted peak parameters (f0, Q, fc, dfc) per window.

        Returns a list of dicts — one per window — each containing:
        ``fc``, ``dfc``, and a ``peaks`` list with ``f0``, ``Q``, ``amplitude``
        per mode.  Only populated when the model was built via :meth:`fit`.
        """
        out = []
        for win in self.windows:
            entry: dict = {"fc": win.fc, "dfc": win.dfc}
            entry["peaks"] = getattr(win, "_fit_results", [
                {"f0": float(f), "Q": None, "amplitude": None}
                for f in win.frequencies
            ])
            out.append(entry)
        return out


# ======================================================================
# Violin-mode parameter catalog per detector and observation run
# ======================================================================



# ======================================================================
# gwpy download + caching helpers
# ======================================================================

def _get_cache_path(cache_dir, ifo: str, gps_start: float, gps_end: float,
                    fs: float):
    """Return a Path for the cached strain .npz file."""
    from pathlib import Path
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "kalman_framework"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ifo}_{int(gps_start)}_{int(gps_end)}_{int(fs)}Hz.npz"


def _download_strain(ifo: str, gps_start: float, gps_end: float,
                     channel: str = None, target_fs: float = 4096.0):
    """Download strain via gwpy and return (strain_array, actual_fs).

    Lazy-imports gwpy so the rest of the module works without it.
    Uses GWOSC open data (``fetch_open_data``) by default; pass a NDS2
    *channel* name to use ``TimeSeries.get`` instead.
    """
    try:
        from gwpy.timeseries import TimeSeries  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "gwpy is required for GPS-based data download. "
            "Install it with: pip install gwpy"
        ) from exc

    if channel is not None:
        ts = TimeSeries.get(channel, gps_start, gps_end, verbose=True)
    else:
        ts = TimeSeries.fetch_open_data(ifo, gps_start, gps_end,
                                         sample_rate=int(target_fs))
    return np.asarray(ts.value, dtype=float), float(ts.sample_rate.value)


# ======================================================================
# Detector-specific base class
# ======================================================================

class _DetectorBase(InterferometerModel):
    """Base for detector-specific ``InterferometerModel`` subclasses.

    Subclasses set ``_IFO`` and ``_CATALOG`` to register known violin-mode
    parameters per observation run.

    Two construction paths are available:

    * :meth:`from_peaks` — build directly from catalog peak list (no strain
      needed; K1 O3GK only so far).
    * :meth:`from_gps` — download strain from GWOSC via gwpy, cache it, and
      call :meth:`InterferometerModel.fit` with the catalog bands.
    """

    _IFO: str     = ""
    _CATALOG: dict = {}

    @classmethod
    def latest_run(cls) -> str:
        """Return the most recently registered observation run key."""
        return list(cls._CATALOG.keys())[-1]

    @classmethod
    def violin_bands(cls, run: str) -> list:
        """Return ``[(f_low, f_high), …]`` search bands for a given run.

        When the catalog entry contains a ``"bands"`` key (from direct
        measurement), those values are returned as-is.  Otherwise, band
        centres are computed from the Fletcher inharmonicity series
        ``f_n = n·f0·√(1 + B·n²)`` and padded by ``band_margin_hz``.

        Parameters
        ----------
        run : str
            Observation run key (e.g. ``"O3GK"``, ``"O4"``).

        Returns
        -------
        list of (f_low, f_high) float tuples [Hz]
        """
        from .oscillators import partial_frequencies as _pf
        if run not in cls._CATALOG:
            raise KeyError(
                f"Run '{run}' not in {cls.__name__}._CATALOG. "
                f"Available: {list(cls._CATALOG.keys())}"
            )
        p = cls._CATALOG[run]
        if "bands" in p:
            return [(float(b[0]), float(b[1])) for b in p["bands"]]
        partials = _pf(p["f0"], p["B"], p["n_harmonics"])
        m = p.get("band_margin_hz", 10.0)
        return [(float(f - m), float(f + m)) for f in partials]

    @classmethod
    def preloaded_peaks(cls, run: str) -> list:
        """Return the catalog peak list for *run* as ``[mode1, mode2, ...]``.

        Each element is a list of ``(f0_hz, Q)`` tuples measured from real data.
        Raises ``KeyError`` for unknown runs and ``ValueError`` when no peaks are
        stored for that run (e.g. K1 O4 placeholder).
        """
        if run not in cls._CATALOG:
            raise KeyError(
                f"Run '{run}' not in {cls.__name__}._CATALOG. "
                f"Available: {list(cls._CATALOG.keys())}"
            )
        peaks = cls._CATALOG[run].get("peaks")
        if not peaks:
            raise ValueError(
                f"No preloaded peaks for {cls._IFO} {run}. "
                "Use from_gps() + fit() to detect peaks from strain data."
            )
        return peaks

    @classmethod
    def from_peaks(cls, run: str = None, *,
                   dfc: float = None,
                   measurement_uncertainty: float = 1.0,
                   process_uncertainty: float = 1e-3,
                   **kwargs) -> "_DetectorBase":
        """Build an ``InterferometerModel`` from catalog peak positions.

        Uses the preloaded ``(f0, Q)`` pairs stored in the catalog, bypassing
        the Welch PSD fitting stage entirely.  Suitable for K1 O3GK where the
        88 peak positions are known from Moisset et al. 2025 (arXiv:2509.20658).

        Creates one ``OscillatorModel`` window per mode (one per band), with all
        catalogue peaks of that mode passed as ``frequencies``.  All windows share
        the same ``dfc`` (required by ``OscillatorBankModel``), chosen as the
        widest band half-width plus 15 % margin — or overridden via ``dfc``.

        Parameters
        ----------
        run : str, optional
            Observation run (default: :meth:`latest_run`).
        dfc : float, optional
            Common half-bandwidth [Hz] for all mode windows.  Default: auto
            from the widest catalog band (e.g. ~30 Hz for K1 O3GK).
        measurement_uncertainty, process_uncertainty : float
            Passed to each ``OscillatorModel``.

        Returns
        -------
        Instance of the calling subclass.

        Example
        -------
        ::

            model = InterferometerK1.from_peaks("O3GK")
            rec   = model.filter(strain, uk=model.make_controller())
            clean = model.subtract(strain, rec)
        """
        if run is None:
            run = cls.latest_run()
        mode_peak_lists = cls.preloaded_peaks(run)
        p    = cls._CATALOG[run]
        ifo  = cls._IFO
        fs   = float(p.get("fs", 4096.0))
        bands = [(float(b[0]), float(b[1])) for b in p["bands"]]

        if dfc is None:
            # One common dfc for all windows (required by OscillatorBankModel).
            # Use the widest mode half-span + 15% margin so all peaks stay in-band.
            dfc = max((hi - lo) / 2.0 for lo, hi in bands) * 1.15

        windows: list[OscillatorModel] = []
        for mode_peaks, (band_lo, band_hi) in zip(mode_peak_lists, bands):
            fc        = (band_lo + band_hi) / 2.0
            peak_freqs = [float(f0) for f0, _ in mode_peaks]
            Qs         = [float(Q)  for _,  Q in mode_peaks]
            win = OscillatorModel(
                ifo, peak_freqs,
                fc=fc, dfc=dfc,
                Q=Qs if len(Qs) > 1 else Qs[0],
                fs=fs,
                measurement_uncertainty=measurement_uncertainty,
                process_uncertainty=process_uncertainty,
            )
            windows.append(win)

        return cls(ifo, windows, **kwargs)

    @classmethod
    def from_gps(cls, gps_start: float, gps_end: float, *,
                 run: str = None,
                 channel: str = None,
                 fs: float = None,
                 cache: bool = True,
                 cache_dir=None,
                 **kwargs) -> "_DetectorBase":
        """Download strain from GWOSC and build a fitted interferometer model.

        Requires gwpy (``pip install gwpy``).  Downloaded data is cached as a
        ``.npz`` file in *cache_dir* (default ``~/.cache/kalman_framework/``)
        so subsequent calls with the same GPS range are instant.

        Parameters
        ----------
        gps_start, gps_end : float
            GPS time range [s].
        run : str, optional
            Observation run key (e.g. ``"O3GK"``).  Defaults to
            :meth:`latest_run`.
        channel : str, optional
            NDS2 channel name for private data.  When set, uses
            ``gwpy.TimeSeries.get`` instead of ``fetch_open_data``.
        fs : float, optional
            Target sample rate [Hz].  Defaults to the catalog entry's ``fs``.
        cache : bool
            Cache the downloaded timeseries on disk.  Default ``True``.
        cache_dir : str or Path, optional
            Cache directory.  Default ``~/.cache/kalman_framework/``.
        **kwargs
            Forwarded to :meth:`InterferometerModel.fit` (e.g.
            ``measurement_uncertainty``, ``process_uncertainty``).

        Returns
        -------
        Instance of the calling subclass.

        Example
        -------
        ::

            from kalman.apps.interferometer import InterferometerK1
            model = InterferometerK1.from_gps(1271267328, 1271281728, run="O3GK")
            rec   = model.filter(strain, uk=model.make_controller(), smooth=True)
            clean = model.subtract(strain, rec)
        """
        if run is None:
            run = cls.latest_run()
        p      = cls._CATALOG[run]
        tfs    = fs or p.get("fs", 4096.0)
        ifo    = cls._IFO

        if cache:
            cache_path = _get_cache_path(cache_dir, ifo, gps_start, gps_end, tfs)
            if cache_path.exists():
                data      = np.load(str(cache_path))
                strain    = data["strain"]
                actual_fs = float(data["fs"])
            else:
                strain, actual_fs = _download_strain(ifo, gps_start, gps_end,
                                                      channel, tfs)
                np.savez(str(cache_path), strain=strain, fs=actual_fs)
        else:
            strain, actual_fs = _download_strain(ifo, gps_start, gps_end,
                                                  channel, tfs)

        bands = cls.violin_bands(run)
        return cls.fit(ifo, strain, actual_fs, bands, **kwargs)


# ======================================================================
# Detector-specific subclasses
# ======================================================================

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

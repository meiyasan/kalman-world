"""Interferometer violin-mode bank model."""
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
from .control import _LeafAdaptiveQR, ViolinBankController
from .spectral import _estimate_half_max_width, find_peaks_in_band, fit_lorentzian_peak, group_peaks_into_windows, welch_psd


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
                half_w = _estimate_half_max_width(freqs, psd, fp, f_low, f_high)
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
        from ...resonators.oscillators import partial_frequencies as _pf
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


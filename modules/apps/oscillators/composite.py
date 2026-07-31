"""Windowed rotating-frame composite oscillator — the KAGRA violin-mode variant."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from ...framework.models.linear import LinearKalmanModel
from ...framework.models.composite import CompositeKalmanModel
from ...framework.constants import KalmanState, KalmanObservable, KalmanObservableReadout
from .analytic import AnalyticOscillatorModel
from .base import (OscillatorState, OscillatorObservable, wire_inharmonicity,
                   wire_fundamental, partial_frequencies)

# ======================================================================
# One analysis window  [fc ± dfc]  — the atomic windowed model
# ======================================================================
class OscillatorModel(CompositeKalmanModel):
    """
    One analysis window ``[fc ± dfc]`` over a **real** signal — the generic rotating-frame
    oscillator tracker.

    ``frequencies`` is the full candidate mode list (already expanded, e.g. from
    :func:`partial_frequencies`).  The window acts as a **mask**: only modes with
    ``|f − fc| ≤ dfc`` are kept and tracked; the rest are silently dropped from the
    state.  This is the key property: even if :class:`HarmonicOscillatorModel` or
    :class:`InharmonicOscillatorModel` generates a full ladder ``[f0, 2f0, 3f0, …]``,
    only those partials inside ``[fc ± dfc]`` are tracked by *this* window.  To track
    the whole ladder, stack one window per partial inside an
    :class:`OscillatorBankModel`.

    The in-band modes share the single demodulated band channel in **FUSED** mode
    (``H = [I₂ … I₂]``, ``z = Σ_i ψ_i``) — the joint Kalman filter of Moisset et al.
    (2025) / Mukherjee & Finn (1999).  A single in-band mode is the trivial (degenerate)
    case.

    **Signal flow:**

    * ``preprocess`` — real strain → Hilbert → demodulate at ``fc`` → ``[Re, Im]``
    * Filter runs in rotating frame on the ``[Re, Im]`` envelope
    * ``postprocess`` — ``[Re, Im]`` → remodulate at ``fc`` → **real band reconstruction**

    The postprocess remodulation ensures that ``sample().zk`` is a real signal in the
    same space as the model's input, so ``model.filter(sample.zk)`` works directly.

    Parameters
    ----------
    sensor_id : str
    frequencies : float | array-like
        Candidate mode frequencies [Hz]; masked to those inside ``[fc ± dfc]``.
    fc : float
        Window center frequency [Hz] — the demodulation carrier.
    dfc : float
        Window half-width [Hz].
    Q : float | list
        Quality factor (scalar, or one per kept mode).
    fs : float
    measurement_uncertainty, process_uncertainty : float
    """

    NAME = "Windowed Oscillator (one fc ± dfc band)"
    DESCRIPTION = ("One demodulated analysis window; in-band modes combined FUSED "
                   "(z = Σ ψ_i). Real in → filter → real band out.")
    VERSION = "3.1"

    partial_frequencies = staticmethod(partial_frequencies)

    @property
    def states(self):
        return OscillatorState
    @property
    def observables(self):
        return OscillatorObservable

    def __init__(self, sensor_id, frequencies, fc, dfc, *,
                 Q=1e4, fs=4096.0, measurement_uncertainty=1.0,
                 process_uncertainty=1e-3, p0=None, **kwargs):

        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        in_band = freqs[np.abs(freqs - fc) <= dfc]                # ← the window mask
        if in_band.size == 0:
            raise ValueError(
                f"OscillatorModel: no frequency within window {fc}±{dfc} Hz "
                f"(candidates: {np.round(freqs, 3)})")

        Qs = [float(Q)] * len(in_band) if np.isscalar(Q) else list(Q)
        self.fc = float(fc)
        self.dfc = float(dfc)
        self.frequencies = in_band
        self._wc = 2.0 * np.pi * float(fc)                        # angular carrier ω_c = 2π f_c

        # Decimated sample rate: 4× dfc gives 2× oversampling above Nyquist
        self._fs_original = float(fs)
        self._fs_dec = 4.0 * float(dfc)
        self._preprocess_N = None

        p0_var = float(p0) if p0 is not None else 1e3 * measurement_uncertainty * np.sqrt(2)
        blocks = [AnalyticOscillatorModel(
                      f"{sensor_id}@{f:.3f}Hz", fs=self._fs_dec,
                      f0=f, fc=fc, Q=q,
                      measurement_uncertainty=measurement_uncertainty,
                      process_uncertainty=process_uncertainty,
                      P0=p0_var * np.eye(2))
                  for f, q in zip(in_band, Qs)]

        super().__init__(*blocks, obs_mode=KalmanObservableReadout.FUSED, **kwargs)
        self.fs = self._fs_dec

        self.preprocess = self._demod_preprocess
        self.postprocess = self._identity_postprocess

    # --- resample_poly helper (handles float sample rates) ---
    @staticmethod
    def _resample(y, fs_old, fs_new, tolerance=1e2, kaiser_beta=12.0):
        """Rational resample with **edge-replicate padding** and a **sharp Kaiser
        anti-alias/anti-image filter**.

        Two boundary effects of a plain ``resample_poly`` are corrected here:

        * **Edge transient** — ``resample_poly`` implicitly zero-pads beyond the
          data, so its FIR runs into zeros at the edges; on **upsampling** the
          trailing ~half-filter of output ramps linearly to zero (an asymmetric
          tail decay — the start recovers, the end collapses).  We pad both ends
          with the edge sample, resample, then trim the padding back out, so the
          FIR always sees real data across the original span.  Output length
          matches the unpadded ``resample_poly(y, up, down)`` exactly.  The pad is
          a **multiple of ``down``** so the trim offset ``pad·up/down`` is an exact
          integer — otherwise the rounded trim leaves a fractional-sample group
          delay (here ~2 ms) that rotates every off-carrier phasor by ``2π·Δf·δt``,
          silently capping the subtraction of detuned modes at ~30 dB.

        * **Upsampling images** — the decimated phasor's spectral copies sit at
          ``±k·fs_dec`` and re-appear around ``fc ± k·fs_dec`` after remodulation.
          The default Kaiser window (``β=5``) only suppresses the first image to
          ≈ −73 dB.  Because the carried signal lives in ``[fc ± dfc]`` while the
          decimated Nyquist is ``2·dfc`` (a full octave of guard band), a steeper
          stopband is essentially free: ``β=12`` pushes the first image below
          −120 dB without touching in-band content.
        """
        from scipy.signal import resample_poly
        from math import gcd
        y = np.asarray(y)
        fs_old_int = int(round(float(fs_old) * tolerance))
        fs_new_int = int(round(float(fs_new) * tolerance))
        g = gcd(fs_old_int, fs_new_int)
        up = fs_new_int // g
        down = fs_old_int // g
        window = ("kaiser", float(kaiser_beta))

        n_in = len(y)
        if n_in == 0:
            return resample_poly(y, up, down, window=window)
        out_len = -(-n_in * up // down)                  # ceil(n_in*up/down), scipy's length
        # Transient ≈ 2×(10·max(up,down)/up) input samples; round the pad up to a
        # whole number of ``down``-blocks so the trim offset is an exact integer.
        base = int(np.ceil(10.0 * max(up, down) / up)) * 2 + 8
        reps = max(1, -(-base // down))                   # ceil(base/down), ≥ 1 block
        pad = reps * down
        yp = np.concatenate([np.full(pad, y[0], dtype=y.dtype), y,
                             np.full(pad, y[-1], dtype=y.dtype)])
        yr = resample_poly(yp, up, down, window=window)
        off = reps * up                                  # = pad·up/down, exact (zero delay)
        return yr[off:off + out_len]

    # --- real strain → heterodyne rotation → resample_poly downsample ---
    def _demod_preprocess(self, zk, uk=None, **kwargs):
        """Preprocess: real strain → complex baseband phasor at decimated rate.

        Heterodyne rotation ``ψ = 2·x·e^{-jωc·t}`` followed by anti-aliased
        downsampling to ``fs_dec = 4·dfc``.  Returns ``(N_dec, 2)`` = ``[Re(ψ), Im(ψ)]``.
        Symmetric counterpart: ``_reconstruct_postprocess``.
        """
        x = np.asarray(zk, dtype=float).reshape(np.shape(zk)[0], -1)[:, 0]   # real strain (T,)
        N = len(x)
        self._preprocess_N = N
        t = self.t0 + np.arange(N) / self._fs_original
        psi = 2.0 * x * np.exp(-1j * self._wc * t)
        psi_dec = self._resample(psi, self._fs_original, self._fs_dec)
        return np.column_stack([psi_dec.real, psi_dec.imag]), uk              # (N_dec, 2)

    # --- optional band-limit: kill the residual upsampling images ---
    def _band_limit_phasor(self, psi, cutoff_hz):
        """Zero-phase low-pass a complex baseband phasor at ``cutoff_hz``.

        Upsampling the decimated phasor leaves residual spectral images at
        ``±fs_dec`` baseband (i.e. ``fc ± fs_dec`` after remodulation, well
        outside the reconstruction band ``[fc ± 2·dfc]``).  Low-passing the
        upsampled phasor at ``cutoff_hz = 2·dfc`` removes them, confining the
        reconstruction to exactly ``[fc ± cutoff_hz]``.  In-band content is
        untouched (the carried envelope lives within ``±dfc``).
        """
        from scipy.signal import firwin, filtfilt
        psi = np.asarray(psi)
        n = len(psi)
        if n < 9 or cutoff_hz <= 0:
            return psi
        fnyq = self._fs_original / 2.0
        # Transition band ~ the (2·dfc → fs_dec) guard; size taps to resolve it.
        ntaps = int(round(3.3 * self._fs_original / cutoff_hz)) | 1
        ntaps = min(ntaps, ((n - 1) // 3) | 1)
        if ntaps < 3:
            return psi
        taps = firwin(ntaps, cutoff_hz / fnyq)
        return filtfilt(taps, 1.0, psi, padlen=min(3 * ntaps, n - 1))

    # --- postprocess: filtered phasor sum → real band signal at original rate ---
    def _reconstruct_postprocess(self, zk_hat, band_limit=False, **kwargs):
        """Postprocess: FUSED phasor sum (decimated) → real band signal at original rate.

        ``zk_hat`` = ``h(xk)`` from the FUSED composite = ``Σψ̂_i``, shape ``(N_dec, 2)``.
        Upsamples to ``fs_original`` then remodulates: ``Re[Σψ̂ · e^{jωc·t}]``.
        Symmetric counterpart: ``_demod_preprocess``.
        Returns ``(N, 1)`` real.

        ``band_limit`` (default ``False``) — when ``True``, low-pass the upsampled
        phasor at ``2·dfc`` to erase the residual ``fc ± fs_dec`` upsampling images.
        """
        z = np.asarray(zk_hat, dtype=float).reshape(len(zk_hat), -1)
        psi = z[:, 0] + 1j * z[:, 1]
        psi_up = self._resample(psi, self._fs_dec, self._fs_original)
        N = self._preprocess_N
        if len(psi_up) > N:
            psi_up = psi_up[:N]
        elif len(psi_up) < N:
            psi_up = np.concatenate([psi_up, np.zeros(N - len(psi_up))])
        if band_limit:
            psi_up = self._band_limit_phasor(psi_up, 2.0 * self.dfc)
        t = self.t0 + np.arange(N) / self._fs_original
        arg = self._wc * t
        return (psi_up.real * np.cos(arg) - psi_up.imag * np.sin(arg)).reshape(-1, 1)

    # --- identity postprocess (kept for backward compat / standalone use) ---
    @staticmethod
    def _identity_postprocess(zk_hat, **kwargs):
        """Identity postprocess: pass ``h(xk)`` through unchanged as ``[Re, Im]`` phasors."""
        return np.asarray(zk_hat, dtype=float).reshape(len(zk_hat), -1)

    def reconstruct(self, record, *, band_limit=False):
        """Upsample the decimated phasors in ``record.zk`` and remodulate to the
        original sample rate — returns a real 1-D array of length ``N_original``.

        ``band_limit`` (default ``False``) clips the output to ``[fc ± 2·dfc]``,
        removing the residual ``fc ± fs_dec`` upsampling images (~−125 dB)."""
        return self._reconstruct_postprocess(record.zk, band_limit=band_limit).ravel()

    def amplitudes(self, record):
        """Per-mode amplitude ``|ψ_i| = √(Re²+Im²)`` for the in-band modes — shape ``(T, n)``."""
        xk = np.asarray(record.xk)
        return np.column_stack([np.hypot(xk[:, 2 * k], xk[:, 2 * k + 1])
                                for k in range(len(self.frequencies))])

    # ------------------------------------------------------------------
    @classmethod
    def fit_inharmonicity(cls, peak_freqs, partial_numbers=None):
        """Least-squares fit of ``(f0, B)`` from measured partial frequencies."""
        peak_freqs = np.asarray(peak_freqs, dtype=float)
        ns = (np.arange(1, len(peak_freqs) + 1, dtype=float)
              if partial_numbers is None else np.asarray(partial_numbers, dtype=float))

        def model_fn(n, f0, B):
            return n * f0 * np.sqrt(1.0 + B * n ** 2)

        popt, _ = curve_fit(model_fn, ns, peak_freqs, p0=[peak_freqs[0], 0.0],
                            bounds=([0.0, 0.0], [np.inf, 1.0]), maxfev=10000)
        return float(popt[0]), float(popt[1])



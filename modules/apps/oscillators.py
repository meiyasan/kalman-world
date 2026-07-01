import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.signal import hilbert  # used by AnalyticOscillatorModel docstring examples

from ..framework.models.linear import LinearKalmanModel
from ..framework.models.composite import CompositeKalmanModel
from ..framework.constants import KalmanState, KalmanObservable, KalmanObservableReadout


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


# ======================================================================
# Single rotating-frame mode — the building block AND the standalone model
# ======================================================================
class AnalyticOscillatorModel(LinearKalmanModel):
    """
    A single oscillator tracked as a **complex envelope** ``ψ = Re + i·Im`` in a frame
    rotating at carrier ``fc`` [Hz].  On resonance (``fc = f0``) ψ is a constant (DC
    in the rotating frame — the lock-in / heterodyne limit); off resonance it precesses
    at the detuning ``Δf = f0 − fc``.

    The state is two real numbers ``[Re, Im]`` — the slowly-varying amplitude and phase
    of the mode.

    **Parameters (preferred — Hz):**
    - ``f0`` : resonance frequency [Hz]
    - ``fc`` : carrier / demodulation frequency [Hz]

    **Legacy parameters (rad/s) — still accepted for backward compat:**
    - ``w0`` : resonance angular frequency [rad/s]
    - ``wc`` : carrier angular frequency [rad/s]

    **Frames:** This is the rotating-frame atom.  The carrier rotation lives in the
    **signal** (``preprocess`` multiplies by ``e^{−iω_c t}``); ``H = identity``; and
    the frequency-offset precession sits in ``F = expm(Δω·dt)``.  Its lab-frame dual
    is :class:`PhasorOscillatorModel`, where the carrier lives in the time-varying
    ``H(t)`` and ``F = I``.

    This model is the atomic block the windowed :class:`OscillatorModel` stacks.  Used
    standalone it expects a 2-column analytic signal ``[real, imag]`` (e.g. from
    ``scipy.signal.hilbert``); inside an :class:`OscillatorModel` window the
    demodulation is done once at the window level.

    **pre/in/postprocess symmetry:** ``preprocess`` rotates the analytic input into the
    rotating frame; ``postprocess`` inverse-rotates back to the lab-frame analytic signal.
    This means ``sample().zk`` is in the same space as the model's input → you can call
    ``model.filter(sample.zk)`` directly.
    """

    NAME = "Analytic Oscillator (rotating-frame envelope)"
    DESCRIPTION = "Single oscillator as a complex envelope [Re, Im] in a frame rotating at fc."
    VERSION = "2.1"

    @property
    def states(self):
        return OscillatorState
    @property
    def observables(self):
        return OscillatorObservable

    def __init__(self, name, **kwargs):
        measurement_uncertainty = kwargs.pop("measurement_uncertainty", 0.5)
        process_uncertainty = kwargs.pop("process_uncertainty", 0.05)

        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or self.fs
        self.measurement_uncertainty = measurement_uncertainty
        self.process_uncertainty = process_uncertainty

        dt = 1.0 / self.fs

        # Accept f0/fc [Hz] as primary parameters; fall through to w0/wc [rad/s]
        # for backward compatibility with existing code that passes w0/wc.
        f0_hz = kwargs.pop("f0", None)
        fc_hz = kwargs.pop("fc", None)
        w0_rad = kwargs.pop("w0", None)
        wc_rad = kwargs.pop("wc", None)

        if f0_hz is not None:
            w0 = 2.0 * np.pi * float(f0_hz)
        elif w0_rad is not None:
            w0 = float(w0_rad)
        else:
            w0 = 2.0 * np.pi          # default 1 Hz

        if fc_hz is not None:
            wc = 2.0 * np.pi * float(fc_hz)
        elif wc_rad is not None:
            wc = float(wc_rad)
        else:
            wc = w0                    # default: carrier on resonance (lock-in)

        self["f0"] = w0 / (2.0 * np.pi)
        self["fc"] = wc / (2.0 * np.pi)
        self["w0"] = w0                # kept for internal / legacy access
        self["wc"] = wc

        dw = w0 - wc                   # detuning Δω [rad/s]
        Q = kwargs.pop("Q", None)

        # F = expm(A·dt),  A = [[−γ, −Δω], [Δω, −γ]],  γ = w0/(2Q)
        gamma = w0 / (2.0 * Q) if Q else 0.0
        A = expm(np.array([[-gamma * dt, -dw * dt],
                           [   dw * dt, -gamma * dt]]))
        C = np.identity(2)
        R = measurement_uncertainty * np.sqrt(2) * np.identity(2)
        Qn = process_uncertainty * np.sqrt(2) * np.identity(2)

        x0, P0 = kwargs.pop("x0", None), kwargs.pop("P0", None)
        super().__init__(name, F=A, H=C, Q=Qn, R=R, P0=P0, x0=x0, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        """Analytic signal [real, imag] → rotate into the rotating frame at wc."""
        zr, zi = zk[:, 0], zk[:, 1]
        times = self.t0 + np.arange(len(zk)) / self.fs
        arg = self["wc"] * times
        zk_rot = np.column_stack([zr * np.cos(arg) + zi * np.sin(arg),
                                  zi * np.cos(arg) - zr * np.sin(arg)])
        return zk_rot, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        """Inverse-rotate back to the lab-frame analytic signal."""
        zr, zi = zk_hat[:, 0], zk_hat[:, 1]
        times = self.t0 + np.arange(len(zk_hat)) / self.fs
        arg = self["wc"] * times
        return np.column_stack([zr * np.cos(arg) - zi * np.sin(arg),
                                zi * np.cos(arg) + zr * np.sin(arg)])

    # ==================================================================
    # Signature figure — the envelope ψ in the rotating frame.
    # ==================================================================
    def _rotframe_sample(self, sample):
        """Return a copy of ``sample`` with zk preprocessed into the rotating frame."""
        if sample is None:
            return None
        from ..framework.io.record import KalmanRecord
        zk_rot = self.preprocess(np.asarray(sample.zk))[0]
        return KalmanRecord(sample.time, zk=zk_rot, xk=sample.xk, Pk=sample.Pk,
                            flavor=sample.flavor, model=sample.model)

    def _plot_interactive(self, record, sample=None, *, component, error=True, **kwargs):
        return super()._plot_interactive(
            record, sample=self._rotframe_sample(sample),
            component=component, error=error, **kwargs)

    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(15, 4), **kwargs):
        import matplotlib.pyplot as plt
        t  = np.asarray(record.time)
        xk = np.asarray(record.xk)
        zk = self.preprocess(np.asarray(sample.zk))[0] if sample is not None else None
        xt = np.asarray(sample.xk) if sample is not None else None

        fig, (ax_re, ax_im, ax_ph) = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"{self.name()} — envelope ψ (rotating frame)", fontweight="bold")
        for ax, j, col_est, col_aux, ylab in (
            (ax_re, 0, "navy", "steelblue", r"Re($\psi$)"),
            (ax_im, 1, "saddlebrown", "darkorange", r"Im($\psi$)"),
        ):
            if zk is not None:
                ax.scatter(t, zk[:, j], s=1, alpha=0.3, color=col_aux, label="Measured (demod)")
            if xt is not None:
                ax.plot(t, xt[:, j], "--", lw=1.5, color=col_aux, label="Truth")
            ax.plot(t, xk[:, j], lw=2, color=col_est, label="Estimate")
            ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab); ax.set_title(ylab)
            ax.legend(fontsize=8, markerscale=4)
        if xt is not None:
            ax_ph.plot(xt[:, 0], xt[:, 1], "--", lw=1.5, color="steelblue", label="Truth")
        ax_ph.plot(xk[:, 0], xk[:, 1], lw=2, color="navy", label="Estimate")
        ax_ph.plot(xk[0, 0], xk[0, 1], "o", ms=6, color="green", label="Start", zorder=5)
        ax_ph.plot(xk[-1, 0], xk[-1, 1], "s", ms=6, color="red", label="End", zorder=5)
        ax_ph.set_xlabel(r"Re($\psi$)"); ax_ph.set_ylabel(r"Im($\psi$)"); ax_ph.set_title("Phase space")
        ax_ph.set_aspect("equal"); ax_ph.axhline(0, color="gray", lw=0.5); ax_ph.axvline(0, color="gray", lw=0.5)
        ax_ph.legend(fontsize=8)
        fig.tight_layout()
        return fig


# ======================================================================
# Lab-frame phasor tracker — the dual of AnalyticOscillatorModel
# ======================================================================
class PhasorState(KalmanState):
    REAL = "re"   # lab-frame phasor — real (in-phase) component
    IMAG = "im"   # lab-frame phasor — imaginary (quadrature) component


class PhasorObservable(KalmanObservable):
    SIGNAL = ""   # the raw real measured signal (single scalar channel)


class PhasorOscillatorModel(LinearKalmanModel):
    r"""
    Track a set of sinusoids as **lab-frame phasors**, straight from a **real** signal —
    no demodulation, no Hilbert transform.

    This is the *dual* of :class:`AnalyticOscillatorModel`:

    * **Rotating frame** (`AnalyticOscillatorModel` / `OscillatorModel`): the carrier
      rotation lives in the **signal** — we demodulate by multiplying ``z(t)`` by
      ``e^{−iω_c t}`` so the mode appears as a slowly-varying envelope.  ``F = expm(Δω)``,
      ``H = identity``.  Right for **weak, narrowband** lines (KAGRA violin modes).

    * **Lab frame** (`PhasorOscillatorModel`): the carrier rotation lives in the
      **observation map** instead.  The phasors ``[Re_k, Im_k]`` are constant (``F = I``)
      and the single real measurement is reconstructed from all of them at once through a
      time-varying row ``H(t) = [cos ω₁t, −sin ω₁t, cos ω₂t, −sin ω₂t, …]``.  No
      Hilbert, causal/streaming.  Because every mode must together explain the one
      measurement, a strong mode cannot leak into an absent neighbour.  Right for
      **strong, wide-spread** lines (mains harmonics 50→250 Hz, or any wide-band set).

    State layout (one phasor per frequency, scalar observation)::

        x = [Re₁, Im₁,  Re₂, Im₂,  …,  Re_N, Im_N]      (2N states)
        z = [v(t)]                                          (1 observation)

    ``F = I`` (phasors constant; ``process_uncertainty`` allows slow drift);
    ``R = σ_meas²`` is a single channel.

    Parameters
    ----------
    sensor_id : str
    frequencies : float | array-like
        Line frequencies [Hz] to track jointly (one ``[Re, Im]`` phasor each).
    fs : float
    measurement_uncertainty : float
        Measurement-noise std (sets ``R = σ²``).
    process_uncertainty : float
        Per-phasor random-walk std (how fast the amplitudes may drift).
    p0_std : float
        Initial per-phasor prior std — large vs. the measurement noise for acquisition.
    """

    NAME = "Phasor Oscillator (lab-frame, real signal)"
    DESCRIPTION = ("Joint lab-frame phasor tracker: real scalar in, time-varying H(t), "
                   "F = I. All modes coupled through one measurement (no demodulation).")
    VERSION = "1.0"

    @property
    def states(self):
        return PhasorState

    @property
    def observables(self):
        return PhasorObservable

    def __init__(self, sensor_id, frequencies, *, fs=4096.0,
                 measurement_uncertainty=1.0, process_uncertainty=0.01,
                 p0_std=100.0, **kwargs):

        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if freqs.size == 0:
            raise ValueError("PhasorOscillatorModel needs at least one frequency.")
        self.frequencies = freqs
        self._omega = 2.0 * np.pi * freqs                       # (N,) carrier angular freqs
        self.measurement_uncertainty = float(measurement_uncertainty)
        self.process_uncertainty = float(process_uncertainty)
        N = len(freqs)

        F = np.eye(2 * N)                                        # phasors constant in lab frame
        Q = (process_uncertainty ** 2) * np.eye(2 * N)          # slow phasor random-walk
        R = np.array([[measurement_uncertainty ** 2]])          # scalar measurement noise

        x0 = kwargs.pop("x0", None)
        if x0 is None:
            x0 = np.zeros((2 * N, 1))
        P0 = kwargs.pop("P0", None)
        if P0 is None:
            P0 = (p0_std ** 2) * np.eye(2 * N)                  # generous prior for acquisition

        super().__init__(sensor_id, F=F, H=self._observation_row, Q=Q, R=R,
                         fs=fs, x0=x0, P0=P0, **kwargs)

    # ------------------------------------------------------------------
    # Time-varying observation: the real scalar as the sum of all phasors.
    # H(t) = [cos ω₁t, −sin ω₁t,  cos ω₂t, −sin ω₂t,  …]   shape (1, 2N)
    # ------------------------------------------------------------------
    def _observation_row(self, t=None):
        t = 0.0 if t is None else float(t)
        row = np.zeros((1, 2 * len(self.frequencies)))
        row[0, 0::2] = np.cos(self._omega * t)
        row[0, 1::2] = -np.sin(self._omega * t)
        return row

    def preprocess(self, zk, uk=None, **kwargs):
        """Use the raw scalar signal. A 2-column analytic ``[real, imag]`` input is
        accepted for convenience — only the real (measured) column is kept."""
        zk = np.asarray(zk)
        if zk.ndim == 1:
            zk = zk[:, None]
        return zk[:, :1], uk

    # ------------------------------------------------------------------
    def amplitudes(self, record):
        """Per-mode amplitude ``a_k = |ψ_k| = √(Re²+Im²)`` — shape ``(T, N)``."""
        xk = np.asarray(record.xk)
        return np.column_stack([np.hypot(xk[:, 2 * k], xk[:, 2 * k + 1])
                                for k in range(len(self.frequencies))])

    def envelope(self, record):
        """Per-mode amplitude and its ±1σ band from the phasor state ``record.xk`` and
        covariance ``record.Pk``. Returns ``(amp, sigma)``, each shape ``(T, N)``."""
        xk = np.asarray(record.xk)
        Pk = np.asarray(record.Pk)
        T, N = len(xk), len(self.frequencies)
        amp = np.zeros((T, N))
        sigma = np.zeros((T, N))
        has_P = Pk.size > 0
        for k in range(N):
            re, im = xk[:, 2 * k], xk[:, 2 * k + 1]
            a = np.hypot(re, im)
            amp[:, k] = a
            if has_P:
                P_RR, P_II = Pk[:, 2 * k, 2 * k], Pk[:, 2 * k + 1, 2 * k + 1]
                with np.errstate(divide="ignore", invalid="ignore"):
                    sigma[:, k] = np.where(
                        a > 1e-3,
                        np.sqrt((re ** 2 * P_RR + im ** 2 * P_II) / a ** 2), 0.0)
        return amp, sigma

    def reconstruct(self, record, t=None):
        """Reconstruct the real signal from the phasor state — the model's own observation
        map: ``z(t) = Σ_k Re_k·cos(ω_k t) − Im_k·sin(ω_k t)``."""
        xk = np.asarray(record.xk)
        t = np.asarray(record.time if t is None else t)
        v = np.zeros(len(t))
        for k in range(len(self.frequencies)):
            w_k = self._omega[k]
            v += xk[:, 2 * k] * np.cos(w_k * t) - xk[:, 2 * k + 1] * np.sin(w_k * t)
        return v


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


# ======================================================================
# A bank of windows over one real signal — INDEPENDENT across distinct fc
# ======================================================================
class OscillatorBankModel(CompositeKalmanModel):
    """
    A bank of :class:`OscillatorModel` windows analysing **one** real signal across
    several distinct centers ``fc_i``.

    Internally this is a :class:`CompositeKalmanModel` with ``obs_mode=INDEPENDENT``:
    each window tracks its own demodulated band and the observations are stacked.
    The standard :class:`CompositeKalmanModel` preprocess would try to split the input
    by channel count, which cannot broadcast one real signal to N demodulators.  This
    class overrides preprocess with :meth:`_broadcast_preprocess` to send the single
    real strain to every window's own Hilbert demodulator.

    Composition pattern::

        OscillatorBankModel(w1, w2, w3)
          = CompositeKalmanModel(w1, w2, w3, obs_mode=INDEPENDENT)
          + preprocess: broadcast real strain → each window's own _demod_preprocess

    Each window's postprocess remodulates to a real band, so ``record.zk`` has one
    real column per window.  ``reconstruct(record)`` sums them to give the total
    extracted signal; ``strain − reconstruct(record)`` is the cleaned residual.
    """

    NAME = "Oscillator Bank (window bank)"
    DESCRIPTION = "INDEPENDENT bank of windowed OscillatorModels over one real signal."
    VERSION = "1.1"

    def __init__(self, *windows, **kwargs):
        if not windows:
            raise ValueError("OscillatorBankModel needs at least one window.")
        self.windows = list(windows)
        super().__init__(*windows, obs_mode=KalmanObservableReadout.INDEPENDENT, **kwargs)
        self.fs = windows[0].fs
        self._fs_original = windows[0]._fs_original
        self.preprocess = self._broadcast_preprocess   # expand: 1 real → N basebands
        self.postprocess = self._identity_postprocess  # keep decimated phasors; reconstruct() upsamples

    @staticmethod
    def _identity_postprocess(zk_hat, **kwargs):
        return np.asarray(zk_hat, dtype=float).reshape(len(zk_hat), -1)

    def _broadcast_preprocess(self, zk, uk=None, **kwargs):
        """Preprocess: broadcast one real strain to every window, each demodulates + downsamples.

        Returns ``(N_dec, 2·N_win)`` — columns ``[Re_ψ1, Im_ψ1, Re_ψ2, Im_ψ2, …]``.
        Symmetric counterpart: ``_reconstruct_postprocess``.
        """
        x = np.asarray(zk, dtype=float).reshape(np.shape(zk)[0], -1)[:, :1]   # (T, 1) real strain
        outs = [w._demod_preprocess(x)[0] for w in self.windows]               # each (N_dec, 2)
        return np.concatenate(outs, axis=1), uk                                 # (N_dec, 2·N_win)

    def _reconstruct_postprocess(self, zk_hat, **kwargs):
        """Postprocess: N window phasors (decimated) → one real signal at original rate.

        ``zk_hat`` = ``h(xk)`` from the INDEPENDENT composite, shape ``(N_dec, 2·N_win)``.
        Columns ``2i, 2i+1`` are ``[Re(Σψ̂_win_i), Im(Σψ̂_win_i)]`` — already the FUSED
        phasor sum within window ``i`` (computed by each window's FUSED ``h``).
        Each window's phasor is upsampled to ``fs_original``, remodulated at ``fc_i``,
        and summed across windows.  Returns ``(N, 1)`` real.
        Symmetric counterpart: ``_broadcast_preprocess``.
        """
        z = np.asarray(zk_hat, dtype=float).reshape(len(zk_hat), -1)
        w0 = self.windows[0]
        N = w0._preprocess_N
        t = w0.t0 + np.arange(N) / self._fs_original
        total = np.zeros(N)
        for i, w in enumerate(self.windows):
            psi = z[:, 2 * i] + 1j * z[:, 2 * i + 1]
            psi_up = w._resample(psi, w._fs_dec, self._fs_original)
            if len(psi_up) > N:
                psi_up = psi_up[:N]
            elif len(psi_up) < N:
                psi_up = np.concatenate([psi_up, np.zeros(N - len(psi_up))])
            total += psi_up.real * np.cos(w._wc * t) - psi_up.imag * np.sin(w._wc * t)
        return total.reshape(-1, 1)

    def reconstruct(self, record, *, per_window=False, band_limit=False):
        """Upsample per-window decimated phasors from ``record.zk`` and sum to a
        single real signal at the original sample rate — shape ``(N_original,)``.

        Parameters
        ----------
        per_window : bool, default False
            If ``True``: return a list of per-window ``(N_original,)`` arrays without
            re-filtering (delegates to :meth:`reconstruct_window`).
        band_limit : bool, default False
            If ``True``: clip each window's output to its ``[fc ± 2·dfc]`` band,
            removing the residual ``fc ± fs_dec`` upsampling images.
        """
        if per_window:
            return [self.reconstruct_window(record, i, band_limit=band_limit)
                    for i in range(len(self.windows))]
        z = np.asarray(record.zk, dtype=float).reshape(len(record.zk), -1)
        w0 = self.windows[0]
        N = w0._preprocess_N
        t = w0.t0 + np.arange(N) / self._fs_original
        total = np.zeros(N)
        for i, w in enumerate(self.windows):
            psi = z[:, 2 * i] + 1j * z[:, 2 * i + 1]
            psi_up = w._resample(psi, w._fs_dec, self._fs_original)
            if len(psi_up) > N:
                psi_up = psi_up[:N]
            elif len(psi_up) < N:
                psi_up = np.concatenate([psi_up, np.zeros(N - len(psi_up))])
            if band_limit:
                psi_up = w._band_limit_phasor(psi_up, 2.0 * w.dfc)
            total += psi_up.real * np.cos(w._wc * t) - psi_up.imag * np.sin(w._wc * t)
        return total

    def reconstruct_window(self, record, i, *, band_limit=False):
        """Return the real-band reconstruction for window ``i`` without re-filtering.

        Slices window ``i``'s state from ``record.xk``, computes the FUSED phasor sum,
        upsamples, and remodulates.  Does not touch the other windows.
        """
        offset = sum(w.n for w in self.windows[:i])
        n = self.windows[i].n
        xk_w = np.asarray(record.xk, dtype=float)[:, offset:offset + n]
        zk_hat_w = np.column_stack([xk_w[:, 0::2].sum(axis=1),
                                    xk_w[:, 1::2].sum(axis=1)])
        return self.windows[i]._reconstruct_postprocess(zk_hat_w, band_limit=band_limit).ravel()


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


# Backward-compatible alias: the bank was renamed OscillatorCompositeModel → OscillatorBankModel.
OscillatorCompositeModel = OscillatorBankModel

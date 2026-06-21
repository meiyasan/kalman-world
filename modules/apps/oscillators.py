import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit
from scipy.signal import hilbert

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
# Single rotating-frame mode — the building block AND the standalone model
# ======================================================================
class AnalyticOscillatorModel(LinearKalmanModel):
    """
    A single oscillator tracked as a **complex envelope** ``ψ = Re + i·Im`` in a frame
    rotating at carrier ``wc``. On resonance (``wc = w0``) ψ is a constant; off resonance
    it precesses at the detuning ``Δω = w0 − wc``. The state is two real numbers
    ``[Re, Im]`` — the slowly-varying amplitude/phase of the mode.

    This is the atomic block the windowed :class:`OscillatorModel` stacks, and a useful
    standalone illustration of the rotating-frame (lock-in / heterodyne) representation.
    Used standalone it expects a 2-column analytic signal ``[real, imag]`` (e.g. from
    ``scipy.signal.hilbert``); inside an :class:`OscillatorModel` window the demodulation
    is done once at the window level instead.
    """

    NAME = "Analytic Oscillator (rotating-frame envelope)"
    DESCRIPTION = "Single oscillator as a complex envelope [Re, Im] in a frame rotating at wc."
    VERSION = "2.0"

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
        w0 = self["w0"] = kwargs.get("w0", 1)
        wc = self["wc"] = kwargs.get("wc", 1)
        dw = w0 - wc                                   # detuning Δω
        Q = kwargs.pop("Q", None)                      # quality factor (not the Q matrix)

        # F = expm(A·dt), A = [[−γ, −Δω], [Δω, −γ]], γ = w0/(2Q)  (slowly-varying envelope)
        A = expm(np.array([[-w0 / (2 * Q) * dt, -dw * dt],
                           [          dw * dt, -w0 / (2 * Q) * dt]]))
        C = np.identity(2)                                            # observe [Re, Im]
        R = measurement_uncertainty * np.sqrt(2) * np.identity(2)
        Q = process_uncertainty * np.sqrt(2) * np.identity(2)

        x0, P0 = kwargs.pop("x0", None), kwargs.pop("P0", None)
        super().__init__(name, F=A, H=C, Q=Q, R=R, P0=P0, x0=x0, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        # zk is (N, 2) — real and imaginary parts → rotate into the carrier frame
        zr, zi = zk[:, 0], zk[:, 1]
        times = self.t0 + np.arange(len(zk)) / self.fs
        arg = self["wc"] * times
        zk_rot = np.column_stack([zr * np.cos(arg) + zi * np.sin(arg),
                                  zi * np.cos(arg) - zr * np.sin(arg)])
        return zk_rot, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        # inverse rotation back to the lab frame
        zr, zi = zk_hat[:, 0], zk_hat[:, 1]
        times = self.t0 + np.arange(len(zk_hat)) / self.fs
        arg = self["wc"] * times
        return np.column_stack([zr * np.cos(arg) - zi * np.sin(arg),
                                zi * np.cos(arg) + zr * np.sin(arg)])

    # ==================================================================
    # Signature figure — the envelope ψ in the rotating frame.
    # ==================================================================
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
                ax.plot(t, xt[:, j], "--", lw=1.5, color=col_aux, label="True")
            ax.plot(t, xk[:, j], lw=2, color=col_est, label="Estimate")
            ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab); ax.set_title(ylab)
            ax.legend(fontsize=8, markerscale=4)
        if xt is not None:
            ax_ph.plot(xt[:, 0], xt[:, 1], "--", lw=1.5, color="steelblue", label="True")
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
    no demodulation, no Hilbert transform. This is the *dual* of
    :class:`AnalyticOscillatorModel`: same ``[Re, Im]`` phasor per mode, but the carrier
    rotation lives in the **observation map** instead of in the signal.

    The state holds one constant phasor per frequency::

        x = [Re_1, Im_1,  Re_2, Im_2,  …,  Re_N, Im_N]      (2N states)

    and the single scalar measurement is reconstructed from **all** phasors at once
    through a **time-varying** observation row::

        z(t) = Σ_k  Re_k·cos(ω_k t) − Im_k·sin(ω_k t),   ω_k = 2π·frequencies[k]
        H(t) = [cos ω_1 t, −sin ω_1 t,  cos ω_2 t, −sin ω_2 t,  …]      shape (1, 2N)

    ``F = I`` — the phasors are constant in the lab frame (``process_uncertainty`` sets a
    slow random-walk drift); ``R = σ_meas²`` is a single channel.

    **Why this is its own model (vs. the rotating-frame oscillators):**

    * It is fed a **real** scalar and never demodulates — the rotation is carried by
      ``H(t)``, applied at each mode's *true* frequency. Causal / streaming, no Hilbert.
    * Because the filter must explain the one measurement with the **whole** phasor set,
      every strong line is fully owned by its own state and cannot leak into a neighbour.
      An absent mode therefore reads ≈ 0 — the joint coupling that fixes the
      even-harmonic leakage of demodulate-then-low-pass schemes.
    * It is the right tool when the lines are **strong** and **spread across a wide band**
      (e.g. mains harmonics 50→250 Hz), where a single demodulation carrier cannot cover
      them and there is nothing to gain from decimation.

    Contrast with :class:`OscillatorModel` (rotating frame): there the *real* signal is
    demodulated inside ``preprocess`` (Hilbert → baseband), the state is the slow envelope,
    ``H = identity`` and the rotation sits in ``F = expm(detuning)``. That path can decimate
    and is the right tool for **weak, narrowband** lines (noise-limited violin modes).

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
# Inharmonic frequency helpers (shared)
# ======================================================================
def partial_frequencies(f0: float, B: float, n_harmonics: int) -> np.ndarray:
    """Fletcher (in)harmonic partials: ``f_n = n·f0·√(1 + B·n²)`` (``B=0`` → harmonic)."""
    n = np.arange(1, n_harmonics + 1, dtype=float)
    return n * f0 * np.sqrt(1.0 + B * n ** 2)


# ======================================================================
# One analysis window  [fc ± dfc]  — the atomic windowed model
# ======================================================================
class OscillatorModel(CompositeKalmanModel):
    """
    One analysis window ``[fc ± dfc]`` over a **real** signal.

    ``frequencies`` is the full candidate mode list (already expanded over harmonics /
    inharmonicity). The window acts as a **mask**: only modes with ``|f − fc| ≤ dfc`` are
    kept and tracked; the rest are dropped from the state. The in-band modes share the
    single demodulated band channel, so they are combined in **FUSED** mode
    (``H = [I₂ … I₂]``, ``z = Σ_i ψ_i``) — the joint Kalman filter of Moisset et al. (2025)
    / Mukherjee & Finn (1999). A single in-band mode is the trivial case.

    Real signal in (`preprocess` demodulates it at ``fc``), real band reconstruction out
    (`postprocess`). Inherits :class:`CompositeKalmanModel` — the window *is* the composite.

    Parameters
    ----------
    sensor_id : str
    frequencies : float | array-like
        Candidate mode frequencies [Hz]; masked to those inside ``[fc ± dfc]``.
    fc : float
        Window center frequency [Hz] (the demodulation carrier).
    dfc : float
        Window half-width [Hz].
    Q : float | list
        Quality factor (scalar, or one per kept mode).
    fs : float
    measurement_uncertainty, process_uncertainty : float
    """

    NAME = "Windowed Oscillator (one fc ± dfc band)"
    DESCRIPTION = ("One demodulated analysis window; in-band modes combined FUSED "
                   "(z = Σ ψ_i). Real in, real band reconstruction out.")
    VERSION = "3.0"

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

        # Acquisition-friendly initial covariance: large vs the measurement noise so the
        # un-seeded filter locks onto the demodulated amplitude (scale-invariant via R).
        p0_var = float(p0) if p0 is not None else 1e3 * measurement_uncertainty * np.sqrt(2)
        blocks = [AnalyticOscillatorModel(
                      f"{sensor_id}@{f:.3f}Hz", fs=fs,
                      w0=2 * np.pi * f, wc=2 * np.pi * fc, Q=q,
                      measurement_uncertainty=measurement_uncertainty,
                      process_uncertainty=process_uncertainty,
                      P0=p0_var * np.eye(2))
                  for f, q in zip(in_band, Qs)]

        super().__init__(*blocks, obs_mode=KalmanObservableReadout.FUSED, **kwargs)
        self.fs = float(fs)
        # Window-level real→complex demodulation (replaces the per-submodel split);
        # postprocess is identity so the record keeps the [ΣRe, ΣIm] band estimate.
        self.preprocess = self._demod_preprocess
        self.postprocess = lambda zk_hat, **kw: zk_hat

    # --- real strain → demodulated [Re, Im] band channel ---
    def _demod_preprocess(self, zk, uk=None, **kwargs):
        x = np.asarray(zk, dtype=float).reshape(np.shape(zk)[0], -1)[:, 0]   # real strain (T,)
        za = hilbert(x)                                                       # analytic signal
        t = self.t0 + np.arange(len(x)) / self.fs
        psi = za * np.exp(-1j * self._wc * t)                            # demodulate to band
        return np.column_stack([psi.real, psi.imag]), uk                     # (T, 2)

    # --- [ΣRe, ΣIm] band estimate (record.zk) → real band reconstruction (T,) ---
    def reconstruct(self, record):
        """Real-valued band reconstruction ``Re[ψ·e^{iω_c t}]`` from the filtered record."""
        z = np.asarray(record.zk, dtype=float).reshape(np.shape(record.zk)[0], -1)
        t = np.asarray(record.time)
        arg = self._wc * t
        return z[:, 0] * np.cos(arg) - z[:, 1] * np.sin(arg)

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
class OscillatorCompositeModel(CompositeKalmanModel):
    """
    A bank of :class:`OscillatorModel` windows analysing **one** real signal across
    several distinct centers ``fc_i``. The windows are combined **INDEPENDENT** (each its
    own demodulated band), and the single input strain is **broadcast** to every window
    (each demodulates it at its own ``fc``).

    ``record.zk`` holds one real band reconstruction per window; their **sum** is the total
    extracted signal and ``strain − sum`` is the cleaned residual.
    """

    NAME = "Oscillator Composite (window bank)"
    DESCRIPTION = "INDEPENDENT bank of windowed OscillatorModels over one real signal."
    VERSION = "1.0"

    def __init__(self, *windows, **kwargs):
        if not windows:
            raise ValueError("OscillatorCompositeModel needs at least one window.")
        self.windows = list(windows)
        super().__init__(*windows, obs_mode=KalmanObservableReadout.INDEPENDENT, **kwargs)
        self.fs = windows[0].fs
        self.preprocess = self._broadcast_preprocess

    def _broadcast_preprocess(self, zk, uk=None, **kwargs):
        """Broadcast the one real strain to every window; each demodulates at its own fc."""
        x = np.asarray(zk, dtype=float).reshape(np.shape(zk)[0], -1)[:, :1]   # (T,1) real strain
        outs = [w._demod_preprocess(x)[0] for w in self.windows]              # each (T,2)
        return np.concatenate(outs, axis=1), uk                              # (T, 2·N_win)

    def reconstruct(self, record):
        """Sum of every window's real band reconstruction — shape ``(T,)``."""
        z = np.asarray(record.zk, dtype=float).reshape(np.shape(record.zk)[0], -1)
        t = np.asarray(record.time)
        offsets = np.cumsum([0] + [w.m for w in self.windows])
        total = np.zeros(len(t))
        for k, w in enumerate(self.windows):
            zk = z[:, offsets[k]:offsets[k + 1]]                 # window k's [ΣRe, ΣIm]
            arg = w._wc * t
            total += zk[:, 0] * np.cos(arg) - zk[:, 1] * np.sin(arg)
        return total


# ======================================================================
# (In)harmonic ladders — an OscillatorModel window whose candidate list is a ladder
# ======================================================================
class InharmonicOscillatorModel(OscillatorModel):
    """
    One :class:`OscillatorModel` window whose candidate frequency list is an **inharmonic
    ladder**: ``f_n = n·f0·√(1+B·n²)`` for ``n = 1 … n_harmonics``, generated from
    ``(f0, B)``. As any window it masks to ``[fc ± dfc]``. To track a whole ladder, place
    one per partial and compose them in an :class:`OscillatorCompositeModel`.
    """

    NAME = "Inharmonic Oscillator (windowed)"
    DESCRIPTION = "Windowed oscillator over an inharmonic ladder f_n = n·f0·√(1+B·n²)."
    VERSION = "3.0"

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
    """Perfectly harmonic ladder — :class:`InharmonicOscillatorModel` with ``B = 0``
    (partials at exact integer multiples ``f_n = n·f0``)."""

    NAME = "Harmonic Oscillator (windowed, B=0)"
    DESCRIPTION = "Windowed oscillator over a harmonic ladder f_n = n·f0."
    VERSION = "3.0"

    def __init__(self, sensor_id, f0, fc, dfc, *, n_harmonics=3, **kwargs):
        if kwargs.pop("B", 0.0):
            raise TypeError("HarmonicOscillatorModel forces B=0; use InharmonicOscillatorModel for B>0.")
        super().__init__(sensor_id, f0, fc, dfc, n_harmonics=n_harmonics, B=0.0, **kwargs)

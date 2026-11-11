import numpy as np

from ....framework.constants import KalmanObservable
from ..oscillators import partial_frequencies, PhasorOscillatorModel


def plot_phasor_diagram(f0, phasors, *, colors=None, frames=30, interval=60, figsize=(14, 4)):
    """Animated phasor diagram for power-grid odd harmonics.

    Three synchronized panels: lab-frame phasors rotating at each harmonic's own
    angular frequency, the individual harmonic waveforms, and their composite vs the
    ideal (fundamental-only) waveform — tied together by a moving time cursor.

    Parameters
    ----------
    f0 : float
        Fundamental frequency [Hz].
    phasors : list of (V, phi)
        Amplitude [V] and phase [rad] for H1, H3, H5, … (odd harmonics, in order) —
        half-wave-symmetric grid harmonics.  ``phasors[0]`` is the fundamental H1.
    colors : list of str, optional
        Per-harmonic colors.  Default: matplotlib tab10 cycle.
    frames, interval : int
        Animation frame count and interval [ms] over one fundamental period.

    Returns
    -------
    Jupyter HTML player (or raw ``FuncAnimation`` outside Jupyter) — see ``helpers.animate``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from ....helpers import animate

    n = len(phasors)
    if colors is None:
        cmap = plt.cm.tab10
        colors = [cmap(k) for k in range(n)]
    labels_short = [f"H{2 * k + 1}" for k in range(n)]
    labels = [f"{lbl} ({'f₀' if k == 0 else f'{2 * k + 1}f₀'})" for k, lbl in enumerate(labels_short)]
    omegas = [2 * np.pi * (2 * k + 1) * f0 for k in range(n)]

    T = 1.0 / f0
    t_cyc = np.linspace(0, T, 500)
    harmonic_waves = [V * np.cos(omegas[k] * t_cyc + phi) for k, (V, phi) in enumerate(phasors)]
    v_clean = sum(harmonic_waves)
    v_ideal = phasors[0][0] * np.cos(omegas[0] * t_cyc)

    frame_t = np.linspace(0, T, frames, endpoint=False)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("Power grid phasors — amplitude, phase, and time-domain reconstruction", fontweight="bold")

    # Phasor diagram — arrows rotate at each harmonic's own angular frequency. Axis is
    # square and sized to the largest amplitude in BOTH dimensions: a rotating phasor
    # swings through the full +/-amplitude range on y too, not just on x.
    ax0 = axes[0]
    ax0.set_aspect("equal")
    Vmax = max(V for V, _ in phasors)
    ax0.set_xlim(-Vmax * 1.15, Vmax * 1.15); ax0.set_ylim(-Vmax * 1.15, Vmax * 1.15)
    ax0.axhline(0, color="gray", lw=0.5); ax0.axvline(0, color="gray", lw=0.5)
    ax0.set_xlabel("Re [V]"); ax0.set_ylabel("Im [V]")
    ax0.set_title("Phasor diagram (rotating, lab frame)")
    arrows = []
    for (V, phi), col in zip(phasors, colors):
        arrow = FancyArrowPatch((0, 0), (V * np.cos(phi), V * np.sin(phi)),
                                color=col, arrowstyle="-|>", mutation_scale=14, lw=2.2)
        ax0.add_patch(arrow)
        arrows.append(arrow)
    # Fixed-position readout instead of labels at each arrow tip: harmonic amplitudes
    # vary by orders of magnitude, so tip-anchored labels collide near the origin.
    info_text = ax0.text(0.02, 0.98, "", transform=ax0.transAxes, va="top", ha="left",
                         fontsize=7.5, family="monospace",
                         bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", pad=3))

    # Individual harmonic waveforms — static curves + moving cursor dots
    ax1 = axes[1]
    for wave, lbl, col in zip(harmonic_waves, labels, colors):
        ax1.plot(t_cyc * 1e3, wave, color=col, lw=1.5, label=lbl)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_xlabel("t [ms]"); ax1.set_ylabel("voltage [V]")
    ax1.set_title("Individual harmonics")
    ax1.legend(fontsize=8)
    cursor1 = ax1.axvline(0, color="black", lw=0.8, ls=":")
    dots1 = [ax1.plot([], [], "o", color=col, ms=5)[0] for col in colors]

    # Composite waveform — static curves + moving cursor dot
    ax2 = axes[2]
    ax2.plot(t_cyc * 1e3, v_clean, color="purple", lw=2, label=" + ".join(labels_short))
    ax2.plot(t_cyc * 1e3, v_ideal, "--", color="steelblue", lw=1, label="H1 only (ideal)")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("t [ms]"); ax2.set_ylabel("voltage [V]")
    ax2.set_title("Composite: distorted vs ideal")
    ax2.legend(fontsize=8)
    cursor2 = ax2.axvline(0, color="black", lw=0.8, ls=":")
    dot2 = ax2.plot([], [], "o", color="purple", ms=5)[0]

    def _draw(i):
        t = frame_t[i]
        tm = t * 1e3
        lines = []
        for k, ((V, phi), arrow, col) in enumerate(zip(phasors, arrows, colors)):
            ang = omegas[k] * t + phi
            x, y = V * np.cos(ang), V * np.sin(ang)
            arrow.set_positions((0, 0), (x, y))
            lines.append(f"{labels[k]}: V={V:>5.1f}V  φ={np.degrees(ang) % 360:>5.1f}°")
            dots1[k].set_data([tm], [x])
        info_text.set_text("\n".join(lines))
        cursor1.set_xdata([tm, tm])
        cursor2.set_xdata([tm, tm])
        v_now = sum(V * np.cos(omegas[k] * t + phi) for k, (V, phi) in enumerate(phasors))
        dot2.set_data([tm], [v_now])
        return tuple(arrows) + (info_text,) + tuple(dots1) + (cursor1, cursor2, dot2)

    plt.tight_layout()
    return animate(fig, _draw, frames=frames, interval=interval)


class PowerGridObservable(KalmanObservable):
    VOLTAGE = "V"   # measured instantaneous voltage (single scalar channel)


class PowerGridStateModel(PhasorOscillatorModel):
    """
    Harmonic voltage tracker for power grid signals (50/60 Hz).

    **Mains specialization of** :class:`PhasorOscillatorModel`. It *is* the **lab-frame**
    joint phasor estimator — the base class supplies all the machinery; this subclass only
    fixes the **mains harmonic structure**.

    **Lab-frame formulation** (contrast with notebook 02's rotating-frame
    :class:`~kalman.apps.oscillators.OscillatorModel`):

    * ``F = I`` — phasors are constant in the lab frame (for a steady sinusoid the phasor
      does not rotate; it is the signal that oscillates).
    * ``H(t) = [cos ω₁t, −sin ω₁t, cos ω₂t, −sin ω₂t, …]`` — the carrier rotation lives
      entirely in the **time-varying observation map**, not in the state transition.
    * Single scalar measurement ``z(t) = v(t)`` — no Hilbert transform, no demodulation;
      the raw voltage is ingested directly.

    Because ``H(t)`` couples all harmonics to the one measured voltage, the strong
    fundamental is fully claimed by H1 and leaves nothing for the even harmonics —
    so **H2/H4 correctly read ≈ 0**.  An independent-demodulator bank cannot do this:
    the 230 V fundamental survives demodulation in the H2 frame as a rotating phasor and
    shows up as a spurious amplitude.

    **Why mains lines are (ideally) odd-only.** A clean AC waveform has half-wave
    symmetry ``v(t ± T/2) = −v(t)``, which forces a Fourier series of **odd harmonics
    only** (f0, 3·f0, 5·f0, …).  Even harmonics vanish unless something breaks that
    symmetry — DC bias, half-wave rectifier loads, transformer saturation, geomagnetically
    induced currents.  Tracking all harmonics and *seeing* the even ones collapse to ≈ 0
    is therefore a **physics verification**; a non-zero even harmonic is a diagnostic
    of load asymmetry.  ``B > 0`` covers inharmonic partials
    (``f_k = k·f0·√(1+B·k²)``) for completeness.

    State layout (state = phasor per harmonic, observation = scalar voltage):
        x = [Re_1, Im_1,  Re_2, Im_2,  ...,  Re_N, Im_N]    (2N states)
        z = [v]                                             (1 observation)

    Parameters
    ----------
    sensor_id : str
        Channel / sensor name (e.g. "L1").
    f0 : float
        Fundamental frequency [Hz], default 50.
    harmonics : int
        Number of harmonics to track (k = 1 … harmonics), default 3.
    fs : float
        Sample rate [Hz], default 1024.
    B : float
        Inharmonicity coefficient (default 0 → ideal harmonic grid).
    measurement_uncertainty : float
        Measurement noise std [V], default 1.0.
    process_uncertainty : float
        Phasor random-walk std [V] (how fast amplitudes may drift), default 0.01.
    p0_std : float
        Initial per-phasor prior std [V] for acquisition, default 100.

    Usage
    -----
    Feed the raw measured voltage directly (no Hilbert transform required)::

        record = grid_model.filter(v, smooth=True)

    A 2-column analytic signal ``[real, imag]`` is still accepted — only the real
    (measured) column is used.
    """

    NAME = "Power Grid Harmonic Tracker"
    DESCRIPTION = (
        "Mains specialization of PhasorOscillatorModel: joint lab-frame phasor "
        "estimator tracking N harmonics of a 50/60 Hz grid from the scalar voltage — "
        "all harmonics coupled through one measurement, so absent (even) harmonics read ≈ 0."
    )
    VERSION = "4.0"

    @property
    def observables(self):
        return PowerGridObservable
    # `states` (PhasorState: REAL, IMAG) is inherited from PhasorOscillatorModel.

    def __init__(self, sensor_id: str,
                 f0: float = 50.0,
                 harmonics: int = 3,
                 fs: float = 1024.0,
                 B: float = 0.0,
                 measurement_uncertainty: float = 1.0,
                 process_uncertainty: float = 0.01,
                 p0_std: float = 100.0,
                 **kwargs):

        kwargs.pop("Q_factor", None)   # accepted for backward-compat; unused (no per-oscillator damping)
        kwargs.pop("n_harmonics", None)

        self.f0 = float(f0)
        self.inharmonicity = float(B)
        self.n_harmonics = int(harmonics)

        # Reuse the inharmonic frequency model: B=0 → exact harmonic grid f_k = k·f0.
        self.partial_freqs = partial_frequencies(f0, B, harmonics)

        super().__init__(sensor_id, self.partial_freqs, fs=fs,
                         measurement_uncertainty=measurement_uncertainty,
                         process_uncertainty=process_uncertainty,
                         p0_std=p0_std, **kwargs)

    # ==================================================================
    # Mains-friendly aliases over the generic PhasorOscillatorModel methods
    # (reusable in notebooks, e.g. for THD analysis).
    # ==================================================================
    def harmonic_envelope(self, record):
        """Per-harmonic amplitude ``a_k = √(Re²+Im²)`` and its ±1σ band — shape ``(T, n_harmonics)``."""
        return self.envelope(record)

    def reconstruct_voltage(self, record, t=None):
        """Reconstruct the time-domain voltage from the phasor state:
        ``v(t) = Σ_k Re_k·cos(ω_k t) − Im_k·sin(ω_k t)``."""
        return self.reconstruct(record, t)

    def subtract_voltage(self, measured, record, t=None) -> np.ndarray:
        """Return ``measured − reconstruct_voltage(record)``: the harmonic-suppressed
        residual — e.g. mains hum/THD removed from a measured voltage, analogous to
        :meth:`InterferometerModel.subtract` for spectral-line noise."""
        return np.asarray(measured, dtype=float).ravel() - self.reconstruct_voltage(record, t)

    # ==================================================================
    # Signature figures — override the generic KalmanModel.plot hooks.
    # Reached via ``model.plot(record, ...)`` / ``model.plot(record, interactive=True)``.
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=False,
                     measured=None, truth=None, true_amplitudes=None,
                     trim=0, window_ms=200.0, figsize=(14, 8), **kwargs):
        """Signature 2×2 figure: H1 real/imaginary phasor components (±1σ),
        all per-harmonic amplitude envelopes (±1σ), and the reconstructed waveform.

        Optional overlays (raw arrays aligned with ``record.time``): ``measured``
        (noisy voltage), ``truth`` (clean voltage), ``true_amplitudes`` (per-harmonic
        reference amplitudes, drawn dashed). ``trim`` drops the last N samples (e.g. to
        hide an acquisition/edge transient in the envelope panels).
        """
        import matplotlib.pyplot as plt
        t  = np.asarray(record.time)
        xk = np.asarray(record.xk)
        Pk = np.asarray(record.Pk)
        N  = self.n_harmonics
        sl = slice(None, -trim if trim else None)
        amp, sigma = self.harmonic_envelope(record)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"{self.name()} — Observables", fontweight="bold")

        # (0,0)/(0,1): H1 phasor components ±1σ
        for j, (lbl, col) in enumerate([("real", "tab:blue"), ("imaginary", "tab:orange")]):
            ax = axes[0, j]
            y = xk[:, j]
            s = np.sqrt(np.maximum(Pk[:, j, j], 0)) if Pk.size else np.zeros_like(y)
            ax.plot(t[sl], y[sl], color=col, lw=0.8, label=f"H1 {lbl[:2]} (Kalman)")
            ax.fill_between(t[sl], (y - s)[sl], (y + s)[sl], color=col, alpha=0.25, label="±1σ")
            if true_amplitudes is not None:
                ax.axhline(true_amplitudes[0] if j == 0 else 0.0, ls="--",
                           color="steelblue", lw=1.5, label="Truth")
            ax.set_title(f"H1 phasor — {lbl} part")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (V)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8)

        # (1,0): per-harmonic amplitude envelopes ±1σ
        ax = axes[1, 0]
        colors = plt.cm.tab10(np.arange(N) % 10)
        for k in range(N):
            ax.plot(t[sl], amp[sl, k], color=colors[k], lw=1.5, label=f"H{k+1}")
            ax.fill_between(t[sl], (amp[:, k] - sigma[:, k])[sl], (amp[:, k] + sigma[:, k])[sl],
                            color=colors[k], alpha=0.18)
            if true_amplitudes is not None and k < len(true_amplitudes):
                ax.axhline(true_amplitudes[k], ls="--", color=colors[k], lw=0.9, alpha=0.55)
        ax.set_title("Per-harmonic amplitude envelope (±1σ; dashed = truth)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (V)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2); ax.set_ylim(bottom=0)

        # (1,1): waveform — first window_ms (reconstruction always; measured/truth if given)
        ax = axes[1, 1]
        n_show = min(len(t), max(2, int(window_ms * 1e-3 * self.fs)))
        tm = t[:n_show] * 1e3
        if measured is not None:
            ax.plot(tm, np.asarray(measured)[:n_show], color="gray", alpha=0.4, lw=0.8, label="Measured")
        if truth is not None:
            ax.plot(tm, np.asarray(truth)[:n_show], "--", color="steelblue", lw=2.0, label="Truth")
        ax.plot(tm, self.reconstruct_voltage(record, t)[:n_show], color="darkorange",
                lw=1.5, label="Reconstructed (Kalman)")
        ax.set_title(f"Waveform — first {window_ms:.0f} ms")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

        fig.tight_layout()
        return fig

    def _plot_interactive(self, record, sample=None, *, measured=None, truth=None,
                          window_ms=200.0, frames=30, interval=120, trim=0,
                          figsize=(11, 4.5), **kwargs):
        """Interactive replacement for ``replay``: slide a ``window_ms`` window across
        the signal and watch the Kalman reconstruction track measured vs truth.
        Returns a Jupyter HTML player (or a ``FuncAnimation`` outside Jupyter)."""
        import matplotlib.pyplot as plt
        from ....helpers import animate
        t = np.asarray(record.time)
        v_hat = self.reconstruct_voltage(record, t)
        win = max(2, int(window_ms * 1e-3 * self.fs))
        stop = max(1, len(t) - win - trim)
        starts = list(range(0, stop, max(1, stop // max(1, frames))))

        fig, ax = plt.subplots(figsize=figsize)
        lines = {}
        if measured is not None:
            (lines["meas"],) = ax.plot([], [], color="gray", alpha=0.5, lw=0.8, label="Measured")
        if truth is not None:
            (lines["truth"],) = ax.plot([], [], "--", color="steelblue", lw=2.0, label="Truth")
        (lines["hat"],) = ax.plot([], [], color="darkorange", lw=1.6, label="Reconstructed (Kalman)")
        ref = np.asarray(measured) if measured is not None else v_hat
        ax.set_ylim(ref.min() * 1.05, ref.max() * 1.05)
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

        def _draw(s):
            sl = slice(s, s + win); tm = t[sl] * 1e3
            if "meas" in lines:  lines["meas"].set_data(tm, np.asarray(measured)[sl])
            if "truth" in lines: lines["truth"].set_data(tm, np.asarray(truth)[sl])
            lines["hat"].set_data(tm, v_hat[sl])
            ax.set_xlim(tm[0], tm[-1])
            ax.set_title(f"{self.name()} — waveform  "
                         f"(t = {t[s]:.2f}–{t[min(s + win, len(t) - 1)]:.2f} s)")
            return tuple(lines.values())

        return animate(fig, _draw, frames=starts, interval=interval)

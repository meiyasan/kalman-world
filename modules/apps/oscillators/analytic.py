"""Analytic (Hilbert-demodulated) single-oscillator model — LINEAR variant."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from ...framework.models.linear import LinearKalmanModel
from ...framework.models.composite import CompositeKalmanModel
from ...framework.constants import KalmanState, KalmanObservable, KalmanObservableReadout
from .base import (OscillatorState, OscillatorObservable, wire_inharmonicity,
                   wire_fundamental, partial_frequencies)

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



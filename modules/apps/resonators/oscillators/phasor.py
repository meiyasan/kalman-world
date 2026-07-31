"""Lab-frame joint phasor estimator — the PHASOR variant (mains/harmonics)."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from ....framework.models.linear import LinearKalmanModel
from ....framework.models.composite import CompositeKalmanModel
from ....framework.constants import KalmanState, KalmanObservable, KalmanObservableReadout
from .base import (OscillatorState, OscillatorObservable, wire_inharmonicity,
                   wire_fundamental, partial_frequencies)

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



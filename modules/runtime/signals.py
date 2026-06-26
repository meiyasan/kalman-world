"""
Signal-processing helpers — *is the Kalman filter just a low-pass filter?*

These are **not** recursive Bayesian filters and they need no Kalman model, so they live
in ``runtime`` next to the other utilities (``chrono``, ``timer``, …) rather than under
``apps``. They make the classic identity computable: for the scalar **local-level /
random-walk** model

    x_{k+1} = x_k + w_k      w_k ~ N(0, q²)        (process / drift)
    z_k     = x_k + v_k      v_k ~ N(0, r²)        (measurement)

the Kalman gain converges to a constant ``K_ss`` and the recursion collapses to

    μ_k = (1 − K_ss)·μ_{k-1} + K_ss·z_k

— *exactly* a first-order low-pass (exponential moving average) with pole ``1 − K_ss``.
The complementary high-pass is ``z − lowpass(z)``, and in the stationary limit the whole
thing coincides with the frequency-domain **Wiener filter** ``H = S_xx /(S_xx + S_nn)``.

Build the baselines straight from the two noise standard deviations ``q``/``r`` with the
``from_qr`` constructors (or directly from a known smoothing factor / variances).
:func:`pull` and :func:`error_ratio` quantify the improvement (Gaussian pull distributions
and error-std ratios). ``00_Kalman101.ipynb`` derives the same identity inline with a plain
``LinearKalmanFilter``, underscoring that none of this needs a special model class.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np


# Steady-state Riccati fixed point (constant-gain Kalman = Wiener = low-pass).
SteadyState = namedtuple("SteadyState", ["gain", "prior_var", "posterior_var"])


def steady_state_gain(process_uncertainty: float, measurement_uncertainty: float) -> SteadyState:
    """Constant-gain fixed point of the scalar Riccati recursion, from the noise stds ``q``/``r``.

    Solving ``P⁻ = P + Q``, ``K = P⁻/(P⁻+R)``, ``P = (1−K)P⁻`` for the steady state (with
    ``Q = q²``, ``R = r²``) gives ``P⁻ = ½(q² + √(q⁴ + 4q²r²))`` and the closed-form gain

        K_ss = (√(q⁴ + 4q²r²) − q²) / (2 r²).

    Returns ``(gain, prior_var P⁻, posterior_var P)``.
    """
    q2 = float(process_uncertainty) ** 2
    r2 = float(measurement_uncertainty) ** 2
    if r2 <= 0.0:                                  # perfect sensor → trust it fully
        return SteadyState(1.0, q2, 0.0)
    if q2 <= 0.0:                                  # static signal → gain decays to 0
        return SteadyState(0.0, 0.0, 0.0)
    disc = np.sqrt(q2 * q2 + 4.0 * q2 * r2)
    prior = 0.5 * (q2 + disc)
    gain = prior / (prior + r2)                    # == (disc − q2) / (2 r2)
    post = (1.0 - gain) * prior
    return SteadyState(float(gain), float(prior), float(post))


# ======================================================================
# Baseline filters (NOT recursive Bayesian filters)
# ======================================================================
class LowPassFilter:
    """First-order IIR low-pass / exponential moving average ``y_k=(1−α)y_{k-1}+α z_k``.

    Seeded with ``alpha = K_ss`` (via :meth:`from_qr`) it reproduces the steady-state
    Kalman estimate exactly once the gain has converged.
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    @classmethod
    def from_qr(cls, process_uncertainty: float, measurement_uncertainty: float) -> "LowPassFilter":
        """Seed ``alpha`` with the steady-state Kalman gain ``K_ss(q, r)``."""
        return cls(steady_state_gain(process_uncertainty, measurement_uncertainty).gain)

    def apply(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float).ravel()
        y = np.empty_like(z)
        acc = z[0] if z.size else 0.0
        for k, zk in enumerate(z):
            acc = (1.0 - self.alpha) * acc + self.alpha * zk
            y[k] = acc
        return y

    __call__ = apply


class HighPassFilter:
    """Complementary high-pass ``z − lowpass(z)`` of :class:`LowPassFilter`."""

    def __init__(self, alpha: float):
        self._lp = LowPassFilter(alpha)

    @classmethod
    def from_qr(cls, process_uncertainty: float, measurement_uncertainty: float) -> "HighPassFilter":
        return cls(steady_state_gain(process_uncertainty, measurement_uncertainty).gain)

    def apply(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float).ravel()
        return z - self._lp.apply(z)

    __call__ = apply


class WienerFilter:
    """Non-causal frequency-domain Wiener filter ``H(ω)=S_xx/(S_xx+S_nn)``.

    For the random-walk-plus-white-noise model ``S_xx(ω)=q²/(2−2cosω)`` (diverging at DC,
    so ``H→1``) and ``S_nn=r²``. This is the stationary optimum the Kalman filter morphs
    into; here it is applied as a zero-phase FFT smoother for comparison.
    """

    def __init__(self, process_variance: float, measurement_variance: float):
        self.q2 = float(process_variance)
        self.r2 = float(measurement_variance)

    @classmethod
    def from_qr(cls, process_uncertainty: float, measurement_uncertainty: float) -> "WienerFilter":
        """Build from the noise standard deviations ``q``/``r`` (variances ``q²``/``r²``)."""
        return cls(float(process_uncertainty) ** 2, float(measurement_uncertainty) ** 2)

    def response(self, omega) -> np.ndarray:
        omega = np.asarray(omega, dtype=float)
        denom = 2.0 - 2.0 * np.cos(omega)
        with np.errstate(divide="ignore", invalid="ignore"):
            Sxx = np.where(denom > 0.0, self.q2 / denom, np.inf)
            H = Sxx / (Sxx + self.r2)
        return np.where(np.isinf(Sxx), 1.0, H)        # DC passes fully

    def apply(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float).ravel()
        N = z.size
        if N == 0:
            return z
        Z = np.fft.rfft(z)
        omega = 2.0 * np.pi * np.fft.rfftfreq(N)       # rad/sample
        return np.fft.irfft(Z * self.response(omega), n=N)

    __call__ = apply


# ======================================================================
# Comparison utilities
# ======================================================================
def pull(estimate, truth, sigma) -> np.ndarray:
    """Normalised residual ``(estimate − truth)/σ`` — should be ~N(0,1) when calibrated."""
    estimate = np.asarray(estimate, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.where(np.abs(sigma) < 1e-30, 1e-30, sigma)
    return (estimate - truth) / sigma


def error_ratio(reference_err, filtered_err) -> float:
    """``std(reference_err)/std(filtered_err)`` — >1 means the filter reduced the error."""
    ref = np.std(np.asarray(reference_err, dtype=float).ravel())
    flt = np.std(np.asarray(filtered_err, dtype=float).ravel())
    return float(ref / flt) if flt > 0 else float("inf")

"""
1-D scalar signal estimation — *is the Kalman filter just a low-pass filter?*

This app makes the classic identity concrete (cf. *thesearesystems* "Is the Kalman
Filter Just a Low-Pass Filter?"): for the scalar **local-level / random-walk** model

    x_{k+1} = x_k + w_k      w_k ~ N(0, q²)        (process / drift)
    z_k     = x_k + v_k      v_k ~ N(0, r²)        (measurement)

the Kalman gain converges to a constant ``K_ss`` and the recursion collapses to

    μ_k = (1 − K_ss)·μ_{k-1} + K_ss·z_k

— *exactly* a first-order low-pass (exponential moving average) with pole ``1 − K_ss``.
The complementary high-pass is ``z − lowpass(z)``, and in the stationary limit the whole
thing coincides with the frequency-domain **Wiener filter** ``H = S_xx /(S_xx + S_nn)``.

The framework side is the :class:`ScalarSignalModel` (a one-state
:class:`~kalman.framework.models.linear.LinearKalmanModel`); the baselines
(:class:`WienerFilter`, :class:`LowPassFilter`, :class:`HighPassFilter`) are *app-level*
helpers — they are not recursive Bayesian filters, so they live here rather than in
``kalman.filters``. :func:`pull` and :func:`error_ratio` quantify the improvement
(Gaussian pull distributions + error-std ratios).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from ..framework.constants import KalmanObservable, KalmanState
from ..framework.models.linear import LinearKalmanModel


# ===== State / observable enums (units) =====
class SignalState(KalmanState):
    LEVEL = "a.u."          # the hidden scalar level being tracked


class SignalObservable(KalmanObservable):
    LEVEL = "a.u."


# Steady-state Riccati fixed point (constant-gain Kalman = Wiener = low-pass).
SteadyState = namedtuple("SteadyState", ["gain", "prior_var", "posterior_var"])


# ===== The framework model =====
class ScalarSignalModel(LinearKalmanModel):
    """Scalar local-level (random-walk) model; its steady-state KF is a 1st-order low-pass.

    Parameters
    ----------
    sensor_id : str
        Sensor / channel name.
    measurement_uncertainty : float
        Measurement noise std ``r`` (so ``R = r²``).
    process_uncertainty : float
        Process/drift noise std ``q`` (so ``Q = q²``). The ratio ``q/r`` sets the
        cutoff: large ``q`` ⇒ trust data ⇒ high cutoff; small ``q`` ⇒ heavy smoothing.
    fs : float
        Sample rate [Hz].

    Note
    ----
    ``model.sample(duration)`` returns a genuine random-walk realisation (the truth
    actually drifts under ``Q``); pass ``covariance=True`` instead if you want the
    deterministic mean plus the analytic prior covariance ``Pk``.
    """

    NAME = "Scalar Signal (random-walk) Model"
    DESCRIPTION = ("1-D local-level random walk x_{k+1}=x_k+w, z=x+v; the steady-state "
                   "Kalman filter is a first-order low-pass (exponential smoother).")
    VERSION = "1.0"

    @property
    def states(self):
        return SignalState

    @property
    def observables(self):
        return SignalObservable

    def __init__(self, sensor_id: str | int,
                 measurement_uncertainty: float = 1.0,
                 process_uncertainty: float = 0.05,
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", None)
        assert isinstance(fs, (int, float)), "You must provide fs or sample_rate"

        # std (uncertainty) → variance for Q/R
        self._q = self.dtype(process_uncertainty)
        self._r = self.dtype(measurement_uncertainty)

        F = np.array([[1.0]], dtype=self.dtype)              # random walk
        H = np.array([[1.0]], dtype=self.dtype)              # observe the level directly
        Q = np.array([[self._q ** 2]], dtype=self.dtype)
        R = np.array([[self._r ** 2]], dtype=self.dtype)

        # Start uncertain at the measurement noise level.
        kwargs.setdefault("P0", np.array([[self._r ** 2]], dtype=self.dtype))

        super().__init__(sensor_id, F=F, H=H, Q=Q, R=R, fs=fs, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs): return zk, uk
    def inprocess(self, state_vector, **kwargs): return state_vector
    def postprocess(self, xk_hat, **kwargs):     return xk_hat

    # ------------------------------------------------------------------
    # The article's argument, made computable
    # ------------------------------------------------------------------
    def steady_state_gain(self) -> SteadyState:
        """Constant-gain fixed point of the scalar Riccati recursion.

        Solving ``P⁻ = P + Q``, ``K = P⁻/(P⁻+R)``, ``P = (1−K)P⁻`` for the steady state
        gives ``P⁻ = ½(q² + √(q⁴ + 4q²r²))`` and the closed-form gain

            K_ss = (√(q⁴ + 4q²r²) − q²) / (2 r²).

        Returns ``(gain, prior_var P⁻, posterior_var P)``.
        """
        q2 = float(self._q) ** 2
        r2 = float(self._r) ** 2
        if r2 <= 0.0:                                  # perfect sensor → trust it fully
            return SteadyState(1.0, q2, 0.0)
        if q2 <= 0.0:                                  # static signal → gain decays to 0
            return SteadyState(0.0, 0.0, 0.0)
        disc = np.sqrt(q2 * q2 + 4.0 * q2 * r2)
        prior = 0.5 * (q2 + disc)
        gain = prior / (prior + r2)                    # == (disc − q2) / (2 r2)
        post = (1.0 - gain) * prior
        return SteadyState(float(gain), float(prior), float(post))

    def cutoff_frequency(self) -> float:
        """−3 dB-ish cutoff [Hz] of the equivalent one-pole low-pass (pole ``1−K_ss``).

        Uses the exponential-smoother time constant ``τ = −1/(fs·ln(1−K))``,
        so ``f_c = 1/(2πτ) = −fs·ln(1−K)/(2π)``.
        """
        K = self.steady_state_gain().gain
        pole = 1.0 - K
        if not (0.0 < pole < 1.0):
            return float(self.fs) / 2.0
        return float(-self.fs * np.log(pole) / (2.0 * np.pi))

    def frequency_response(self, freqs) -> np.ndarray:
        """Complex response ``H(e^{jω}) = K/(1 − (1−K)e^{−jω})`` of the steady-state low-pass."""
        K = self.steady_state_gain().gain
        w = 2.0 * np.pi * np.asarray(freqs, dtype=float) / float(self.fs)
        return K / (1.0 - (1.0 - K) * np.exp(-1j * w))

    # ==================================================================
    # Signature figure — reached via model.plot(record, sample=sample).
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(14, 9), **kwargs):
        """Three-panel signature: (A) time series with Kalman ±1σ, low-pass and Wiener
        overlays; (B) Bode magnitude of the equivalent low-pass + complementary high-pass;
        (C) Gaussian pull histograms (Kalman vs Wiener) with the N(0,1) reference and the
        printed error-std ratios."""
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        t = np.asarray(record.time)
        xk = np.asarray(record.xk).reshape(len(record), -1)[:, 0]                # Kalman estimate
        Pk = np.asarray(record.Pk).reshape(len(record), 1, 1)[:, 0, 0]
        sig = np.sqrt(np.maximum(Pk, 0.0))
        xt = np.asarray(sample.xk).reshape(len(sample), -1)[:, 0] if sample is not None else None
        zk = np.asarray(sample.zk).reshape(len(sample), -1)[:, 0] if sample is not None else None

        lp = LowPassFilter.from_model(self).apply(zk) if zk is not None else None
        wf = WienerFilter.from_model(self).apply(zk) if zk is not None else None
        ss = self.steady_state_gain()

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.27, height_ratios=[1.2, 1])

        # --- Panel A: time series ---
        axA = fig.add_subplot(gs[0, :])
        if zk is not None:
            axA.scatter(t, zk, s=8, color="black", alpha=0.25, label="measurement z", zorder=1)
        if xt is not None:
            axA.plot(t, xt, color="gray", lw=1.4, ls="--", alpha=0.9, label="truth", zorder=2)
        axA.fill_between(t, xk - sig, xk + sig, color="steelblue", alpha=0.25, label=r"Kalman $\pm1\sigma$")
        axA.plot(t, xk, color="steelblue", lw=2.0, label="Kalman", zorder=4)
        if lp is not None:
            axA.plot(t, lp, color="darkorange", lw=1.3, alpha=0.9, label=f"low-pass (K={ss.gain:.3f})")
        if wf is not None:
            axA.plot(t, wf, color="seagreen", lw=1.2, alpha=0.85, label="Wiener (freq-domain)")
        axA.set_xlabel("Time (s)"); axA.set_ylabel(f"level [{SignalState.LEVEL.value}]")
        axA.legend(fontsize=8, ncol=2, loc="best"); axA.grid(True, alpha=0.3)
        axA.set_title("Estimate vs. measurement — the Kalman filter as an adaptive low-pass")

        # --- Panel B: Bode magnitude ---
        axB = fig.add_subplot(gs[1, 0])
        fmax = float(self.fs) / 2.0
        freqs = np.linspace(fmax / 500.0, fmax, 500)
        Hlp = self.frequency_response(freqs)
        axB.semilogx(freqs, 20.0 * np.log10(np.abs(Hlp)), color="darkorange", lw=1.8, label="low-pass")
        axB.semilogx(freqs, 20.0 * np.log10(np.abs(1.0 - Hlp) + 1e-12), color="purple", lw=1.4,
                     alpha=0.8, label="high-pass (complementary)")
        fc = self.cutoff_frequency()
        axB.axvline(fc, color="gray", ls=":", lw=1.2, label=f"$f_c$≈{fc:.3g} Hz")
        axB.axhline(-3.0, color="lightgray", ls="--", lw=1.0)
        axB.set_xlabel("Frequency (Hz)"); axB.set_ylabel("Magnitude (dB)")
        axB.set_ylim(-40, 5); axB.legend(fontsize=8); axB.grid(True, alpha=0.3, which="both")
        axB.set_title("Equivalent filter response")

        # --- Panel C: pull distributions ---
        axC = fig.add_subplot(gs[1, 1])
        if xt is not None:
            pk = pull(xk, xt, sig)
            axC.hist(pk, bins=30, density=True, color="steelblue", alpha=0.55, label="Kalman pull")
            if wf is not None:
                pw = pull(wf, xt, np.sqrt(max(ss.posterior_var, 1e-30)))
                axC.hist(pw, bins=30, density=True, color="seagreen", alpha=0.40, label="Wiener pull")
            gx = np.linspace(-4, 4, 200)
            axC.plot(gx, norm.pdf(gx), color="black", lw=1.5, label="N(0,1)")
            axC.set_xlabel(r"pull $(\hat{x}-x)/\sigma$"); axC.set_ylabel("density")
            axC.legend(fontsize=8); axC.grid(True, alpha=0.3)
            r_raw = error_ratio(zk - xt, xk - xt) if zk is not None else float("nan")
            r_wnr = error_ratio(wf - xt, xk - xt) if wf is not None else float("nan")
            axC.set_title(f"Pull dist. — error↓ ×{r_raw:.2f} vs raw, ×{r_wnr:.2f} vs Wiener")

        fig.suptitle(f"{self.name()} — Kalman / Wiener / low-pass", fontsize=13, y=0.99)
        return fig


# ======================================================================
# App-level baseline filters (NOT recursive Bayesian filters → live here)
# ======================================================================
class LowPassFilter:
    """First-order IIR low-pass / exponential moving average ``y_k=(1−α)y_{k-1}+α z_k``.

    Seeded with ``alpha = K_ss`` (via :meth:`from_model`) it reproduces the steady-state
    Kalman estimate exactly once the gain has converged.
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    @classmethod
    def from_model(cls, model: ScalarSignalModel) -> "LowPassFilter":
        return cls(model.steady_state_gain().gain)

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
    def from_model(cls, model: ScalarSignalModel) -> "HighPassFilter":
        return cls(model.steady_state_gain().gain)

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
    def from_model(cls, model: ScalarSignalModel) -> "WienerFilter":
        return cls(float(model._q) ** 2, float(model._r) ** 2)

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
# Comparison utilities (reused by the notebook + tests)
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

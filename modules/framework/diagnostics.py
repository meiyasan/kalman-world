# kalman/framework/diagnostics.py
"""
Kalman filter **consistency diagnostics**.

Everything here is computed *post-hoc* from a completed forward pass — the stored
``prior`` / ``posterior`` / ``measurement`` logs plus the model's ``H`` / ``R`` / ``h`` —
so it needs no engine changes and works for any filter (linear, EKF, UKF, …) and any
model (named state enum or generic matrix model).

:class:`KalmanDiagnostics` wraps a :class:`~kalman.framework.filter.KalmanFilter` that has
already run ``forward()`` / ``filter()`` and exposes:

* :meth:`innovations` — the measurement residual ν and its covariance S,
* :meth:`gains`       — the Kalman gain over time,
* :meth:`nis`         — Normalized Innovation Squared (no ground truth needed),
* :meth:`nees`        — Normalized Estimation Error Squared (needs the truth),
* :meth:`plot`        — the model-agnostic consistency dashboard.

``KalmanFilter`` delegates its ``innovations()/gains()/nis()/nees()/diagnostics()`` to this
class, so ``kf.diagnostics(sample)`` simply *draws* the dashboard.
"""
from __future__ import annotations
import numpy as np

from .io.record import KalmanRecord


class KalmanDiagnostics:
    """Post-hoc consistency diagnostics for one Kalman forward pass.

    Construct from a filter that has already run ``forward()``/``filter()``::

        KalmanDiagnostics(kf).nis()
        KalmanDiagnostics(kf).plot(sample=truth)
    """

    def __init__(self, filter):
        self.filter = filter

    # ------------------------------------------------------------------
    #  Stored forward-pass quantities → plain arrays
    # ------------------------------------------------------------------
    def _arrays(self):
        """Stack the stored forward-pass quantities into plain arrays.

        Returns (N, n, m, z, x_prior, P_prior, x_post, P_post)."""
        f = self.filter
        N = len(f.posterior)
        if N == 0:
            raise RuntimeError("No forward pass to diagnose — run forward()/filter() first.")
        n, m = f.n, f.m
        z    = np.array([np.reshape(zz, (m, 1)) for zz in f.measurement[:N]])
        xpr  = np.array([np.reshape(p[0], (n, 1)) for p in f.prior])
        Ppr  = np.array([np.asarray(p[1]).reshape(n, n) for p in f.prior])
        xpo  = np.array([np.reshape(p[0], (n, 1)) for p in f.posterior])
        Ppo  = np.array([np.asarray(p[1]).reshape(n, n) for p in f.posterior])
        return N, n, m, z, xpr, Ppr, xpo, Ppo

    # ------------------------------------------------------------------
    #  Numeric diagnostics
    # ------------------------------------------------------------------
    def innovations(self):
        """Per-step innovation (measurement residual) and its covariance.

        ν_k = z_k − h(x⁻_k),   S_k = H_k P⁻_k H_kᵀ + R   (H_k = ∂h/∂x at the prior).
        A well-tuned filter has ν zero-mean and white, ~95% inside ±2·√diag(S).

        Returns (nu, S): shapes (N, m) and (N, m, m)."""
        model = self.filter.model
        N, n, m, z, xpr, Ppr, _, _ = self._arrays()
        R = np.asarray(model.R()).reshape(m, m)
        nu = np.zeros((N, m)); S = np.zeros((N, m, m))
        for k in range(N):
            Hk = np.asarray(model.H(xpr[k])).reshape(m, n)
            nu[k] = (z[k] - np.asarray(model.h(xpr[k])).reshape(m, 1)).ravel()
            S[k]  = Hk @ Ppr[k] @ Hk.T + R
        return nu, S

    def gains(self):
        """Per-step Kalman gain K_k = P⁻_k H_kᵀ S_k⁻¹. Shape (N, n, m).

        K[k, i, j] = how strongly measurement channel j moves state i at step k."""
        model = self.filter.model
        N, n, m, z, xpr, Ppr, _, _ = self._arrays()
        R = np.asarray(model.R()).reshape(m, m)
        K = np.zeros((N, n, m))
        for k in range(N):
            Hk = np.asarray(model.H(xpr[k])).reshape(m, n)
            S  = Hk @ Ppr[k] @ Hk.T + R
            K[k] = Ppr[k] @ Hk.T @ np.linalg.inv(S)
        return K

    def nis(self):
        """Normalized Innovation Squared  ε_k = ν_kᵀ S_k⁻¹ ν_k  (≈ χ²(m)). Shape (N,).

        Consistency check that needs NO ground truth — usable on real data."""
        nu, S = self.innovations()
        return np.array([float(nu[k] @ np.linalg.solve(S[k], nu[k])) for k in range(len(nu))])

    def nees(self, truth):
        """Normalized Estimation Error Squared (needs ground truth, e.g. ``sample.xk``):
        (x_k − x̂_k)ᵀ P_k⁻¹ (x_k − x̂_k)  (≈ χ²(n)). Shape (N,).

        Checks whether P is honest: time-mean ≈ n means well-calibrated, ≫ n means
        the filter is over-confident, ≪ n means over-conservative."""
        N, n, m, z, xpr, Ppr, xpo, Ppo = self._arrays()
        truth = np.asarray(truth)
        truth = truth.reshape(truth.shape[0], -1)
        e = truth[:N] - xpo.reshape(N, n)
        return np.array([float(e[k] @ np.linalg.pinv(Ppo[k]) @ e[k]) for k in range(N)])

    # ------------------------------------------------------------------
    #  Dashboard
    # ------------------------------------------------------------------
    def plot(self, sample=None, *, sigma=3, figsize=(14, 9), **kwargs):
        """Filter-consistency dashboard computed from the forward pass:

          1. **Innovation** ν per measurement channel, shaded ±``sigma``·√S — should
             be zero-mean and stay (mostly) inside the band.
          2. **Kalman gain** K(t) for measurement channel 0 across states — shows how
             strongly a measurement moves each state, converging to steady state.
          3. **NIS** with a χ²(m) 95% band (no truth needed).
          4. **NEES** with a χ²(n) 95% band — only if ``sample`` (ground truth) given.

        ``sample`` may be a ``KalmanRecord`` (uses ``sample.xk``) or a truth array.
        """
        import matplotlib.pyplot as plt
        f = self.filter
        N = len(f.posterior)
        n = f.n; m = f.m
        fs = f.sample_rate() or 1.0
        t0 = float(f.model.t0) if f.model.t0 is not None else 0.0
        t  = t0 + np.arange(N) / fs

        nu, S = self.innovations()
        K     = self.gains()
        nis   = self.nis()
        truth = sample.xk if isinstance(sample, KalmanRecord) else sample
        has_nees = truth is not None
        nees  = self.nees(truth) if has_nees else None

        try:
            from scipy.stats import chi2
            band = lambda df: (chi2.ppf(0.025, df), df, chi2.ppf(0.975, df))
        except Exception:
            band = lambda df: (None, df, None)

        # state labels (use the model's state enum when it lines up with n)
        try:
            names = [str(s).lower() for s in list(f.model.states)]
            labels = names if len(names) == n else [f"x{i}" for i in range(n)]
        except Exception:
            labels = [f"x{i}" for i in range(n)]

        npan = 4 if has_nees else 3
        fig, ax = plt.subplots(2, 2, figsize=figsize) if npan == 4 else plt.subplots(1, 3, figsize=(figsize[0], figsize[1] / 2.2))
        ax = np.atleast_1d(ax).ravel()
        fig.suptitle("Kalman filter consistency diagnostics", fontweight="bold")

        # 1) Innovations + ±sigma·sqrt(S)
        a0 = ax[0]
        for j in range(m):
            sd = np.sqrt(np.clip(S[:, j, j], 0, None))
            line, = a0.plot(t, nu[:, j], lw=1.0, label=f"ν[{j}]")
            a0.fill_between(t, -sigma * sd, sigma * sd, color=line.get_color(), alpha=0.15, lw=0,
                            label=(f"±{sigma:g}σ" if j == 0 else None))
        a0.axhline(0, color="gray", lw=0.8, ls="--")
        a0.set_title("Innovation (measurement residual)  ν = z − H·x⁻"); a0.set_xlabel("Time [s]"); a0.set_ylabel("residual")
        a0.legend(fontsize=8); a0.grid(True, ls=":", alpha=0.6)

        # 2) Kalman gain for measurement channel 0  (log y-scale)
        a1 = ax[1]
        for i in range(n):
            a1.plot(t, K[:, i, 0], lw=1.0, label=labels[i])
        # symlog: log-like compression so gains spanning orders of magnitude are all
        # visible, while still handling zero / negative (off-diagonal) gains.
        kmax = float(np.max(np.abs(K[:, :, 0]))) if K.size else 1.0
        linthresh = max(kmax * 1e-3, 1e-12)
        a1.set_yscale("symlog", linthresh=linthresh)
        a1.set_title("Kalman gain  |K(t)|  (meas. channel 0, log scale)"); a1.set_xlabel("Time [s]"); a1.set_ylabel("gain (symlog)")
        a1.legend(fontsize=8); a1.grid(True, ls=":", alpha=0.6, which="both")

        # 3) NIS + chi2(m) band
        a2 = ax[2]
        a2.plot(t, nis, lw=0.8, color="steelblue", label="NIS")
        lo, mid, hi = band(m)
        a2.axhline(mid, color="gray", lw=1.0, ls="--", label=f"E=m={m}")
        if lo is not None:
            a2.axhline(lo, color="crimson", lw=0.8, ls=":"); a2.axhline(hi, color="crimson", lw=0.8, ls=":", label="95% χ²")
        a2.set_title(f"Normalized Innovation Squared\ntime-mean ⟨NIS⟩ = {nis.mean():.2f}  ≈  m = {m}  (meas. dim)"); a2.set_xlabel("Time [s]"); a2.set_ylabel("νᵀS⁻¹ν")
        a2.legend(fontsize=8); a2.grid(True, ls=":", alpha=0.6)

        # 4) NEES + chi2(n) band
        if has_nees:
            a3 = ax[3]
            a3.plot(t, nees, lw=0.8, color="darkorange", label="NEES")
            lo, mid, hi = band(n)
            a3.axhline(mid, color="gray", lw=1.0, ls="--", label=f"E=n={n}")
            if lo is not None:
                a3.axhline(lo, color="crimson", lw=0.8, ls=":"); a3.axhline(hi, color="crimson", lw=0.8, ls=":", label="95% χ²")
            a3.set_title(f"Normalized Estimation Error Squared\ntime-mean ⟨NEES⟩ = {nees.mean():.2f}  ≈  n = {n}  (state dim)"); a3.set_xlabel("Time [s]"); a3.set_ylabel("eᵀP⁻¹e")
            a3.legend(fontsize=8); a3.grid(True, ls=":", alpha=0.6)

        fig.tight_layout()
        return fig

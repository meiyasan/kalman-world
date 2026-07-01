"""Particle engine (PF) — sequential importance resampling (SIR)."""
from __future__ import annotations
import numpy as np

from ..engine import KalmanEngine
from ..constants import KalmanEngineKind


class ParticleEngine(KalmanEngine):
    """
    Bootstrap / SIR particle filter.

    The **one stateful engine**: it holds the particle cloud for the duration of a
    forward pass (the filter calls ``reset(model)`` to (re)seed it from ``x0/P0``).
    ``_predict`` propagates particles through ``model.f`` with ``model.Q`` draws;
    ``_update`` reweights by the Gaussian innovation likelihood (``model.h``,
    ``model.R``), normalises, and systematic-resamples when the effective sample size
    drops. Both return the weighted mean/cov as ``(x, P)`` so records/logs are
    unchanged. No RTS smoother (forward-only).
    """

    NAME = "Particle Engine"
    DESCRIPTION = "Sequential importance-resampling particle filter (forward-only)."
    VERSION = "1.0"
    KIND = KalmanEngineKind.PARTICLE

    def __init__(self, n_particles: int = 500, resample_threshold: float = 0.5):
        self.n_particles = int(n_particles)
        self.resample_threshold = float(resample_threshold)
        self._particles = None   # (N, n)
        self._weights = None     # (N,)

    # ------------------------------------------------------------------
    # Per-run state
    # ------------------------------------------------------------------
    def reset(self, model):
        n = model.n
        x0 = np.asarray(model.x0, dtype=np.float64).reshape(-1)
        P0 = np.asarray(model.P0, dtype=np.float64)
        self._particles = np.random.multivariate_normal(x0, P0, size=self.n_particles)
        self._weights = np.full(self.n_particles, 1.0 / self.n_particles)
        return self

    def _estimate(self):
        w = self._weights
        mean = np.einsum("i,ij->j", w, self._particles)
        d = self._particles - mean
        cov = (d * w[:, None]).T @ d
        cov = 0.5 * (cov + cov.T) + 1e-9 * np.eye(cov.shape[0])
        return mean.reshape(-1, 1), cov

    def _resample(self):
        N = self.n_particles
        from ..native import HAS_C_ACCEL
        if HAS_C_ACCEL:
            from ..native import lib, as_contiguous, dptr, iptr
            weights = as_contiguous(self._weights)
            idx = np.empty(N, dtype=np.intc)
            lib.pf_systematic_resample(N, dptr(weights), float(np.random.uniform()), iptr(idx))
        else:
            positions = (np.arange(N) + np.random.uniform()) / N
            cumsum = np.cumsum(self._weights)
            cumsum[-1] = 1.0
            idx = np.searchsorted(cumsum, positions)
        self._particles = self._particles[idx]
        self._weights = np.full(N, 1.0 / N)

    # ------------------------------------------------------------------
    # Steps (stateless signature, internal particle state)
    # ------------------------------------------------------------------
    def _predict(self, model, x, P, u=None, t=None, **kw):
        if self._particles is None:
            self.reset(model)

        n = model.n
        u = (np.asarray(u).reshape(-1, 1) if (u is not None and model.l > 0) else None)
        Q = np.asarray(model.Q(t=t), dtype=np.float64)
        noise = np.random.multivariate_normal(np.zeros(n), Q, size=self.n_particles)
        accepts_w = self._accepts_param(model.f, "w")

        prop = np.empty_like(self._particles)
        for i, xp in enumerate(self._particles):
            xi = xp.reshape(-1, 1)
            if accepts_w:
                xi2 = model.f(xi, u=u, w=noise[i].reshape(-1, 1), t=t)
            else:
                xi2 = np.asarray(model.f(xi, u=u, t=t)).reshape(-1, 1) + noise[i].reshape(-1, 1)
            prop[i] = np.asarray(xi2, dtype=np.float64).reshape(-1)
        self._particles = prop

        x_mean, P_cov = self._estimate()
        return x_mean, P_cov, np.eye(n)

    def _update(self, model, x, P, z, t=None, **kw):
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        keep = ~np.isnan(z)
        if not keep.any():
            return self._estimate()

        idx = np.where(keep)[0]
        mv = len(idx)
        R = np.asarray(model.R(t=t), dtype=np.float64)[np.ix_(idx, idx)]
        Rinv = np.linalg.pinv(R)
        zc = z[idx]

        # model.h() is unavoidable Python (per-particle nonlinear map); only the
        # quadratic-form reduction below is pushed into C.
        zp = np.empty((self.n_particles, mv), dtype=np.float64)
        for i, xp in enumerate(self._particles):
            zp[i] = np.asarray(model.h(xp.reshape(-1, 1), t=t), dtype=np.float64).reshape(-1)[idx]

        logw = np.log(self._weights + 1e-300)
        from ..native import HAS_C_ACCEL
        if HAS_C_ACCEL:
            from ..native import lib, as_contiguous, dptr
            lib.pf_weight_update(
                self.n_particles, mv,
                dptr(as_contiguous(zc)), dptr(as_contiguous(zp)), dptr(as_contiguous(Rinv)), dptr(logw),
            )
        else:
            for i in range(self.n_particles):
                dz = zc - zp[i]
                logw[i] += -0.5 * dz @ Rinv @ dz
        logw -= logw.max()
        wts = np.exp(logw)
        wts /= wts.sum()
        self._weights = wts

        ess = 1.0 / np.sum(wts ** 2)
        if ess < self.resample_threshold * self.n_particles:
            self._resample()

        return self._estimate()

    def _smooth(self, *args, **kwargs):
        raise NotImplementedError("Particle filter has no RTS smoother; use forward() only.")

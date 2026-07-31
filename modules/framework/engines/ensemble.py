"""Ensemble engine (EnKF) — perturbed-observation ensemble Kalman filter."""
from __future__ import annotations
import numpy as np

from ..engine import KalmanEngine
from ..constants import KalmanEngineKind


def gaspari_cohn(dist: np.ndarray, L: float) -> np.ndarray:
    """Gaspari–Cohn 5th-order compactly-supported correlation function.

    Standard covariance-localization taper for ensemble filters: ≈ Gaussian of
    scale ``L`` but exactly zero beyond ``2L``, so spurious long-range sample
    correlations from a small ensemble are cut off instead of merely damped.
    ``dist`` is any array of distances; returns the taper with the same shape.
    """
    r = np.asarray(dist, dtype=float) / float(L)
    C = np.zeros_like(r)
    m1 = r <= 1
    C[m1] = 1 - 5/3*r[m1]**2 + 5/8*r[m1]**3 + 0.5*r[m1]**4 - 0.25*r[m1]**5
    m2 = (r > 1) & (r <= 2)
    rr = r[m2]
    C[m2] = 4 - 5*rr + 5/3*rr**2 + 5/8*rr**3 - 0.5*rr**4 + 1/12*rr**5 - 2/(3*rr)
    return np.maximum(C, 0.0)


class EnsembleKalmanEngine(KalmanEngine):
    """
    Perturbed-observation EnKF (Evensen) with multiplicative inflation and
    optional covariance localization.

    Like :class:`ParticleEngine` this is a **stateful** engine: it holds the
    ensemble ``(n, N)`` for the duration of a forward pass and (re)seeds it via
    ``reset(model)`` — from ``model.initial_ensemble(N, rng)`` when the model
    provides one (e.g. climatology snapshots), else from a Gaussian ``x0/P0``
    draw.  ``_predict``/``_update`` return the ensemble mean/covariance as
    ``(x, P)`` so records and diagnostics are unchanged; the raw members stay
    available as ``.members`` for DA-style workflows that never want the dense
    ``(n, n)`` covariance (see :meth:`forecast` / :meth:`analyze`).

    Analysis step (per cycle, forecast ensemble ``X``, anomalies ``X'``)::

        P Hᵀ  = ρ_xy ∘ X' Y'ᵀ / (N−1)          Y' = h(X)'   (obs-space anomalies)
        H P Hᵀ= ρ_yy ∘ Y' Y'ᵀ / (N−1)
        K     = P Hᵀ (H P Hᵀ + R)⁻¹
        Xᵃ    = X + K (y + ε − h(X)),   ε ~ N(0, R)  per member (perturbed obs)

    ``ρ_xy (n,p)`` / ``ρ_yy (p,p)`` are localization tapers (e.g.
    :func:`gaspari_cohn` of the state↔obs distances), taken from
    ``model.localization()`` when available, else no localization.

    **RNG discipline** — the perturbed-obs draws ``ε`` come from an
    engine-owned ``numpy`` Generator (``seed=``), *never* the global stream.
    When several filter variants run in the same loop for comparison, each
    engine consuming its own stream guarantees that adding/removing one method
    cannot silently change the random draws (and hence the results) of
    another.  Keep the *observation* noise on a separate shared stream in the
    experiment harness so every method sees the same measurement realization.

    Parameters
    ----------
    n_members : int
        Ensemble size ``N`` (typically 10–100; sampling error is tamed by
        localization + inflation, not by brute-force ``N``).
    inflation : float
        Multiplicative prior inflation ``a``: anomalies scaled by ``a`` before
        each analysis (compensates sampling error and model bias; surrogate
        forecast models generally need a larger ``a`` than the true model).
    localization : (rho_xy, rho_yy) or None
        Explicit taper matrices; default asks the model, else identity.
    seed : int or numpy.random.Generator, optional
        Engine-local RNG stream (see RNG discipline above).
    """

    NAME = "Ensemble Engine"
    DESCRIPTION = "Perturbed-observation EnKF: localized, inflated ensemble analysis (forward-only)."
    VERSION = "1.0"
    KIND = KalmanEngineKind.ENSEMBLE

    def __init__(self, n_members: int = 50, inflation: float = 1.0,
                 localization=None, seed=None):
        self.n_members = int(n_members)
        self.inflation = float(inflation)
        self._localization = localization
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._members = None     # (n, N) — state vectors as columns

    # ------------------------------------------------------------------
    # Per-run state
    # ------------------------------------------------------------------
    def reset(self, model):
        N = self.n_members
        init = getattr(model, "initial_ensemble", None)
        if callable(init):
            X = np.asarray(init(N, rng=self.rng), dtype=np.float64)
            if X.shape != (model.n, N):
                raise ValueError(f"initial_ensemble must return ({model.n},{N}); got {X.shape}")
        else:
            x0 = np.asarray(model.x0, dtype=np.float64).reshape(-1, 1)
            P0 = np.asarray(model.P0, dtype=np.float64)
            if np.allclose(P0, np.diag(np.diag(P0))):
                X = x0 + np.sqrt(np.diag(P0))[:, None] * self.rng.standard_normal((model.n, N))
            else:
                X = x0 + np.linalg.cholesky(P0) @ self.rng.standard_normal((model.n, N))
        self._members = X

        loc = self._localization
        if loc is None:
            hook = getattr(model, "localization", None)
            loc = hook() if callable(hook) else None
        self._rho_xy, self._rho_yy = (None, None) if loc is None else loc
        return self

    @property
    def members(self) -> np.ndarray:
        """Current ensemble, shape ``(n, N)`` (state vectors as columns)."""
        return self._members

    @property
    def mean(self) -> np.ndarray:
        return self._members.mean(axis=1, keepdims=True)

    def spread(self) -> float:
        """Mean per-state ensemble standard deviation (compare against RMSE)."""
        return float(self._members.std(axis=1).mean())

    def _estimate(self):
        x = self._members.mean(axis=1, keepdims=True)
        A = self._members - x
        P = (A @ A.T) / max(self.n_members - 1, 1)
        P = 0.5 * (P + P.T) + 1e-12 * np.eye(P.shape[0])
        return x, P

    # ------------------------------------------------------------------
    # Ensemble-native steps (no dense covariance) — the DA workhorse
    # ------------------------------------------------------------------
    def forecast(self, model, u=None, t=None):
        """Propagate every member through the model dynamics (+ ``Q`` draws).

        Uses ``model.forecast(X, t=t)`` when the model offers a batched map
        (columns → columns; also the hook where an AI surrogate plugs in),
        else loops ``model.f`` per member.  Returns the ensemble mean.
        """
        X = self._members
        n, N = X.shape
        u = (np.asarray(u).reshape(-1, 1) if (u is not None and model.l > 0) else None)

        batch = getattr(model, "forecast", None)
        if callable(batch):
            X = np.asarray(batch(X, t=t), dtype=np.float64)
        else:
            out = np.empty_like(X)
            for i in range(N):
                out[:, i] = np.asarray(
                    model.f(X[:, i].reshape(-1, 1), u=u, t=t), dtype=np.float64).reshape(-1)
            X = out

        Q = np.asarray(model.Q(t=t), dtype=np.float64)
        if Q.any():
            if np.allclose(Q, np.diag(np.diag(Q))):
                X = X + np.sqrt(np.diag(Q))[:, None] * self.rng.standard_normal((n, N))
            else:
                X = X + np.linalg.cholesky(Q) @ self.rng.standard_normal((n, N))
        self._members = X
        return self.mean

    def analyze(self, model, z, t=None):
        """Perturbed-obs analysis update of the ensemble; returns the mean.

        NaN channels in ``z`` are skipped (masked update, same convention as
        the other engines).  Inflation is applied to the forecast anomalies
        *before* the update.
        """
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        keep = ~np.isnan(z)
        if not keep.any():
            return self.mean

        X = self._members
        n, N = X.shape
        idx = np.where(keep)[0]

        if self.inflation != 1.0:
            m = X.mean(axis=1, keepdims=True)
            X = m + self.inflation * (X - m)

        # Obs-space ensemble h(X) — batched model hook if offered, else per member.
        hbatch = getattr(model, "observe", None)
        if callable(hbatch):
            Y = np.asarray(hbatch(X, t=t), dtype=np.float64)[idx, :]
        else:
            Y = np.empty((len(idx), N))
            for i in range(N):
                Y[:, i] = np.asarray(
                    model.h(X[:, i].reshape(-1, 1), t=t), dtype=np.float64).reshape(-1)[idx]

        R = np.asarray(model.R(t=t), dtype=np.float64)[np.ix_(idx, idx)]
        Xp = X - X.mean(axis=1, keepdims=True)
        Yp = Y - Y.mean(axis=1, keepdims=True)

        PHt  = (Xp @ Yp.T) / (N - 1)
        HPHt = (Yp @ Yp.T) / (N - 1)
        if self._rho_xy is not None:
            PHt  *= self._rho_xy[:, idx]
        if self._rho_yy is not None:
            HPHt *= self._rho_yy[np.ix_(idx, idx)]

        K = PHt @ np.linalg.pinv(HPHt + R)
        eps = (np.linalg.cholesky(R) @ self.rng.standard_normal((len(idx), N))
               if not np.allclose(R, np.diag(np.diag(R)))
               else np.sqrt(np.diag(R))[:, None] * self.rng.standard_normal((len(idx), N)))
        self._members = X + K @ (z[idx][:, None] + eps - Y)
        return self.mean

    # ------------------------------------------------------------------
    # Standard engine interface (wraps the ensemble steps; adds mean/cov)
    # ------------------------------------------------------------------
    def _predict(self, model, x, P, u=None, t=None, **kw):
        if self._members is None:
            self.reset(model)
        self.forecast(model, u=u, t=t)
        x, P = self._estimate()
        return x, P, np.eye(model.n)

    def _update(self, model, x, P, z, t=None, **kw):
        self.analyze(model, z, t=t)
        return self._estimate()

    def _smooth(self, *args, **kwargs):
        raise NotImplementedError("EnKF has no RTS smoother; use forward() only.")

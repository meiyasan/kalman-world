# kalman/framework/engine.py
from __future__ import annotations
import numpy as np

from .abstract.metadata import KalmanEngineMetadata
from .abstract.mixins import KalmanEngineMixin

import warnings
warnings.simplefilter("once", RuntimeWarning)


class KalmanEngine(KalmanEngineMetadata, KalmanEngineMixin):
    """
    Abstract Kalman **engine** base — the inference algorithm (HOW), one step at a time.

    Stateless w.r.t. run state: every method takes ``(model, x, P, …)`` and returns the
    new belief. The base provides the **shared** machinery — innovation ``S`` / gain
    ``K`` / Joseph-form update (with missing-measurement masking), the RTS smoother
    ``_smooth``, and ``infer`` (predict→update, from the mixin). Per-algorithm parts are
    the hooks ``_propagate`` / ``_measure`` (overridden by :class:`LinearKalmanEngine` /
    :class:`ExtendedKalmanEngine`); UKF/PF override ``_predict``/``_update`` wholesale.

    The engine *knows the model* (it calls ``model.F/H`` or ``model.f/h``) but owns no
    run state and nothing about workflow — that is the ``KalmanFilter``'s job.
    """

    NAME = "Kalman Engine"
    DESCRIPTION = "Abstract engine base: shared masked update + RTS smoother; per-algorithm hooks."
    VERSION = "2.0"
    KIND = None

    # ------------------------------------------------------------------
    # Per-algorithm hooks (Linear / Extended override these)
    # ------------------------------------------------------------------
    def _propagate(self, model, x, u, t):
        """Return (x_pred, F): the propagated mean and the transition matrix."""
        raise NotImplementedError

    def _measure(self, model, x, t):
        """Return (H, h_pred): the observation matrix and the predicted measurement."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_L(model, x, u, t):
        """Process-noise coupling L (handles both L(x,u,w,t) and composite L(t))."""
        try:
            L = model.L(x=x, u=u, w=np.zeros((model.n, 1)), t=t)
        except TypeError:
            L = model.L(t=t)
        return L

    @staticmethod
    def _accepts_param(fn, name: str) -> bool:
        """True if callable ``fn`` accepts a keyword/positional parameter ``name``
        (used to auto-detect non-additive noise: ``f(...,w)`` / ``h(...,v)``)."""
        import inspect
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return any(
            p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name == name
            for p in sig.parameters.values()
        )

    # ------------------------------------------------------------------
    # One estimation step
    # ------------------------------------------------------------------
    def _predict(self, model, x, P, u=None, w=None, t=None, **kw):
        """Prediction step → (x_pred, P_pred, F)."""
        if u is not None:
            u = np.asarray(u).reshape(-1, 1)
        u_for = u if model.l > 0 else None

        xp, F = self._propagate(model, x, u_for, t)

        # Pass the current state to Q only for models whose Q accepts it (e.g. a
        # speed-dependent air-friction Q). Constant-Q models keep Q(t=t) untouched.
        Q = model.Q(t=t, x=x) if self._accepts_param(model.Q, "x") else model.Q(t=t)
        L = self._safe_L(model, x, u_for, t)
        if L is not None:
            Q = L @ Q @ L.T

        return xp, F @ P @ F.T + Q, F

    def _update(self, model, x, P, z, t=None, **kw):
        """Update step → (x_post, P_post). Missing channels (NaN in z) are skipped."""
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        keep = ~np.isnan(z).ravel()
        if not keep.any():                      # all measurements missing → predict-only
            return x, P

        H, h_pred = self._measure(model, x, t)
        H = np.asarray(H)[keep, :]
        R = np.asarray(model.R(t=t))[np.ix_(keep, keep)]
        y = z[keep] - np.asarray(h_pred, dtype=float).reshape(-1, 1)[keep]

        S = R + H @ P @ H.T
        try:
            K = P @ H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            raise RuntimeError("Kalman update failed; check model stability / noise covariances.")

        I = np.eye(model.n)
        x = x + K @ y
        IKH = I - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T       # Joseph form (numerically stable)
        return x, P

    def _smooth(self, model, post_k, prior_next, smooth_next, F_k):
        """One reverse RTS step → (x_s, P_s)."""
        x, P = post_k
        xp, Pp = prior_next
        xs, Ps = smooth_next
        if np.linalg.cond(Pp) < 1e12:
            C = P @ F_k.T @ np.linalg.inv(Pp)
        else:
            C = P @ F_k.T @ np.linalg.pinv(Pp)
        return x + C @ (xs - xp), P + C @ (Ps - Pp) @ C.T

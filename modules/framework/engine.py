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

    **Core quantities** (passed to every method as arguments, never stored here):

    - ``x`` (ψ in math notation) — state vector, shape ``(n, 1)``.  The hidden
      quantities the filter estimates (e.g. position, velocity, acceleration).
    - ``P`` — state-error **covariance matrix**, shape ``(n, n)``.  Encodes uncertainty
      in the current estimate: ``P = E[(ψ - ψ̂)(ψ - ψ̂)ᵀ]``.  Diagonal entries are
      per-state variances; off-diagonals capture cross-correlations.
    - ``F`` — **state-transition matrix**, shape ``(n, n)``.  The discrete-time linear
      map ``ψ_{k+1} = F ψ_k`` (no noise); equals ``exp(A Δt)`` for a continuous system
      with dynamics matrix ``A``.  For nonlinear models ``F = ∂f/∂ψ`` (Jacobian).
    - ``Q`` — process-noise covariance, ``(n, n)``; ``R`` — measurement-noise
      covariance, ``(m, m)``.  Larger ``Q`` → trust model less; larger ``R`` → trust
      sensor less.

    **Two layers**:

    - ``_propagate`` / ``_measure`` — pure model evaluation for one step: the
      nonlinear map and its Jacobian (``f``/``F`` for the state transition,
      ``h``/``H`` for the observation). This is the *only* part that differs
      between algorithms, so it's the hook each engine overrides
      (:class:`LinearKalmanEngine` / :class:`ExtendedKalmanEngine`).
    - ``_predict`` / ``_update`` — the **shared** covariance machinery: propagate/
      update ``P``, inject process noise ``Q``, compute the gain ``K``, apply the
      masked Joseph-form correction. Identical for every Jacobian-based engine;
      it calls ``_propagate``/``_measure`` to get the algorithm-specific piece.

    UKF/PF have no single Jacobian (sigma points / particles instead), so they
    skip the hooks and override ``_predict``/``_update`` wholesale.

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
        """Return (xk, F): the propagated mean and the transition matrix."""
        raise NotImplementedError

    def _measure(self, model, x, t):
        """Return (zk, H): the predicted measurement and the observation matrix."""
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
    def _predict(self, model, x, P, u=None, t=None, **kw):
        """Prediction step → ``(xk, Pk, F)``.

        Propagates the prior belief one step forward using the motion model:

        .. code-block:: text

            xk  = F · x  + B · u          (mean prediction)
            Pk  = F · P · Fᵀ + Q          (covariance prediction)

        where ``F`` is the state-transition matrix (``∂f/∂x`` for nonlinear models)
        and ``Q`` is the process-noise covariance.  Returns the predicted mean
        ``xk``, the predicted covariance ``Pk``, and the transition matrix
        ``F`` used (needed later by the RTS smoother).
        """
        if u is not None:
            u = np.asarray(u).reshape(-1, 1)
        u = u if model.l > 0 else None

        xk, F = self._propagate(model, x, u, t)

        # Pass the current state to Q only for models whose Q accepts it (e.g. a
        # speed-dependent air-friction Q). Constant-Q models keep Q(t=t) untouched.
        Q = model.Q(t=t, x=x) if self._accepts_param(model.Q, "x") else model.Q(t=t)
        L = self._safe_L(model, x, u, t)
        if L is not None:
            Q = L @ Q @ L.T

        return xk, F @ P @ F.T + Q, F
    
    def _update(self, model, x, P, z, t=None, **kw):
        """Update step → ``(x, P)``.  Missing channels (NaN in ``z``) are skipped.

        Fuses the predicted belief with the measurement using the standard equations:

        .. code-block:: text

            S  = H · P · Hᵀ + R              (innovation covariance)
            K  = P · Hᵀ · S⁻¹               (Kalman gain)
            x  = x + K · (z − H · x)        (corrected mean, overwrites prior)
            P  = (I − KH) · P · (I − KH)ᵀ + K · R · Kᵀ   (Joseph form, overwrites prior)

        The Joseph form is used for ``P`` to preserve symmetry and positive-definiteness
        even under numerical round-off.
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        keep = ~np.isnan(z).ravel()
        if not keep.any():                      # all measurements missing → predict-only
            return x, P

        zk, H = self._measure(model, x, t)
        H = np.asarray(H, dtype=float)
        R = np.asarray(model.R(t=t), dtype=float)
        # Residual against the true (possibly nonlinear) h(x), NOT H @ x -- NaN
        # propagates automatically wherever z is missing, matching `keep`.
        yk = z.reshape(-1) - np.asarray(zk, dtype=float).reshape(-1)

        from .native import HAS_C_ACCEL
        if HAS_C_ACCEL:
            from .native import lib, as_contiguous, dptr
            n = model.n
            xk = np.empty(n, dtype=np.float64)
            Pk = np.empty((n, n), dtype=np.float64)
            lib.kf_update_residual(
                n, H.shape[0],
                dptr(as_contiguous(H)), dptr(as_contiguous(np.asarray(x, dtype=float).reshape(-1))),
                dptr(as_contiguous(P)),
                dptr(as_contiguous(yk)), dptr(as_contiguous(R)),
                dptr(xk), dptr(Pk),
            )
            return xk.reshape(-1, 1), Pk

        H = H[keep, :]
        R = R[np.ix_(keep, keep)]
        y = yk[keep].reshape(-1, 1)

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
        """One reverse RTS (Rauch–Tung–Striebel) step → ``(x_s, P_s)``.

        Given the forward-pass posterior at step ``k`` and the already-smoothed estimate
        at step ``k+1``, computes the smoothed estimate at step ``k``:

        .. code-block:: text

            C   = P_k · Fᵀ · P_{k+1|k}⁻¹    (smoother gain)
            x_s = x_k + C · (x_s_{k+1} − x_{k+1|k})
            P_s = P_k + C · (P_s_{k+1} − P_{k+1|k}) · Cᵀ

        where ``F_k`` is the transition matrix stored during the forward pass at step ``k``.
        """
        x, P = post_k
        xp, Pp = prior_next
        xs, Ps = smooth_next
        if np.linalg.cond(Pp) < 1e12:
            C = P @ F_k.T @ np.linalg.inv(Pp)
        else:
            C = P @ F_k.T @ np.linalg.pinv(Pp)
        return x + C @ (xs - xp), P + C @ (Ps - Pp) @ C.T

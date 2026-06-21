"""Combinatorial Kalman engine (CKF) — track-following step with a χ² validation gate.

This is the per-step *math* for combinatorial track finding (the Belle II CKF family).
It reuses the Extended engine's nonlinear propagation (``model.f`` / Jacobian ``model.F``)
and measurement map (``model.h`` / ``model.H``); the only addition is :meth:`gate`, which
scores a *candidate hit* against the prediction with a Mahalanobis χ² and returns the
Kalman-updated belief. The branch/prune/arbitrate **workflow** lives in
:class:`~kalman.filters.ckf.CombinatorialKalmanFilter`, not here — the engine stays
stateless, exactly like the other engines.
"""
from __future__ import annotations
import numpy as np

from .extended import ExtendedKalmanEngine
from ..constants import EngineKind


class CombinatorialKalmanEngine(ExtendedKalmanEngine):
    """
    Extended-Kalman step + a measurement **gate**.

    Inherits ``_propagate``/``_measure`` (hence ``_predict``/``_update``/``_smooth``) from
    :class:`ExtendedKalmanEngine`, so layer-to-layer propagation through the (curved)
    trajectory and the RTS smoother come for free. :meth:`gate` is the combinatorial
    primitive: given a *prediction* ``(x_pred, P_pred)`` and one candidate measurement
    ``z`` it returns ``(chi2, x_post, P_post)`` — the χ² compatibility of the hit and the
    belief updated with it (Joseph form). The filter opens a branch per hit whose
    ``chi2`` is below its gate.
    """

    NAME = "Combinatorial Kalman Engine"
    DESCRIPTION = "EKF track-following step plus a χ² (Mahalanobis) measurement gate for branching."
    VERSION = "1.0"
    KIND = EngineKind.COMBINATORIAL

    def gate(self, model, x_pred, P_pred, z, t=None):
        """Score and apply one candidate measurement.

        Parameters
        ----------
        model : KalmanModel       the system (provides ``h``/``H``/``R``).
        x_pred, P_pred            the *predicted* state mean (n×1) and covariance (n×n).
        z                         one candidate measurement (m×1); NaN channels are skipped.
        t                         station/time tag forwarded to ``h``/``H``/``R``.

        Returns
        -------
        (chi2, x_post, P_post) : the innovation χ² (float), and the Kalman-updated mean
        and covariance. ``chi2`` is ``+inf`` if every channel is missing.
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        keep = ~np.isnan(z).ravel()
        if not keep.any():
            return np.inf, x_pred, P_pred

        H, h_pred = self._measure(model, x_pred, t)
        H = np.asarray(H)[keep, :]
        R = np.asarray(model.R(t=t))[np.ix_(keep, keep)]
        y = z[keep] - np.asarray(h_pred, dtype=float).reshape(-1, 1)[keep]

        S = R + H @ P_pred @ H.T
        Sinv = np.linalg.pinv(S)
        chi2 = float((y.T @ Sinv @ y).item())

        K = P_pred @ H.T @ Sinv
        I = np.eye(model.n)
        x_post = x_pred + K @ y
        IKH = I - K @ H
        P_post = IKH @ P_pred @ IKH.T + K @ R @ K.T   # Joseph form (numerically stable)
        return chi2, x_post, P_post

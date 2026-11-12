"""BoostKalmanController — innovation-based Kalman gain boost via R-scaling."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class BoostKalmanController(KalmanController):
    """When the filter is surprised (NIS >> m), shrink R so K grows.

    Monitors Normalised Innovation Squared over a sliding window.
    When NIS exceeds ``threshold × m``, R is scaled down by
    ``boost_factor``, causing K to grow (trust measurements more).
    When NIS is nominal, R returns to its original value.

    Integrates with ``KalmanFilter.forward(uk=boost)`` — S is computed
    internally from model.H and model.R so no manual step-by-step loop
    is needed.

    Note: ``model._R`` mutation is effective only for leaf
    ``LinearKalmanModel`` instances.  For composite window models (e.g.
    ``OscillatorModel``) R is bound as a closure and ``_R`` is ignored;
    use ``AdaptiveQRKalmanController`` for composite models.

    Usage::

        boost = BoostKalmanController(model, boost_factor=3.0, threshold=2.0)
        record = filter.forward(zk, uk=boost)

    With libKalman installed the NIS window and the decision are computed by
    ``Kalman::Controllers::Boost`` (reading the model's ``H`` and its current ``R``
    through callbacks); the factor it decides is applied here, to ``model._R``,
    exactly as the Python reference (:meth:`_reference_update`) does. One
    difference: an innovation with missing (NaN) channels is scored on the
    channels present in C++, where the reference's NIS is NaN (and never boosts).
    """

    NATIVE = "boost"

    def __init__(self, model, boost_factor=3.0, threshold=2.0, window=10, **kw):
        # Initialize before super().__init__() since that calls reset()
        self.boost_factor = float(boost_factor)
        self.threshold = float(threshold)
        self.window = int(window)
        self._R_nominal = None
        self._nis_history: list = []
        self.scale_log: list = []
        super().__init__(model, **kw)

    def _policy(self, x_hat, P, t, *, nu=None):
        u = self._native_policy(x_hat, P, t, nu)
        return u if u is not None else self._reference_policy(x_hat, P, t, nu=nu)

    def _reference_policy(self, x_hat, P, t, *, nu=None):
        return np.zeros((max(self.model.l, 1), 1))

    def _update(self, x_hat, P, t, *, nu=None):
        if nu is None:
            return None
        if self._native_handle() is not None and self._native_belief(x_hat, P, nu) is not None:
            if self._R_nominal is None:
                self._R_nominal = self.model.R(t=t).copy()
            _, scale = self._native_update(x_hat, P, t, nu)
            if scale is not None:
                self.model._R = scale * self._R_nominal
                self.scale_log.append((t, scale))
            return None
        return self._reference_update(x_hat, P, t, nu=nu)

    def _reference_update(self, x_hat, P, t, *, nu=None):
        if nu is None:
            return None
        if self._R_nominal is None:
            self._R_nominal = self.model.R(t=t).copy()

        H = self.model.H(x=x_hat, t=t)
        S = H @ P @ H.T + self.model.R(t=t)
        nis = float((nu.T @ np.linalg.pinv(S) @ nu).item())
        self._nis_history.append(nis)

        recent = self._nis_history[-self.window:]
        nis_avg = np.mean(recent)
        expected = int(nu.shape[0])

        scale = 1.0 / self.boost_factor if nis_avg > self.threshold * expected else 1.0
        self.model._R = scale * self._R_nominal
        self.scale_log.append((t, scale))
        return None

    def reset(self):
        super().reset()
        if self._R_nominal is not None:
            self.model._R = self._R_nominal.copy()
        self._nis_history.clear()
        self.scale_log.clear()

    def _native_reads_noise(self):
        return True

    def _native_key(self):
        if not (self.boost_factor > 0) or self.window <= 0:
            return None
        costs = self._cost_key()
        return None if costs is None else (self.boost_factor, self.threshold, self.window) + costs

    def _native_create(self, model_handle):
        from ..native import core
        return core.create_controller("boost", model_handle, int(self.model.n), max(int(self.model.l), 1),
                                      boost_factor=self.boost_factor, threshold=self.threshold,
                                      window=self.window, **self._cost_args())

"""AdaptiveQRKalmanController — innovation-based Q scaling."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class AdaptiveQRKalmanController(KalmanController):
    """Compare empirical innovation covariance vs model-predicted S.

    If the ratio exceeds 1, Q is too small (model overconfident) — scale
    Q up.  Clipped to ``[q_min, q_max]`` for stability.

    Integrates with ``KalmanFilter.forward(uk=aqr)`` — S is computed
    internally from model.H and model.R.  For leaf ``LinearKalmanModel``
    instances, ``model._Q`` is mutated directly.  For composite window
    models (e.g. ``OscillatorModel``) use ``_LeafAdaptiveQR`` (in
    interferometer.py) which reaches into the leaf sub-models.

    Usage::

        aqr = AdaptiveQRKalmanController(model, window=20)
        record = filter.forward(zk, uk=aqr)

    With libKalman installed the innovation window and the factor are computed by
    ``Kalman::Controllers::AdaptiveQR`` (reading the model's ``H`` and ``R`` through
    callbacks); the factor is applied here, to ``model._Q``, exactly as the Python
    reference (:meth:`_reference_update`) does. One difference: an innovation with
    missing (NaN) channels is left out of the window in C++, where the reference
    takes it in and its factor becomes NaN.
    """

    NATIVE = "adaptive_qr"

    def __init__(self, model, window=20, q_min=0.1, q_max=10.0, **kw):
        # Initialize before super().__init__() since that calls reset()
        self.window = int(window)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self._Q_nominal = None
        self._nu_history: list = []
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
            if self._Q_nominal is None:
                self._Q_nominal = self.model.Q(t=t).copy()
            _, lam = self._native_update(x_hat, P, t, nu)
            if lam is not None:
                self.model._Q = lam * self._Q_nominal
                self.scale_log.append((t, lam))
            return None
        return self._reference_update(x_hat, P, t, nu=nu)

    def _reference_update(self, x_hat, P, t, *, nu=None):
        if nu is None:
            return None
        if self._Q_nominal is None:
            self._Q_nominal = self.model.Q(t=t).copy()

        self._nu_history.append(np.asarray(nu, dtype=float).reshape(-1, 1).copy())
        if len(self._nu_history) < self.window:
            return None

        recent = self._nu_history[-self.window:]
        S_empirical = np.mean([v @ v.T for v in recent], axis=0)

        H = self.model.H(x=x_hat, t=t)
        S_model = H @ P @ H.T + self.model.R(t=t)
        lam = float(np.clip(
            np.trace(S_empirical) / max(np.trace(S_model), 1e-12),
            self.q_min, self.q_max,
        ))

        self.model._Q = lam * self._Q_nominal
        self.scale_log.append((t, lam))
        return None

    def reset(self):
        super().reset()
        if self._Q_nominal is not None:
            self.model._Q = self._Q_nominal.copy()
        self._nu_history.clear()
        self.scale_log.clear()

    def _native_reads_noise(self):
        return True

    def _native_key(self):
        if self.window <= 0 or not (self.q_min > 0) or self.q_max < self.q_min:
            return None
        costs = self._cost_key()
        return None if costs is None else (self.window, self.q_min, self.q_max) + costs

    def _native_create(self, model_handle):
        from ..native import core
        return core.create_controller("adaptive_qr", model_handle, int(self.model.n), max(int(self.model.l), 1),
                                      window=self.window, q_min=self.q_min, q_max=self.q_max,
                                      **self._cost_args())

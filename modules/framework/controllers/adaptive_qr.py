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
    """

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
        return np.zeros((max(self.model.l, 1), 1))

    def _update(self, x_hat, P, t, *, nu=None):
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

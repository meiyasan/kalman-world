"""AdaptiveQRKalmanController — innovation-based Q scaling."""
from __future__ import annotations
import numpy as np


class AdaptiveQRKalmanController:
    """Compare empirical innovation covariance vs model-predicted S.

    If the ratio exceeds 1, Q is too small (model overconfident) — scale
    Q up.  Clipped to ``[q_min, q_max]`` for stability.

    Usage::

        aqr = AdaptiveQRKalmanController(model, window=20)
        # after each filter step:
        aqr(nu, S_model)   # scales model._Q internally
    """

    def __init__(self, model, window=20, q_min=0.1, q_max=10.0):
        self.model = model
        self.window = int(window)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self._Q_nominal = None
        self._nu_history = []
        self.scale_log = []

    def __call__(self, nu, S_model, t=None):
        if self._Q_nominal is None:
            self._Q_nominal = self.model.Q().copy()
        nu = np.asarray(nu, dtype=float).reshape(-1, 1)
        self._nu_history.append(nu.copy())

        if len(self._nu_history) < self.window:
            return 1.0

        recent = self._nu_history[-self.window:]
        S_empirical = np.mean([v @ v.T for v in recent], axis=0)
        lam = np.trace(S_empirical) / max(np.trace(S_model), 1e-12)
        lam = float(np.clip(lam, self.q_min, self.q_max))

        self.model._Q = lam * self._Q_nominal
        self.scale_log.append((t, lam))
        return lam

    def reset(self):
        if self._Q_nominal is not None:
            self.model._Q = self._Q_nominal.copy()
        self._nu_history.clear()
        self.scale_log.clear()

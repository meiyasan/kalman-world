"""BoostKalmanController — innovation-based Kalman gain boost via R-scaling."""
from __future__ import annotations
import numpy as np


class BoostKalmanController:
    """When the filter is surprised (NIS >> m), shrink R so K grows.

    Monitors Normalised Innovation Squared over a sliding window.
    When NIS exceeds ``threshold × m``, R is scaled down by
    ``boost_factor``, causing K to grow (trust measurements more).
    When NIS is nominal, R returns to its original value.

    Usage::

        boost = BoostKalmanController(model, boost_factor=3.0, threshold=2.0)
        # inside a controller or callback:
        boost(nu, S)   # scales model._R internally
    """

    def __init__(self, model, boost_factor=3.0, threshold=2.0, window=10):
        self.model = model
        self.boost_factor = float(boost_factor)
        self.threshold = float(threshold)
        self.window = int(window)
        self._R_nominal = None
        self._nis_history = []
        self.scale_log = []

    def __call__(self, nu, S, t=None):
        if self._R_nominal is None:
            self._R_nominal = self.model.R().copy()
        nu = np.asarray(nu, dtype=float).reshape(-1, 1)
        nis = float((nu.T @ np.linalg.pinv(S) @ nu).item())
        self._nis_history.append(nis)

        recent = self._nis_history[-self.window:]
        nis_avg = np.mean(recent)
        expected = nu.shape[0]

        if nis_avg > self.threshold * expected:
            scale = 1.0 / self.boost_factor
        else:
            scale = 1.0

        self.model._R = scale * self._R_nominal
        self.scale_log.append((t, scale))
        return scale

    def reset(self):
        if self._R_nominal is not None:
            self.model._R = self._R_nominal.copy()
        self._nis_history.clear()
        self.scale_log.clear()

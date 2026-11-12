"""ThresholdRouter — NIS threshold-based mode routing."""
from __future__ import annotations
import numpy as np
from .base import KalmanRouter


class ThresholdRouter(KalmanRouter):
    """Route based on how close each model's smoothed NIS is to the χ²(m) mean.

    Under the correct model NIS_j ~ χ²(m) with E[NIS_j] = m.
    This router assigns credit to the model whose recent NIS is closest to m.

    Hard mode (``soft=False``, default):
        Full weight on the single best-matching model.

    Soft mode (``soft=True``):
        Inverse-distance weighting over all models.

    No training required — ``fit()`` is a no-op.

    Parameters
    ----------
    m : int
        Measurement dimension (expected NIS under the correct model).
    window : int
        Number of past NIS values to average before comparing to m.
    soft : bool
        Hard argmin (False) vs inverse-distance softmax (True).

    Decided by ``Kalman::Routers::Threshold`` when libKalman is installed;
    :meth:`_reference_route` is the Python reference.
    """

    NATIVE = "threshold"

    def __init__(self, m: int = 1, window: int = 5, soft: bool = False):
        self.m      = int(m)
        self.window = int(window)
        self.soft   = bool(soft)

    def route(self, nis_arr, c, likelihoods):
        mu = self._native_route(nis_arr, c, likelihoods)
        return mu if mu is not None else self._reference_route(nis_arr, c, likelihoods)

    def _native_key(self):
        if self.m < 0 or self.window < 0:
            return None
        return (self.m, self.window, self.soft)

    def _native_rows(self):
        return self.window if self.window > 0 else None

    def _native_create(self):
        from ..native import core
        return core.create_router("threshold", m=self.m, window=self.window, soft=self.soft), (), []

    def _reference_route(self, nis_arr, c, likelihoods):
        if len(nis_arr) == 0:                           # nothing observed yet
            return np.full(len(c), 1.0 / len(c))
        block    = nis_arr[-self.window:]
        nis_mean = block.mean(axis=0)
        dist     = np.abs(nis_mean - self.m)

        if self.soft:
            return self._normalise(1.0 / (dist + 1e-6))

        mu = np.zeros(len(dist))
        mu[dist.argmin()] = 1.0
        return mu

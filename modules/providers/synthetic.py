"""Synthetic provider — generate truth + measurements from a model (KalmanSampler)."""
from __future__ import annotations
import numpy as np

from .base import KalmanProvider
from ..framework.constants import KalmanProviderKind


class SyntheticProvider(KalmanProvider):
    """Wraps a model (or a ``KalmanSpec`` dict) and samples it on demand."""

    NAME = "Synthetic Provider"
    DESCRIPTION = "Generate truth + measurements from a model via KalmanSampler."
    VERSION = "1.0"
    KIND = KalmanProviderKind.SYNTHETIC

    def __init__(self, model=None, *, spec=None, **kw):
        super().__init__(**kw)
        self._model = model
        self._spec = spec

    def _resolve_model(self):
        if self._model is not None:
            return self._model
        from ..api import make_model
        self._model = make_model(self._spec)
        return self._model

    def fetch(self, duration=1.0, **kw):
        model = self._resolve_model()
        rec = model.sample(float(duration), **kw)
        return {"zk": np.asarray(rec.zk, dtype=float),
                "xk": np.asarray(rec.xk, dtype=float),
                "truth": np.asarray(rec.xk, dtype=float),
                "time": np.asarray(rec.time, dtype=float),
                "meta": {"provider": "synthetic", "duration": float(duration)}}

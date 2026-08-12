"""Sensor provider — a live streaming source of measurements.

``stream()`` yields measurement rows as they arrive from any iterable, generator,
or zero-arg callback (returning a row or ``None`` to stop).  Compose with the
filter's streaming append (``forward(reset=False, buffer=...)``).
"""
from __future__ import annotations
import numpy as np

from .base import KalmanProvider
from ..framework.constants import KalmanProviderKind


class SensorProvider(KalmanProvider):
    """Wrap a live source (iterable / generator / callback) of measurement rows."""

    NAME = "Sensor Provider"
    DESCRIPTION = "Live streaming measurement feed (iterable / generator / callback)."
    VERSION = "1.0"
    KIND = KalmanProviderKind.SENSOR

    MEASUREMENTS = (
        {"name": "zk", "units": "as delivered",
         "description": "Whatever the wrapped source yields, row per item — the "
                        "provider does not interpret it, so the units are the "
                        "sensor's own."},
    )

    REQUIRES = ("a live source= (iterable, generator, or callback)",)

    def __init__(self, source=None, *, max_items=None, **kw):
        super().__init__(cache=False, **kw)
        self.source = source
        self.max_items = max_items

    def stream(self, source=None, max_items=None, **kw):
        src = source if source is not None else self.source
        cap = max_items if max_items is not None else self.max_items
        if src is None:
            raise ValueError("SensorProvider needs a source.")
        n = 0
        if callable(src):
            while cap is None or n < cap:
                row = src()
                if row is None:
                    break
                yield np.asarray(row, dtype=float)
                n += 1
        else:
            for row in src:
                if cap is not None and n >= cap:
                    break
                yield np.asarray(row, dtype=float)
                n += 1

    def fetch(self, source=None, max_items=None, **kw):
        """Drain the stream into a stacked ``zk`` array (bounded by ``max_items``)."""
        rows = list(self.stream(source=source, max_items=max_items, **kw))
        zk = np.array(rows, dtype=float) if rows else np.empty((0, 0))
        return {"zk": zk, "meta": {"provider": "sensor", "n": len(rows)}}

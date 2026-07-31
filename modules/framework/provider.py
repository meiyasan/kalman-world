# kalman/framework/provider.py
"""
Abstract **provider** base — the data/knowledge-source role ("WHERE the data
comes from").  Lives in ``framework/`` with every other base contract; concrete
providers (synthetic/file/sensor/remote and the domain ones like weather's
ERA5/turbulence) live outside it and register into ``kalman.providers``.

A provider implements ``fetch`` (pull) and/or ``stream`` (live) and returns a
plain dict payload — some of ``{"zk", "xk", "truth", "time", "meta"}`` — so any
transport (facade/MCP/REST) can carry it as JSON.  ``measurements``/``truth``
are conveniences that pull the filter/OSSE-ready arrays out of the payload; the
base offers a tiny query-keyed cache.
"""
from __future__ import annotations
import numpy as np

from .abstract.metadata import KalmanProviderMetadata


class KalmanProvider(KalmanProviderMetadata):

    NAME = "Kalman Provider"
    DESCRIPTION = "Abstract data/knowledge provider (fetch / stream)."
    VERSION = "1.0"
    KIND = None

    def __init__(self, *, cache: bool = True):
        self._cache = {} if cache else None

    # ------------------------------------------------------------------
    def fetch(self, **query):
        """Pull a dataset for ``query`` → payload dict.  Override."""
        raise NotImplementedError(f"{self.name()} does not support fetch().")

    def stream(self, **query):
        """Yield live data items.  Override on streaming providers."""
        raise NotImplementedError(f"{self.name()} is not a streaming provider.")

    def _cached(self, key, produce):
        if self._cache is None:
            return produce()
        if key not in self._cache:
            self._cache[key] = produce()
        return self._cache[key]

    # ------------------------------------------------------------------
    def measurements(self, **query):
        """Filter-ready measurements ``zk`` from the payload."""
        p = self.fetch(**query)
        z = p.get("zk", p.get("measurements"))
        if z is None:
            raise KeyError(f"{self.name()} payload has no measurements.")
        return np.asarray(z, dtype=float)

    def truth(self, **query):
        """OSSE-ready truth trajectory from the payload."""
        p = self.fetch(**query)
        x = p.get("truth", p.get("xk"))
        if x is None:
            raise KeyError(f"{self.name()} payload has no truth.")
        return np.asarray(x, dtype=float)

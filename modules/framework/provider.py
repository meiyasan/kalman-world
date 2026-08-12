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

    #: What this provider can actually supply, declared the way models declare
    #: their params — each entry ``{"name":.., "units":.., "description":..}``.
    #:
    #: A provider that only advertises its *name* is half-discoverable: a caller
    #: can see that ``era5`` exists but not that it yields 500 hPa wind, and has
    #: to read the source to find out.  Models are introspectable through
    #: ``describe()``; measurements are the other half of the same question, so
    #: they are declared here and harvested by ``kalman.api.describe_provider``.
    MEASUREMENTS: tuple = ()

    #: Named preconditions — Python packages, credentials, or a local file the
    #: caller must supply.  Stated so a fetch that cannot possibly work says why
    #: before it tries (ERA5 needs ``cdsapi`` and CDS credentials, and finding
    #: that out from an ImportError mid-download is nobody's idea of discovery).
    REQUIRES: tuple = ()

    def __init__(self, *, cache: bool = True):
        self._cache = {} if cache else None

    # ------------------------------------------------------------------
    @classmethod
    def describe(cls) -> dict:
        """Everything discoverable about this provider, as plain JSON-able data.

        Harvests the declared metadata plus the real ``fetch`` signature, so the
        query keys a caller must pass are read off the code rather than kept in
        a docstring that can drift away from it."""
        import inspect
        try:
            sig = inspect.signature(cls.fetch)
            query = [{"name": name,
                      "required": p.default is inspect.Parameter.empty
                                  and p.kind is not inspect.Parameter.VAR_KEYWORD,
                      "default": None if p.default is inspect.Parameter.empty
                                 else p.default}
                     for name, p in sig.parameters.items()
                     if name not in ("self", "kw", "kwargs")]
        except (TypeError, ValueError):
            query = []

        kind = getattr(cls, "KIND", None)
        return {
            "name": getattr(cls, "NAME", cls.__name__),
            "kind": getattr(kind, "name", str(kind)).lower() if kind else None,
            "class": cls.__name__,
            "description": getattr(cls, "DESCRIPTION", ""),
            "version": getattr(cls, "VERSION", ""),
            "measurements": [dict(m) for m in getattr(cls, "MEASUREMENTS", ())],
            "requires": list(getattr(cls, "REQUIRES", ())),
            "query": query,
            "streams": cls.stream is not KalmanProvider.stream,
        }

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

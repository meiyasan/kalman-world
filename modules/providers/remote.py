"""Remote provider — fetch measurements from an HTTP endpoint.

Guarded on ``httpx``.  Caches responses (so a repeat query hits no network),
enforces a minimum inter-request interval (rate-limit), and injects an
**outbound** bearer credential resolved server-side via
:mod:`kalman.providers.credentials` — never from a caller's token.  Fails closed
if a credential is required but missing.
"""
from __future__ import annotations
import time
import numpy as np

from .base import KalmanProvider
from ..framework.constants import KalmanProviderKind
from .credentials import get_credential


class RemoteProvider(KalmanProvider):

    NAME = "Remote Provider"
    DESCRIPTION = "Fetch measurements from a remote HTTP endpoint (cached, rate-limited, authed)."
    VERSION = "1.0"
    KIND = KalmanProviderKind.REMOTE

    def __init__(self, base_url=None, *, credential_name=None, min_interval=0.0,
                 field="zk", client=None, **kw):
        super().__init__(**kw)
        self.base_url = base_url
        self.credential_name = credential_name     # outbound secret (optional)
        self.min_interval = float(min_interval)
        self.field = field
        self._client = client                      # injectable (tests pass a mock)
        self._last = 0.0

    def _http(self):
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as e:
            raise ImportError("RemoteProvider needs httpx (pip install 'kalman[remote]').") from e
        return httpx.Client(timeout=30.0)

    def _headers(self):
        if not self.credential_name:
            return {}
        tok = get_credential(self.credential_name, required=True)   # fail-closed
        return {"Authorization": f"Bearer {tok}"}

    def fetch(self, path="", *, params=None, **kw):
        url = (self.base_url or "").rstrip("/") + ("/" + path.lstrip("/") if path else "")
        key = (url, tuple(sorted((params or {}).items())))

        def _do():
            gap = time.monotonic() - self._last
            if self.min_interval and gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            resp = self._http().get(url, params=params, headers=self._headers())
            resp.raise_for_status()
            self._last = time.monotonic()
            data = resp.json()
            zk = data.get(self.field, data) if isinstance(data, dict) else data
            return {"zk": np.asarray(zk, dtype=float),
                    "meta": {"provider": "remote", "url": url}}

        return self._cached(key, _do)

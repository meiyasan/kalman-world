"""Where the MCP tools send their work: this process, or the REST API.

  - :class:`LocalBackend` calls :mod:`kalman.api` in-process and checks each
    scope here. Models and plugins are still shared through the registry file
    (:mod:`kalman.api.store`), but runs stay in this process.
  - :class:`RemoteBackend` (``KALMAN_API_URL`` set) forwards every call to the
    REST API with the caller's bearer token. The REST API is then the one
    authority: it checks the token, owns the registry and keeps the runs, so a
    ``run_id`` from an agent works in Unity and the other way round. This
    process needs no JWT secret and never loads native code.

Both expose the same verbs, so :mod:`.server` does not care which one it has.
"""
from __future__ import annotations
import hashlib
import os
import time
from urllib.parse import quote


class LocalBackend:
    kind = "local"

    def __init__(self):
        from .. import api
        from ..api.auth import authorize
        self.api, self._authorize = api, authorize

    def _scope(self, token, scope):
        self._authorize(token or None, scope)

    # ---- read ----
    def whoami(self, token):
        return self.api.whoami(token or None)

    def catalog(self, token):
        self._scope(token, "read"); return self.api.catalog()

    def describe(self, name, token):
        self._scope(token, "read"); return self.api.describe(name)

    def diagnostics(self, run_id, token):
        self._scope(token, "read"); return self.api.diagnostics(run_id)

    def providers(self, token):
        self._scope(token, "read"); return self.api.providers()

    def describe_provider(self, name, token):
        self._scope(token, "read"); return self.api.describe_provider(name)

    def models(self, token):
        self._scope(token, "read"); return self.api.models()

    def sources(self, token):
        self._scope(token, "read"); return self.api.sources()

    def plugins(self, token):
        self._scope(token, "read"); return self.api.plugins()

    def events(self, since, limit, token):
        self._scope(token, "read"); return self.api.events(since, limit)

    # ---- send_measurement ----
    def run_filter(self, spec, data, source, query, smooth, token):
        self._scope(token, "send_measurement")
        return self.api.run_filter(spec, data, smooth=smooth, source=source, query=query,
                                   credential=token or None)

    def forecast(self, spec, x0, P0, steps, token):
        self._scope(token, "send_measurement"); return self.api.forecast(spec, x0, P0, steps)

    def run_osse(self, spec, n_cycles, n_members, token):
        self._scope(token, "send_measurement")
        return self.api.run_osse(spec, n_cycles=n_cycles, n_members=n_members)

    def run_control(self, model, controller, duration, token):
        self._scope(token, "send_measurement")
        return self.api.run_control(model, controller, duration)

    def fetch(self, provider, query, token):
        return self.api.fetch(provider, query or {}, credential=token or None)

    def push_measurements(self, name, rows, time_, token):
        return self.api.push_measurements(name, rows, time_, credential=token or None)

    # ---- write_models / load_plugins ----
    def register_model(self, spec, name, replace, token):
        return self.api.register_model(spec, name, replace=replace, credential=token or None)

    def delete_model(self, name, token):
        return self.api.delete_model(name, credential=token or None)

    def register_source(self, name, provider, replace, token):
        return self.api.register_source(name, provider, replace=replace,
                                        credential=token or None)

    def load_plugin(self, name, token):
        return self.api.load_plugin(name, credential=token or None)

    def scaffold(self, spec, class_name, token):
        return {"source": self.api.scaffold(spec, class_name, credential=token or None)}


class RemoteError(Exception):
    """The REST API answered with an error; the message is its ``detail``."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


class RemoteBackend:
    kind = "remote"

    #: How long a verified token is trusted before /whoami is asked again.
    WHOAMI_TTL = 30.0

    def __init__(self, base_url: str, *, timeout: float | None = None, client=None):
        if client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError("KALMAN_API_URL needs httpx (pip install 'kalman[remote]').") from e
            client = httpx.Client(base_url=base_url.rstrip("/"),
                                  timeout=timeout or float(os.environ.get("KALMAN_API_TIMEOUT", "300")))
        self.base_url = base_url.rstrip("/")
        self._http = client
        self._whoami: dict = {}

    def _call(self, method, path, token, *, json=None, params=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = self._http.request(method, path, json=json, params=params, headers=headers)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            if r.status_code in (401, 403):
                from ..api.auth import AuthError
                raise AuthError(detail)
            raise RemoteError(r.status_code, str(detail))
        return r.json()

    def health(self):
        return self._call("GET", "/healthz", None)

    # ---- read ----
    def whoami(self, token):
        key = hashlib.sha256((token or "").encode()).hexdigest()
        hit = self._whoami.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        info = self._call("GET", "/whoami", token)
        if len(self._whoami) > 1024:
            self._whoami.clear()
        self._whoami[key] = (time.monotonic() + self.WHOAMI_TTL, info)
        return info

    def catalog(self, token):
        return self._call("GET", "/catalog", token)

    def describe(self, name, token):
        return self._call("GET", f"/describe/{quote(name, safe='')}", token)

    def diagnostics(self, run_id, token):
        return self._call("GET", f"/diagnostics/{quote(run_id, safe='')}", token)

    def providers(self, token):
        return self._call("GET", "/providers", token)

    def describe_provider(self, name, token):
        return self._call("GET", f"/providers/{quote(name, safe='')}", token)

    def models(self, token):
        return self._call("GET", "/models", token)

    def sources(self, token):
        return self._call("GET", "/sources", token)

    def plugins(self, token):
        return self._call("GET", "/plugins", token)

    def events(self, since, limit, token):
        return self._call("GET", "/events", token, params={"since": since, "limit": limit})

    # ---- send_measurement ----
    def run_filter(self, spec, data, source, query, smooth, token):
        body = {"spec": spec, "smooth": smooth}
        if data is not None:
            body["data"] = data
        if source is not None:
            body["source"], body["query"] = source, query or {}
        return self._call("POST", "/filter", token, json=body)

    def forecast(self, spec, x0, P0, steps, token):
        return self._call("POST", "/forecast", token,
                          json={"spec": spec, "x0": x0, "P0": P0, "steps": steps})

    def run_osse(self, spec, n_cycles, n_members, token):
        return self._call("POST", "/osse", token, json={
            "spec": spec, "options": {"n_cycles": n_cycles, "n_members": n_members}})

    def run_control(self, model, controller, duration, token):
        return self._call("POST", "/control", token,
                          json={"model": model, "controller": controller, "duration": duration})

    def fetch(self, provider, query, token):
        body = {"source": provider} if isinstance(provider, str) else {"provider": provider}
        body["query"] = query or {}
        return self._call("POST", "/fetch", token, json=body)

    def push_measurements(self, name, rows, time_, token):
        body = {"rows": rows}
        if time_ is not None:
            body["time"] = time_
        return self._call("POST", f"/sources/{quote(name, safe='')}/measurements", token,
                          json=body)

    # ---- write_models / load_plugins ----
    def register_model(self, spec, name, replace, token):
        return self._call("POST", "/models", token,
                          json={"spec": spec, "name": name, "replace": replace})

    def delete_model(self, name, token):
        return self._call("DELETE", f"/models/{quote(name, safe='')}", token)

    def register_source(self, name, provider, replace, token):
        return self._call("POST", "/sources", token,
                          json={"name": name, "provider": provider, "replace": replace})

    def load_plugin(self, name, token):
        return self._call("POST", "/plugins", token, json={"name": name})

    def scaffold(self, spec, class_name, token):
        return self._call("POST", "/scaffold", token,
                          json={"spec": spec, "class_name": class_name})


def from_env():
    """``RemoteBackend`` when ``KALMAN_API_URL`` is set, else ``LocalBackend``."""
    url = os.environ.get("KALMAN_API_URL")
    return RemoteBackend(url) if url else LocalBackend()

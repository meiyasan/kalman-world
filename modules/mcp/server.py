"""MCP server over the :mod:`kalman.api` facade — one thin adapter.

Tools mirror the facade verbs, each declaring a required **scope**. Where the
work happens is the backend's business (:mod:`.backend`): in this process, or
— with ``KALMAN_API_URL`` set — in the REST API, which then owns the registry
and the runs for everybody. Apps need no per-app MCP code — they're
auto-cataloged; an authored spec is indistinguishable from a built-in.  The
``mcp`` SDK is imported lazily.

The caller's token, first match wins:

  1. the ``token`` argument of the tool (clients that cannot set headers);
  2. the ``Authorization: Bearer`` header of the HTTP transports (``sse``,
     ``streamable-http``), checked by :class:`JwtVerifier` before any tool runs;
  3. on stdio only, ``KALMAN_MCP_TOKEN`` — the token the client was registered
     with (``tools/mcp_install.py --token``).
"""
from __future__ import annotations
import os


def _server_class():
    """The SDK's decorator-style server class.

    MCP SDK 2.x exposes it as ``mcp.server.MCPServer``; 1.x called the same
    thing ``mcp.server.fastmcp.FastMCP``.  Support both so the server works
    against whichever generation is installed."""
    try:
        from mcp.server import MCPServer          # SDK >= 2.0
        return MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP    # SDK 1.x
        return FastMCP


SERVER_NAME = "Kalman-AI"          # identifier: also the tool-name prefix and the
                                   # config key most clients display in their list
SERVER_TITLE = "Kalman-AI"         # human-facing name shown by clients


class JwtVerifier:
    """Bearer-token check for the HTTP transports (the SDK's ``TokenVerifier``).

    Asks the backend who the token belongs to — verified locally with
    ``KALMAN_JWT_SECRET``, or by the REST API's ``/whoami`` — and refuses the
    connection when the token is missing, forged or expired. Scopes are then
    checked per tool."""

    def __init__(self, backend):
        self.backend = backend

    async def verify_token(self, token: str):
        import anyio
        from mcp.server.auth.provider import AccessToken
        try:
            info = await anyio.to_thread.run_sync(self.backend.whoami, token)
        except Exception:
            return None
        if not info.get("user"):
            return None
        return AccessToken(token=token, client_id=info["user"], subject=info["user"],
                           scopes=list(info.get("scopes", [])), expires_at=info.get("exp"))


def _transport_token():
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token
    except ImportError:
        return None
    at = get_access_token()
    return at.token if at is not None else None


def build_server(backend=None, *, transport: str = "stdio", auth_settings=None,
                 token_verifier=None):
    """The MCP server. ``backend`` defaults to :func:`.backend.from_env`;
    ``auth_settings``/``token_verifier`` turn on bearer auth for HTTP transports
    (see ``__main__``)."""
    from .backend import from_env
    backend = backend or from_env()
    service_token = os.environ.get("KALMAN_MCP_TOKEN") if transport == "stdio" else None

    def tok(token: str) -> str | None:
        return token or _transport_token() or service_token or None

    cls = _server_class()
    kw = {}
    if token_verifier is not None:
        kw = {"token_verifier": token_verifier, "auth": auth_settings}
    try:                            # SDK 2.x carries a display title/description
        mcp = cls(SERVER_NAME, title=SERVER_TITLE,
                  description="Kalman estimation framework: models, filters, "
                              "solvers, providers, and spec authoring.", **kw)
    except TypeError:               # SDK 1.x FastMCP takes the name only
        mcp = cls(SERVER_TITLE, **kw)

    # ---- read ----
    @mcp.tool()
    def list_models(token: str = "") -> dict:
        """List catalog models / engines / integrators / solvers / controllers /
        providers, the authored models in the shared registry, and the named
        measurement sources."""
        return backend.catalog(tok(token))

    @mcp.tool()
    def describe_model(name: str, token: str = "") -> dict:
        """Describe a model or controller: params, dims, bindings. For an
        authored model: its spec and owner."""
        return backend.describe(name, tok(token))

    @mcp.tool()
    def list_registered_models(token: str = "") -> dict:
        """The shared registry's authored models, with owner, dimensions and
        timestamps — what other people and agents have published."""
        return backend.models(tok(token))

    @mcp.tool()
    def get_diagnostics(run_id: str, token: str = "") -> dict:
        """Filter-consistency diagnostics (NIS + innovation) for a stored run."""
        return backend.diagnostics(run_id, tok(token))

    @mcp.tool()
    def list_providers(token: str = "") -> dict:
        """List data providers with the measurements each can supply, the query
        keys fetch_data takes, and any credentials/packages they need."""
        return backend.providers(tok(token))

    @mcp.tool()
    def describe_provider(name: str, token: str = "") -> dict:
        """Describe one provider (synthetic/era5/turbulence/file/sensor/remote):
        its measurements, query keys, and prerequisites."""
        return backend.describe_provider(name, tok(token))

    @mcp.tool()
    def list_sources(token: str = "") -> dict:
        """Named measurement sources: push buffers that devices, simulations or
        Unity write rows into, and server-side providers set up by an admin.
        Fetch one with fetch_data(provider="<name>") or filter it directly with
        run_filter(source="<name>")."""
        return backend.sources(tok(token))

    @mcp.tool()
    def list_events(since: int = 0, limit: int = 100, token: str = "") -> dict:
        """Registry events after `since` (model.created/replaced/deleted,
        plugin.loaded, source.*, measurement.received). Pass the returned
        `last` as `since` next time to see only what is new."""
        return backend.events(since, limit, tok(token))

    # ---- send_measurement ----
    @mcp.tool()
    def run_filter(spec: dict, data: list = None, source: str = None, query: dict = None,
                   smooth: bool = False, token: str = "") -> dict:
        """Filter measurements through a KalmanSpec's filter: give `data`
        (a list of rows), or `source` (a registered source name, with an
        optional `query`) to measure and filter in one call."""
        return backend.run_filter(spec, data, source, query, smooth, tok(token))

    @mcp.tool()
    def forecast(spec: dict, x0: list, P0: list = None, steps: int = 1, token: str = "") -> dict:
        """Forecast ahead from a state: a track cone where the model defines one,
        else the propagated state + covariance."""
        return backend.forecast(spec, x0, P0, steps, tok(token))

    @mcp.tool()
    def run_osse(spec: dict, n_cycles: int = 40, n_members: int = 16, token: str = "") -> dict:
        """Twin-experiment OSSE for a flow spec (EnKF vs free run)."""
        return backend.run_osse(spec, n_cycles, n_members, tok(token))

    @mcp.tool()
    def run_control(model: dict, controller: dict, duration: float, token: str = "") -> dict:
        """Close the loop: run a controller over a model for a duration."""
        return backend.run_control(model, controller, duration, tok(token))

    @mcp.tool()
    def fetch_data(provider: dict | str, query: dict = None, token: str = "") -> dict:
        """Pull data: a registered source by name ("lab-1", or {"source":
        "lab-1"}), or a raw provider {"kind": "synthetic"|..., "init": {...}}.
        Raw file/remote/sensor providers need the admin scope."""
        return backend.fetch(provider, query or {}, tok(token))

    @mcp.tool()
    def push_measurements(source: str, rows: list, time: list = None, token: str = "") -> dict:
        """Append measurement rows to a push source; subscribers get a
        measurement.received event."""
        return backend.push_measurements(source, rows, time, tok(token))

    # ---- write_models ----
    @mcp.tool()
    def register_model(spec: dict, name: str = None, replace: bool = False,
                       token: str = "") -> dict:
        """Validate + register an authored KalmanSpec in the shared registry,
        where every client (REST, Unity, other agents) can use it as a spec
        base. The same spec again is a no-op; a different one under a taken
        name needs replace=true and ownership."""
        return backend.register_model(spec, name, replace, tok(token))

    @mcp.tool()
    def delete_model(name: str, token: str = "") -> dict:
        """Remove an authored model from the shared registry (owner or admin)."""
        return backend.delete_model(name, tok(token))

    @mcp.tool()
    def register_source(name: str, provider: dict, replace: bool = False,
                        token: str = "") -> dict:
        """Set up a named measurement source: {"kind": "push", "m": <width>} for
        a buffer clients write into, or {"kind": "<provider>", "init": {...}}
        (file/remote/sensor need admin)."""
        return backend.register_source(name, provider, replace, tok(token))

    # ---- load_plugins ----
    @mcp.tool()
    def load_plugin(name: str, token: str = "") -> dict:
        """Load a compiled (C++) model library by *name* from the server's allow-listed
        plugin directories (KALMAN_PLUGIN_PATH); its models then appear in list_models
        and can be used as a spec base, in every process sharing the registry. Paths
        are refused unless inside those directories."""
        return backend.load_plugin(name, tok(token))

    @mcp.tool()
    def scaffold_app(spec: dict, class_name: str = None, token: str = "") -> dict:
        """Emit reviewable modules/apps/<name>.py source from a spec (human commits)."""
        return backend.scaffold(spec, class_name, tok(token))

    # ---- resources ----
    @mcp.resource("catalog://models")
    def catalog_models() -> dict:
        return {"models": backend.catalog(tok(""))["models"]}

    return mcp

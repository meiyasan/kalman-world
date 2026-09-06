"""MCP server over the :mod:`kalman.api` facade — one thin adapter.

Tools mirror the facade verbs, each declaring a required **scope** enforced by
:func:`kalman.api.auth.authorize` (a ``token`` argument, empty when auth is
disabled).  Apps need no per-app MCP code — they're auto-cataloged; an authored
spec is indistinguishable from a built-in.  The ``mcp`` SDK is imported lazily.
"""
from __future__ import annotations


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


def build_server():
    from .. import api as facade
    from ..api.auth import authorize

    cls = _server_class()
    try:                            # SDK 2.x carries a display title/description
        mcp = cls(SERVER_NAME, title=SERVER_TITLE,
                  description="Kalman estimation framework: models, filters, "
                              "solvers, providers, and spec authoring.")
    except TypeError:               # SDK 1.x FastMCP takes the name only
        mcp = cls(SERVER_TITLE)

    # ---- read ----
    @mcp.tool()
    def list_models(token: str = "") -> dict:
        """List catalog models / engines / integrators / solvers / controllers / providers."""
        authorize(token or None, "read")
        return facade.catalog()

    @mcp.tool()
    def describe_model(name: str, token: str = "") -> dict:
        """Describe a model or controller: params, dims, bindings."""
        authorize(token or None, "read")
        return facade.describe(name)

    @mcp.tool()
    def get_diagnostics(run_id: str, token: str = "") -> dict:
        """Filter-consistency diagnostics (NIS + innovation) for a stored run."""
        authorize(token or None, "read")
        return facade.diagnostics(run_id)

    @mcp.tool()
    def list_providers(token: str = "") -> dict:
        """List data providers with the measurements each can supply, the query
        keys fetch_data takes, and any credentials/packages they need."""
        authorize(token or None, "read")
        return facade.providers()

    @mcp.tool()
    def describe_provider(name: str, token: str = "") -> dict:
        """Describe one provider (synthetic/era5/turbulence/file/sensor/remote):
        its measurements, query keys, and prerequisites."""
        authorize(token or None, "read")
        return facade.describe_provider(name)

    # ---- send_measurement ----
    @mcp.tool()
    def run_filter(spec: dict, data: list, token: str = "") -> dict:
        """Filter a measurement set through a KalmanSpec's filter."""
        authorize(token or None, "send_measurement")
        return facade.run_filter(spec, data)

    @mcp.tool()
    def forecast(spec: dict, x0: list, P0: list = None, steps: int = 1, token: str = "") -> dict:
        """Forecast ahead from a state: a track cone where the model defines one,
        else the propagated state + covariance."""
        authorize(token or None, "send_measurement")
        return facade.forecast(spec, x0, P0, steps)

    @mcp.tool()
    def run_osse(spec: dict, n_cycles: int = 40, n_members: int = 16, token: str = "") -> dict:
        """Twin-experiment OSSE for a flow spec (EnKF vs free run)."""
        authorize(token or None, "send_measurement")
        return facade.run_osse(spec, n_cycles=n_cycles, n_members=n_members)

    @mcp.tool()
    def run_control(model: dict, controller: dict, duration: float, token: str = "") -> dict:
        """Close the loop: run a controller over a model for a duration."""
        authorize(token or None, "send_measurement")
        return facade.run_control(model, controller, duration)

    @mcp.tool()
    def fetch_data(provider: dict, query: dict = None, token: str = "") -> dict:
        """Pull data from a provider (synthetic/file/era5/turbulence/remote/sensor)."""
        authorize(token or None, "send_measurement")
        return facade.fetch(provider, query or {}, credential=token or None)

    # ---- write_models ----
    @mcp.tool()
    def register_model(spec: dict, name: str = None, token: str = "") -> dict:
        """Validate + register an authored KalmanSpec (session)."""
        return facade.register_model(spec, name, credential=token or None)

    # ---- load_plugins ----
    @mcp.tool()
    def load_plugin(name: str, token: str = "") -> dict:
        """Load a compiled (C++) model library by *name* from the server's allow-listed
        plugin directories (KALMAN_PLUGIN_PATH); its models then appear in list_models
        and can be used as a spec base. Paths are refused unless inside those directories."""
        return facade.load_plugin(name, credential=token or None)

    @mcp.tool()
    def scaffold_app(spec: dict, class_name: str = None, token: str = "") -> dict:
        """Emit reviewable modules/apps/<name>.py source from a spec (human commits)."""
        return {"source": facade.scaffold(spec, class_name, credential=token or None)}

    # ---- resources ----
    @mcp.resource("catalog://models")
    def catalog_models() -> dict:
        return {"models": facade.catalog()["models"]}

    return mcp

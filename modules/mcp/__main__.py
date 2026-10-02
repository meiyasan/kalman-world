"""Run the Kalman MCP server.  ``python -m kalman.mcp`` / ``kalman-mcp``.

stdio locally; set ``KALMAN_MCP_TRANSPORT=sse`` or ``streamable-http`` (+
``KALMAN_MCP_HOST/PORT``, default ``127.0.0.1:8765``) to serve over HTTP. Set
``KALMAN_API_URL`` to make this server a client of the REST API, which then
owns the registry, the runs and the token checks.

Over HTTP, clients send ``Authorization: Bearer <JWT>``. That is required
whenever tokens can be checked — ``KALMAN_JWT_SECRET`` here, or a REST API that
reports auth on — and a server on a network interface refuses to start with
neither (see :func:`kalman.api.auth.check_server`).

The ``mcp`` SDK is optional — absent, we print an install hint and exit 0 so
the core is never blocked.
"""
from __future__ import annotations
import os
import sys


def _auth_for_http(backend, host: str, port: int):
    """``(auth_settings, verifier)`` when tokens can be checked, else ``(None, None)``."""
    from ..api.auth import enabled
    remote_auth = False
    if backend.kind == "remote":
        try:
            remote_auth = bool(backend.health().get("auth"))
        except Exception as e:      # fail closed: no API, no server
            raise RuntimeError(f"KALMAN_API_URL={backend.base_url} is not reachable: {e}")
    if not (enabled() or remote_auth):
        return None, None
    from mcp.server.auth.settings import AuthSettings
    from .server import JwtVerifier
    public = os.environ.get("KALMAN_MCP_PUBLIC_URL") or f"http://{host}:{port}"
    issuer = os.environ.get("KALMAN_JWT_ISSUER", "")
    if not issuer.startswith(("http://", "https://")):
        issuer = os.environ.get("KALMAN_API_URL") or public
    return AuthSettings(issuer_url=issuer, resource_server_url=public), JwtVerifier(backend)


def main(argv=None):
    try:
        import mcp  # noqa
        from .server import build_server
        from .backend import from_env
    except ImportError:
        print("The MCP server needs the MCP SDK:\n    pip install 'kalman[mcp]'", file=sys.stderr)
        return 0
    from ..api.auth import check_server, AuthError
    transport = os.environ.get("KALMAN_MCP_TRANSPORT", "stdio")
    backend = from_env()
    if transport == "stdio":
        build_server(backend, transport="stdio").run(transport="stdio")
        return 0

    host = os.environ.get("KALMAN_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("KALMAN_MCP_PORT", "8765"))
    try:
        auth_settings, verifier = _auth_for_http(backend, host, port)
        check_server(host, service="MCP server", verified_elsewhere=verifier is not None)
    except (AuthError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    server = build_server(backend, transport=transport, auth_settings=auth_settings,
                          token_verifier=verifier)
    try:
        server.run(transport=transport, host=host, port=port)       # SDK >= 2.0
    except TypeError:                                              # SDK 1.x: settings
        server.settings.host, server.settings.port = host, port
        server.run(transport=transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

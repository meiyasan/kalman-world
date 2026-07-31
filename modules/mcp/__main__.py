"""Run the Kalman MCP server.  ``python -m kalman.mcp`` / ``kalman-mcp``.

stdio locally; set ``KALMAN_MCP_TRANSPORT=sse`` (+ ``KALMAN_MCP_HOST/PORT``) in a
container.  The ``mcp`` SDK is optional — absent, we print an install hint and
exit 0 so the core is never blocked.
"""
from __future__ import annotations
import os
import sys


def main(argv=None):
    try:
        import mcp  # noqa
        from .server import build_server
    except ImportError:
        print("The MCP server needs the MCP SDK:\n    pip install 'kalman[mcp]'", file=sys.stderr)
        return 0
    server = build_server()
    transport = os.environ.get("KALMAN_MCP_TRANSPORT", "stdio")
    server.run(transport=transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

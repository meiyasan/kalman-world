"""Run the Kalman REST API (uvicorn).  ``python -m kalman.rest`` / ``kalman-rest``.

Listens on ``KALMAN_REST_HOST:KALMAN_REST_PORT`` (default ``127.0.0.1:8000``).
On any other interface it refuses to start without ``KALMAN_JWT_SECRET`` —
see :func:`kalman.api.auth.check_server`.
"""
from __future__ import annotations
import os
import sys


def main(argv=None):
    try:
        import uvicorn  # noqa
        from .app import create_app
    except ImportError:
        print("The REST API needs FastAPI + uvicorn:\n    pip install 'kalman[rest]'",
              file=sys.stderr)
        return 0
    import uvicorn
    from ..api.auth import check_server, AuthError
    host = os.environ.get("KALMAN_REST_HOST", "127.0.0.1")
    port = int(os.environ.get("KALMAN_REST_PORT", "8000"))
    try:
        check_server(host, service="REST API")
    except AuthError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

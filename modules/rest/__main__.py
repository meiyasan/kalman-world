"""Run the Kalman REST API (uvicorn).  ``python -m kalman.rest`` / ``kalman-rest``."""
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
    host = os.environ.get("KALMAN_REST_HOST", "0.0.0.0")
    port = int(os.environ.get("KALMAN_REST_PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

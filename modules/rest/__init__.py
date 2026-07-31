"""``kalman.rest`` — FastAPI transport over the ``kalman.api`` facade (optional dep)."""
from .app import create_app

__all__ = ["create_app"]

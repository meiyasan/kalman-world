"""Back-compat shim — this module moved to ``kalman.apps.weather.geo``
as part of the apps domain-package reorganisation.  Import from there (or from
``kalman.apps`` / ``kalman.apps.weather``); this shim will be removed.
"""
from .weather.geo import *          # noqa: F401,F403
from .weather import geo as _mod

__all__ = getattr(_mod, "__all__", [n for n in dir(_mod) if not n.startswith("_")])

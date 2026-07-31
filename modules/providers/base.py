"""Back-compat shim — the ``KalmanProvider`` base now lives in ``framework/``
(all base contracts do).  Import it from there."""
from ..framework.provider import KalmanProvider

__all__ = ["KalmanProvider"]

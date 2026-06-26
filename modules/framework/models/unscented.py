"""Unscented model — nonlinear f/h intended for the sigma-point (UKF) engine."""
from __future__ import annotations

from ..model import KalmanModel
from ..constants import KalmanEngineKind, ALL_ENGINES


class UnscentedKalmanModel(KalmanModel):
    """
    Nonlinear model (same math contract as :class:`ExtendedKalmanModel`) that defaults
    to the :class:`~kalman.engines.unscented.UnscentedKalmanEngine`. Disallows the
    linear engine; composable with EKF / PF.
    """

    NAME = "Unscented Kalman Model"
    DESCRIPTION = "Nonlinear state-space model intended for the sigma-point (UKF) engine."
    VERSION = "1.0"

    DEFAULT_ENGINE  = KalmanEngineKind.UNSCENTED
    ALLOWED_ENGINES = ALL_ENGINES - {KalmanEngineKind.LINEAR}

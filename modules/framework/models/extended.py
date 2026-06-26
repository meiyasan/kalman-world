"""Extended model — nonlinear f/h; pins the EKF engine, forbids the linear engine."""
from __future__ import annotations

from ..model import KalmanModel
from ..constants import KalmanEngineKind, ALL_ENGINES


class ExtendedKalmanModel(KalmanModel):
    """
    Nonlinear model: override ``f`` / ``h`` (Jacobians numerical by default, or supply
    analytic ``F`` / ``H``). Defaults to the
    :class:`~kalman.engines.extended.ExtendedKalmanEngine` and **disallows** the linear
    engine (which would ignore the nonlinearity). Still composable with UKF / PF.
    """

    NAME = "Extended Kalman Model"
    DESCRIPTION = "Nonlinear state-space model (override f/h); numerical/analytic Jacobians."
    VERSION = "1.0"

    DEFAULT_ENGINE  = KalmanEngineKind.EXTENDED
    ALLOWED_ENGINES = ALL_ENGINES - {KalmanEngineKind.LINEAR}

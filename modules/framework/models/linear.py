"""Linear model — constant-matrix system; pins the linear engine."""
from __future__ import annotations

from ..model import KalmanModel
from ..constants import KalmanEngineKind, ALL_ENGINES


class LinearKalmanModel(KalmanModel):
    """
    A model defined by **constant matrices** ``F/H/B/L/M`` (guaranteed linear).

    Thin specialization of :class:`KalmanModel`: requires the matrices and defaults to
    the :class:`~kalman.engines.linear.LinearKalmanEngine`. Linear dynamics are safe on
    any engine, so all engines remain allowed.
    """

    NAME = "Linear Kalman Model"
    DESCRIPTION = "Linear state-space model: f(x)=Fx+Bu+Lw, h(x)=Hx+Mv (constant matrices)."
    VERSION = "1.0"

    DEFAULT_ENGINE  = KalmanEngineKind.LINEAR
    ALLOWED_ENGINES = ALL_ENGINES

    def __init__(self, sensor_id, **kwargs):
        has_F = kwargs.get("F") is not None or kwargs.get("A") is not None
        if not has_F or kwargs.get("H") is None:
            raise ValueError("LinearKalmanModel requires H= and either F= or A= (guaranteed-linear model).")
        super().__init__(sensor_id, **kwargs)

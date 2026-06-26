"""Linear Kalman Filter (KF) — wraps a model with the linear engine."""
from ..framework.filter import KalmanFilter
from ..framework.constants import KalmanEngineKind


class LinearKalmanFilter(KalmanFilter):
    """
    Standard linear Kalman filter.

    Compose with any model::

        kf  = LinearKalmanFilter(OscillatorModel("s", fs=100))
        rec = kf.forward(zk)
        rec_s = kf.backward()
    """

    NAME = "Linear Kalman Filter"
    DESCRIPTION = "Linear Kalman filter (KF): linear engine over a model."
    KIND = KalmanEngineKind.LINEAR

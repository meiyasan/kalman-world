"""Extended Kalman Filter (EKF) — wraps a model with the Jacobian engine."""
from ..filter import KalmanFilter
from ..constants import KalmanEngineKind


class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman filter (Jacobian linearisation). Runs on nonlinear models and
    reduces to the linear computation on matrix models::

        ekf = ExtendedKalmanFilter(BallisticModel("gps", fs=10))
        rec = ekf.forward(zk)
    """

    NAME = "Extended Kalman Filter"
    DESCRIPTION = "Extended Kalman filter (EKF): Jacobian engine over a model."
    KIND = KalmanEngineKind.EXTENDED

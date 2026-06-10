import numpy as np
from scipy.linalg import expm

from ..models.linear import KalmanLinearModel
from ..framework.constants import KalmanState, KalmanObservable


class SuspensionState(KalmanState):
    SPRUNG_DISP   = "m"     # sprung mass displacement (car body)
    SPRUNG_VEL    = "m/s"   # sprung mass velocity
    UNSPRUNG_DISP = "m"     # unsprung mass displacement (wheel/axle)
    UNSPRUNG_VEL  = "m/s"   # unsprung mass velocity


class SuspensionObservable(KalmanObservable):
    SPRUNG_DISP = "m"       # measured by proximity / laser sensor


class SuspensionModel(KalmanLinearModel):
    """
    Quarter-car linear suspension model (2-DOF).

    State vector (axis-major order):
        x = [z_s, ż_s, z_u, ż_u]

    where:
        z_s  — sprung mass displacement   (car body)
        ż_s  — sprung mass velocity
        z_u  — unsprung mass displacement (wheel + axle)
        ż_u  — unsprung mass velocity

    Continuous-time dynamics:
        z̈_s =  -(k_s/m_s)(z_s - z_u) - (b_s/m_s)(ż_s - ż_u)
        z̈_u =   (k_s/m_u)(z_s - z_u) + (b_s/m_u)(ż_s - ż_u) - (k_t/m_u)z_u + (k_t/m_u)u

    The road input u is a displacement excitation (control input, l=1).
    Only sprung displacement z_s is observed → H = [1, 0, 0, 0].

    Parameters
    ----------
    sensor_id : str
        Sensor / channel name.
    m_s : float
        Sprung mass [kg], default 300.
    m_u : float
        Unsprung mass [kg], default 50.
    k_s : float
        Suspension spring stiffness [N/m], default 20 000.
    b_s : float
        Suspension damper coefficient [Ns/m], default 1 500.
    k_t : float
        Tyre stiffness [N/m], default 180 000.
    measurement_uncertainty : float
        Measurement noise std [m], default 0.001 (1 mm).
    process_uncertainty : float
        Process noise std scaling, default 0.001.
    fs : float
        Sample rate [Hz], default 256.
    """

    NAME = "Quarter-Car Suspension Model"
    DESCRIPTION = (
        "Linear 2-DOF quarter-car suspension model. "
        "States: [z_s, dz_s, z_u, dz_u]. Observable: sprung displacement z_s. "
        "Control input: road displacement u."
    )
    VERSION = "1.0"

    @property
    def states(self):
        return SuspensionState

    @property
    def observables(self):
        return SuspensionObservable

    def __init__(self, sensor_id: str,
                 m_s: float = 300.0,
                 m_u: float = 50.0,
                 k_s: float = 20_000.0,
                 b_s: float = 1_500.0,
                 k_t: float = 180_000.0,
                 measurement_uncertainty: float = 0.001,
                 process_uncertainty: float = 0.001,
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", 256.0)
        dt = 1.0 / fs

        # ------------------------------------------------------------------
        # Continuous-time system matrix A  (4×4)
        # x = [z_s, ż_s, z_u, ż_u]
        # ------------------------------------------------------------------
        A = np.array([
            [0,           1,            0,          0],
            [-k_s / m_s, -b_s / m_s,   k_s / m_s,  b_s / m_s],
            [0,           0,            0,          1],
            [k_s / m_u,   b_s / m_u,  -(k_s + k_t) / m_u, -b_s / m_u],
        ], dtype=self.dtype)

        # Discrete state-transition via matrix exponential
        F = expm(A * dt)

        # ------------------------------------------------------------------
        # Continuous-time input matrix B  (4×1)  [road excitation]
        # ------------------------------------------------------------------
        B_c = np.array([[0], [0], [0], [k_t / m_u]], dtype=self.dtype)
        # Discretise: B_d = A^{-1}(F - I) B_c  (zero-order hold)
        try:
            B_d = np.linalg.solve(A, (F - np.eye(4, dtype=self.dtype)) @ B_c)
        except np.linalg.LinAlgError:
            B_d = dt * B_c  # fallback: Euler approximation

        # ------------------------------------------------------------------
        # Measurement matrix (only z_s observed)
        # ------------------------------------------------------------------
        H = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=self.dtype)

        # ------------------------------------------------------------------
        # Noise covariances
        # ------------------------------------------------------------------
        R = (measurement_uncertainty ** 2) * np.eye(1, dtype=self.dtype)

        q = process_uncertainty ** 2
        Q = q * np.eye(4, dtype=self.dtype)

        # ------------------------------------------------------------------
        # Initial covariance (unit-appropriate)
        # ------------------------------------------------------------------
        if kwargs.get("P0") is None:
            kwargs["P0"] = np.diag([
                measurement_uncertainty ** 2,          # z_s  uncertainty
                (measurement_uncertainty * 10) ** 2,   # ż_s  uncertainty
                measurement_uncertainty ** 2,          # z_u  uncertainty
                (measurement_uncertainty * 10) ** 2,   # ż_u  uncertainty
            ]).astype(self.dtype)

        super().__init__(sensor_id, F=F, H=H, Q=Q, R=R, B=B_d, fs=fs, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, xk_hat, **kwargs):
        return xk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat

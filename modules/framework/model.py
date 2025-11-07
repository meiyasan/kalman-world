from typing import Optional
import numpy as np
from ..helpers import set_random_seed, as_col, numerical_jacobian
from .abstract.mixins import KalmanModelMixin
from .abstract.metadata import KalmanModelMetadata

class KalmanModel(KalmanModelMetadata, KalmanModelMixin):
    """
    A generic model container holding dimensions (n, m, l), noise covariances (Q, R),
    and initial state (x0, P0). It does NOT assume linearity; engines (Linear/EKF/UKF)
    consume the f/h/jacobians contract on subclasses.
    """

    NAME = "Kalman Model"
    DESCRIPTION = "Generic model for Kalman-family filters; provide f/h/jacobians in subclasses."
    VERSION = "1.1"

    def __init__(self, sensor_id : str | int, **kwargs):

        self.add_sensor(sensor_id)

        # --- Sampling rate ---
        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", None)
        if self.fs is not None:
            if not isinstance(self.fs, (int, float)) or self.fs <= 0:
                raise ValueError(f"Sampling frequency must be a positive number, got {self.fs!r}")

        # --- RNG seed (optional) ---
        seed = kwargs.pop("random_seed", None)
        if seed is not None:
            set_random_seed(seed)

        # --- Time base ---
        self.t0 = kwargs.pop("t0", 0.0)

        # --- Infer dimensions ---
        self.n = self._infer_n()
        self.m = self._infer_m()
        self.l = self._infer_l()

        # --- Attach default engine ---
        from .filter import KalmanFilter
        engine = kwargs.pop("engine", KalmanFilter)
        if not isinstance(engine, type):
            raise ValueError("engine parameter must be a class, e.g., KalmanFilter, LinearKalmanFilter, etc.")
        self.attach(engine(self))

        # --- Infer covariances ---
        self._Q = self._infer_Q(kwargs.pop("Q", None))
        self._R = self._infer_R(kwargs.pop("R", None))

        # --- Initial state ---
        x0 = kwargs.pop("x0", None)
        if x0 is None: x0 = np.zeros((self.n, 1))

        P0 = kwargs.pop("P0", None)
        self.initial_state(x0, P0)

        self._check_shapes(strict=False)

    # ============================================================
    # Jacobians (numerical defaults)
    # ============================================================
    def F(self, x: np.ndarray = None, u: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """State-transition Jacobian: ∂f/∂x."""
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda xx: self.f(xx, u=u, w=None, t=t), as_col(x))

    def H(self, x: np.ndarray = None, t: Optional[float] = None) -> np.ndarray:
        """Observation Jacobian: ∂h/∂x."""
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda xx: self.h(xx, v=None, t=t), as_col(x))

    def B(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Control-input Jacobian: ∂f/∂u (if u provided)."""
        if u is None:
            return None
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda uu: self.f(as_col(x), u=uu, w=None, t=t), as_col(u))

    def L(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None,
           w: Optional[np.ndarray] = None, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Process-noise Jacobian: ∂f/∂w (if w provided)."""
        if w is None:
            return None
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda ww: self.f(as_col(x), u=u, w=ww, t=t), as_col(w))

    def M(self, x: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None,
           t: Optional[float] = None) -> Optional[np.ndarray]:
        """Measurement-noise Jacobian: ∂h/∂v (if v provided)."""
        if v is None:
            return None
        x = self.x0 if x is None else x
        return numerical_jacobian(lambda vv: self.h(as_col(x), v=vv, t=t), as_col(v))

    # Unified interface returning all Jacobians
    def jacobians(self, x: np.ndarray,
                        u: Optional[np.ndarray] = None,
                        v: Optional[np.ndarray] = None,
                        w: Optional[np.ndarray] = None,
                        t: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        
        """Return (F, H, B, L, M) in a consistent order."""
        F = self.F(x, u, t)
        H = self.H(x, t)
        B = self.B(x, u, t)
        L = self.L(x, u, w, t)
        M = self.M(x, v, t)
        return F, H, B, L, M

    # ============================================================
    # Covariances
    # ============================================================
    def Q(self, t: Optional[float] = None) -> np.ndarray:
        """Process noise covariance."""
        return self._Q(t) if callable(self._Q) else self._Q

    def R(self, t: Optional[float] = None) -> np.ndarray:
        """Measurement noise covariance."""
        return self._R(t) if callable(self._R) else self._R

    def covariances(self, t: Optional[float] = None,
                          x: Optional[np.ndarray] = None,
                          u: Optional[np.ndarray] = None,
                          w: Optional[np.ndarray] = None,
                          v: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (Q, R)."""
        Q = self.Q(t)
        L = self.L(x, u, w, t)
        if L is not None:
            Q = L @ Q @ L.T
        
        R = self.R(t)
        M = self.M(x, v, t)
        if M is not None:
            R = M @ R @ M.T

        return Q, R
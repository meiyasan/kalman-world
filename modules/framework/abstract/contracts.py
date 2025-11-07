from abc import abstractmethod
from typing import Optional

from ..constants import KalmanState, KalmanStateUnits
from ..io.record import KalmanStateRecord

import numpy as np

# =========================
# Abstract Base Model
# =========================
class KalmanModelABC():
    """
    Abstract base class for Kalman models.
    Subclasses must implement f, h, and optionally override Jacobians / covariances.
    """

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------
    @property
    @abstractmethod
    def type(self):
        return KalmanState

    @property
    @abstractmethod
    def units(self):
        return KalmanStateUnits

    # ------------------------------------------------------
    # Data processing hooks
    # ------------------------------------------------------
    @abstractmethod
    def preprocess(self, zk, uk: Optional[np.ndarray] = None, **kwargs):
        """Prepare measurements/control (batch). Return (zk, uk)."""
        ...

    @abstractmethod
    def inprocess(self, zk_hat, **kwargs):
        """Transform state vectors before mapping to observations."""
        ...

    @abstractmethod
    def postprocess(self, zk_hat, **kwargs):
        """Transform predicted observations (e.g., unit conversions)."""
        ...

    # ------------------------------------------------------------
    # Required model type metadata
    # ------------------------------------------------------------
    @property
    @abstractmethod
    def type(self) -> KalmanState:
        return KalmanState

    @property
    @abstractmethod
    def units(self) -> KalmanStateUnits:
        return KalmanStateUnits

    # ------------------------------------------------------------
    # Core model equations
    # ------------------------------------------------------------
    @abstractmethod
    def f(self, x: np.ndarray, u: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """State transition: xₖ₊₁ = f(xₖ, uₖ, wₖ, tₖ)."""
        ...

    @abstractmethod
    def h(self, x: np.ndarray, v: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """Observation mapping: zₖ = h(xₖ, vₖ, tₖ)."""
        ...

    # ------------------------------------------------------------
    # Jacobians / linear mappings (x optional!)
    # ------------------------------------------------------------
    @abstractmethod
    def F(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """∂f/∂x |_(x,u,t) or predefined F(t)."""
        ...

    @abstractmethod
    def H(self, x: Optional[np.ndarray] = None, t: Optional[float] = None) -> np.ndarray:
        """∂h/∂x |_(x,t) or predefined H(t)."""
        ...

    @abstractmethod
    def B(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Control-input mapping ∂f/∂u |_(x,u,t) or predefined B(t)."""
        ...

    @abstractmethod
    def L(self, x: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Process-noise influence matrix ∂f/∂w |_(x,u,w,t) or predefined L(t)."""
        ...

    @abstractmethod
    def M(self, x: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None, t: Optional[float] = None) -> Optional[np.ndarray]:
        """Measurement-noise influence matrix ∂h/∂v |_(x,v,t) or predefined M(t)."""
        ...

    # ------------------------------------------------------------
    # Noise covariances (Q, R)
    # ------------------------------------------------------------
    @abstractmethod
    def Q(self, t: Optional[float] = None) -> np.ndarray:
        """Process noise covariance (time-varying if callable)."""
        ...

    @abstractmethod
    def R(self, t: Optional[float] = None) -> np.ndarray:
        """Measurement noise covariance (time-varying if callable)."""
        ...

class KalmanEngineABC():

    """
    Abstract base class for Kalman engines.
    Subclasses must implement _predict / _update methods.
    """
    @abstractmethod
    def _predict(self, x, P, u = None, w = None, **kwargs):
        """Perform prediction step."""
        ...
    @abstractmethod
    def _update(self, z):
        """Perform update step."""
        ...

    def forward(self, z, u = None, **kwargs) -> 'KalmanEngineABC | KalmanStateRecord':
        """Perform forward filtering over given measurements and controls."""
        ...
    def backward(self, **kwargs) -> KalmanStateRecord:
        """Perform backward smoothing over stored states."""
        ...
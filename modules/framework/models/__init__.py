"""
Generic Kalman model bases (the *what* — system definition).

The flexible contract is :class:`kalman.framework.model.KalmanModel`; these are thin
specializations that pin an intended engine and (for nonlinear ones) forbid the linear
engine. Domain applications live in top-level ``apps`` and subclass these.
"""
from .linear import LinearKalmanModel
from .extended import ExtendedKalmanModel
from .unscented import UnscentedKalmanModel
from .compose import Composite
from .operation import OperationModel
from .joint import Joint
from .lag import Lag
from .fusion import Fusion
from .native import NativeModel, native_model_classes

__all__ = [
    "LinearKalmanModel",
    "ExtendedKalmanModel",
    "UnscentedKalmanModel",
    "Composite",
    "OperationModel",
    "Joint",
    "Lag",
    "Fusion",
    "NativeModel",
    "native_model_classes",
]

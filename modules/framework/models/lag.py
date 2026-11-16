"""
``lag``: a model that carries its last ``L`` states along with the present one.

The Python face of ``Kalman::Models::Lag`` (``include/Kalman/Models/Lag.h``), which
does the work. The state is the inner model's, ``L + 1`` times over,

    X_k = [ x_k ; x_{k-1} ; ... ; x_{k-L} ]

each step computing the new present state and shifting the others down. Two
things come of it, with no change to any filter:

- a sensor with a ``delay`` reads the state as it was (``KalmanSensor(...,
  delay=0.3)``): the update corrects that past state and, through the
  covariance between the blocks, the present one;
- block ``j`` of the estimate is the state ``j`` steps ago given everything
  measured since: fixed-lag smoothing, as the filter runs.

Three ways to the same model::

    model.lag(3)
    lag(model, 3)
    Lag(model, 3)

The sensors are added first, then the model is lagged. Nothing is computed here.
"""
from __future__ import annotations

from ..constants import KalmanState
from ..model import KalmanModel
from ..native import core
from .operation import OperationModel


class Lag(OperationModel):
    """``model`` carrying its last ``depth`` states."""

    NAME = "Lag Model"
    DESCRIPTION = "A model that carries its last states: for sensors that report late, and fixed-lag smoothing."
    VERSION = "1.0"

    def __init__(self, model: KalmanModel, depth: int, **kwargs):
        if not isinstance(model, KalmanModel):
            raise TypeError(f"lag: the model must be a KalmanModel, got {type(model).__name__}.")
        if isinstance(depth, bool) or int(depth) != depth or int(depth) < 1:
            raise ValueError(f"lag: carry at least one past state (a whole number of steps), got {depth!r}.")
        lib, ffi = core.require(), core.ffi
        self.inner, self.depth, self.n_inner = model, int(depth), int(model.n)
        self._inner_cxx = inner = self._inner(model)               # borrowed by the C++ model
        handle = lib.kalman_model_create_lag(inner, self.depth)
        if handle == ffi.NULL:
            raise self._failed("lag")
        self._adopt(handle, [model], **kwargs)

    def block(self, steps: int = 0) -> slice:
        """Where the state ``steps`` ago sits in this model's state:
        ``record.xk[:, model.block(3)]`` is the state three steps before each row,
        given the measurements up to that row."""
        if not 0 <= int(steps) <= self.depth:
            raise IndexError(f"lag: the model carries {self.depth} past states, not {steps}.")
        return slice(int(steps) * self.n_inner, (int(steps) + 1) * self.n_inner)

    @property
    def states(self) -> KalmanState:
        """The inner state components, then the same ``@-1``, ``@-2`` ...: as the C++ model names them."""
        if getattr(self, "_states_enum", None) is None:
            self._states_enum = self._label_enum(True, KalmanState, "LagState")
        return self._states_enum

    def __repr__(self):
        return f"Lag({type(self.inner).__name__}, depth={self.depth}, n={self.n}, m={self.m}, fs={self.fs})"

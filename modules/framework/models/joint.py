"""
``joint``: a model whose state also carries some of another model's parameters.

The Python face of ``Kalman::Models::Joint`` (``include/Kalman/Models/Joint.h``),
which does the work: it appends the chosen parameters to the inner model's state,
as constants that may drift a little,

    x = [ s ; theta ]                 s: the inner state (n),  theta: p parameters

    s_{k+1}     = f_inner(s_k, u_k, t ; theta_k)
    theta_{k+1} = theta_k + w_theta,  w_theta ~ N(0, diag(drift**2))
    z_k         = h_inner(s_k, t ; theta_k)

sets the inner parameters before every evaluation and puts them back after, and
assembles the Jacobian by blocks (the inner model's own for the motion, the
parameter columns differenced). See the header for the details. Nothing is
computed here.

Three ways to the same model::

    model.joint({"c_drag": (1e-3, 1e-6)})
    joint(model, {"c_drag": (1e-3, 1e-6)})
    Joint(model, {"c_drag": (1e-3, 1e-6)})
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from ..constants import KalmanState
from ..model import KalmanModel
from ..native import core
from .operation import OperationModel


def _spec(name, value):
    """``(sigma, drift)`` from a tuple or a mapping. The values are checked in C++."""
    if isinstance(value, Mapping):
        unknown = set(value) - {"sigma", "drift"}
        if unknown:
            raise ValueError(f"joint: {name!r} takes sigma= and drift=, not {sorted(unknown)}.")
        if "sigma" not in value:
            raise ValueError(f"joint: {name!r} needs sigma=.")
        return float(value["sigma"]), float(value.get("drift", 0.0))
    try:
        sigma, drift = value
    except (TypeError, ValueError):
        raise ValueError(f"joint: {name!r} needs (sigma, drift) or {{'sigma': ..., 'drift': ...}}, "
                         f"got {value!r}.") from None
    return float(sigma), float(drift)


class Joint(OperationModel):
    """Estimate parameters of ``model`` along with its state.

    ``Joint(model, {"c_drag": (1e-3, 1e-6)})`` appends the inner model's
    ``c_drag`` to its state, with an initial standard deviation of ``1e-3`` and a
    random walk of ``1e-6`` per step, starting from the value the inner model
    holds. A parameter is one the model keeps as metadata (``model["c_drag"]``)
    or one its C++ twin declares.

    ``initial_state`` takes either the joint state (``n + p`` numbers) or one the
    inner model takes; the latter is handed to the inner model, and the
    parameters start again from the values it holds.
    """

    NAME = "Joint Model"
    DESCRIPTION = "A model whose state also carries some of its parameters, estimated with the motion."
    VERSION = "1.0"

    def __init__(self, model: KalmanModel, parameters: Mapping, **kwargs):
        if not isinstance(model, KalmanModel):
            raise TypeError(f"joint: the inner model must be a KalmanModel, got {type(model).__name__}.")
        if not parameters:
            raise ValueError("joint: name at least one parameter to estimate.")
        lib, ffi = core.require(), core.ffi
        names = list(parameters)
        specs = [_spec(name, parameters[name]) for name in names]
        self.inner, self.names = model, names
        self.estimated = dict(zip(names, specs))
        self.n_inner = int(model.n)

        self._inner_cxx = inner = self._inner(model, names)        # borrowed by the C++ model
        encoded = [ffi.new("char[]", name.encode()) for name in names]
        sigma = np.ascontiguousarray([s for s, _ in specs], dtype=np.float64)
        drift = np.ascontiguousarray([d for _, d in specs], dtype=np.float64)
        handle = lib.kalman_model_create_joint(inner, len(names), ffi.new("const char *[]", encoded),
                                               core.cdptr(sigma), core.cdptr(drift))
        if handle == ffi.NULL:
            raise self._failed("joint")

        self._adopt(handle, [model], **kwargs)

    @property
    def states(self) -> KalmanState:
        """The inner state components, then the parameters: as the C++ model names them."""
        if getattr(self, "_states_enum", None) is None:
            self._states_enum = self._label_enum(True, KalmanState, "JointState")
        return self._states_enum

    def __repr__(self):
        return f"Joint({type(self.inner).__name__}, {self.names}, n={self.n}, m={self.m}, l={self.l}, fs={self.fs})"

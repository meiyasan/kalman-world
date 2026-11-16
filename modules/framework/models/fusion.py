"""
``add_sensor``: a model read by one more sensor.

The Python face of ``Kalman::Models::Fusion`` (``include/Kalman/Models/Fusion.h``):
the state and its dynamics are the inner model's, and the measurement grows by
the channels of a :class:`~kalman.framework.sensor.KalmanSensor`,

    z = [ h_inner(x, t) ; g(x, t) ]          R = diag(R_inner, R_sensor)

One filter update then weighs every sensor by its noise: that is sensor fusion.
A model of matrices with a matrix sensor added is still a model of matrices.
Nothing is computed here.

Three ways to the same model::

    model.add_sensor(sensor)
    add_sensor(model, sensor)
    Fusion(model, sensor)

This is not ``compose``, which puts models with their own states side by side:
here a second instrument reads the same state.
"""
from __future__ import annotations

import numpy as np

from ..constants import KalmanObservable
from ..model import KalmanModel
from ..native import core
from ..sensor import KalmanSensor, _timed
from .operation import OperationModel


class Fusion(OperationModel):
    """``model`` read by one more ``sensor``."""

    NAME = "Fusion Model"
    DESCRIPTION = "A model read by one more sensor on the same state."
    VERSION = "1.0"

    def __init__(self, model: KalmanModel, sensor: KalmanSensor, *, instrument=None, **kwargs):
        if not isinstance(model, KalmanModel):
            raise TypeError(f"add_sensor: the model must be a KalmanModel, got {type(model).__name__}.")
        if not isinstance(sensor, KalmanSensor):
            raise TypeError(f"add_sensor: a KalmanSensor is expected, got {type(sensor).__name__}.")
        lib, ffi = core.require(), core.ffi
        n = int(model.n)
        self.inner, self.sensor = model, sensor
        k, R, _ = sensor.resolved(n, model.x0, float(model.t0 or 0.0))
        labels = sensor.labels or []                               # none: C++ names them after the sensor
        # A sensor carried by an instrument takes its rate when it states none.
        rate = float(sensor.rate or (instrument.rate if instrument is not None else None) or 0.0)
        self.n_extra = k

        self._inner_cxx = inner = self._inner(model)               # borrowed by the C++ model
        encode = lambda texts: [ffi.new("char[]", text.encode("utf-8", "replace")) for text in texts]
        names, units = encode([name for name, _ in labels]), encode([unit for _, unit in labels])
        cnames = ffi.new("const char *[]", names) if labels else ffi.NULL
        cunits = ffi.new("const char *[]", units) if labels else ffi.NULL
        if sensor.is_linear:
            handle = lib.kalman_model_add_sensor_matrix(inner, sensor.name.encode(), k, sensor.G.shape[1],
                                                        core.cdptr(sensor.G), core.cdptr(R), cnames, cunits, rate)
        else:
            errors: list[BaseException] = []
            state = lambda x: np.frombuffer(ffi.buffer(x, n * 8), dtype=np.float64).reshape(n, 1).copy()

            def guard(fn, size):
                def wrapped(user, x, t, out):
                    try:
                        values = np.asarray(fn(state(x), t), dtype=np.float64).reshape(-1)
                        if values.size != size:
                            raise ValueError(f"add_sensor: the sensor {sensor.name!r} returned {values.size} "
                                             f"values, {size} expected.")
                        ffi.memmove(out, core.cdptr(np.ascontiguousarray(values)), size * 8)
                        return 0
                    except BaseException as exc:            # must not unwind through C++
                        errors.append(exc)
                        return 1
                return wrapped

            signature = "int(void *, const double *, double, double *)"
            measure = ffi.callback(signature)(guard(_timed(sensor.g), k))
            dg = (ffi.callback(signature)(guard(_timed(sensor.jacobian), k * n))
                  if sensor.jacobian is not None else ffi.NULL)
            self._keeps.append((measure, dg, errors))
            handle = lib.kalman_model_add_sensor(inner, sensor.name.encode(), k, measure, dg, core.cdptr(R),
                                                 cnames, cunits, rate, ffi.NULL)
        if handle == ffi.NULL:
            raise self._failed("add_sensor")
        if sensor.delay:
            core.check(lib.kalman_model_set_sensor_delay(handle, float(sensor.delay)), "set_sensor_delay")
        if instrument is not None:
            # Its name, and its calibration: the C++ model then declares the
            # parameters "<instrument>.gain" and "<instrument>.offset".
            core.check(lib.kalman_model_set_instrument(handle, instrument.name.encode(), int(instrument.calibrated),
                                                       float(instrument.gain), float(instrument.offset)),
                       "set_instrument")

        self._adopt(handle, [model], **kwargs)

    # ------------------------------------------------------------------
    # Labels: the C++ model's
    # ------------------------------------------------------------------
    @property
    def sensors(self):
        """The inner model's sensors, then the added one (``Model::Sensors()``)."""
        return KalmanModel.sensors.fget(self)

    @sensors.setter
    def sensors(self, value):
        raise AttributeError("the sensors of a model are set by add_sensor().")

    @property
    def states(self):
        return self.inner.states

    @property
    def observables(self) -> KalmanObservable:
        """The observables of its sensors, in the order of ``z``: as the C++ model names them."""
        if getattr(self, "_observables_enum", None) is None:
            self._observables_enum = self._label_enum(False, KalmanObservable, "FusionObservable")
        return self._observables_enum

    def __repr__(self):
        return (f"Fusion({type(self.inner).__name__}, sensors={[s.name for s in self.sensors]}, "
                f"n={self.n}, m={self.m}, l={self.l}, fs={self.fs})")

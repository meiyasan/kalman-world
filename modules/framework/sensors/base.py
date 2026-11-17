"""A sensor written in C++: the Python object holds it and asks it everything."""
from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np

from ..native import core
from ..sensor import KalmanSensor


class _Settings(MutableMapping):
    """The settings of a C++ sensor, read and written in place (``Kalman::Sensor::settings``)."""

    def __init__(self, handle):
        self._handle = handle

    def _names(self):
        lib = core.lib
        count = core.check(lib.kalman_sensor_setting_count(self._handle), "sensor_setting_count")
        return [core._s(lib.kalman_sensor_setting_name(self._handle, i)) for i in range(count)]

    def __getitem__(self, name):
        value = core.ffi.new("double *")
        if core.lib.kalman_sensor_get_setting(self._handle, str(name).encode(), value) < 0:
            raise KeyError(name)
        return float(value[0])

    def __setitem__(self, name, value):
        if core.lib.kalman_sensor_set_setting(self._handle, str(name).encode(), float(value)) < 0:
            raise KeyError(name)

    def __delitem__(self, name):
        raise TypeError("the settings of a sensor are fixed by its kind; set their values.")

    def __iter__(self):
        return iter(self._names())

    def __len__(self):
        return len(self._names())

    def __repr__(self):
        return repr(dict(self))


class NativeSensor(KalmanSensor):
    """A :class:`KalmanSensor` whose readings, Jacobian and noise are a C++ sensor's
    (``Kalman::SensorFactory``): nothing is computed in Python.

    ``kind`` and ``numbers`` are what the C++ kind is built from; a subclass turns
    its own arguments into them and documents what it reads. What does not fit is
    refused by the C++ sensor, and its message is raised as a ``ValueError``.
    """

    def __init__(self, kind, name, n, numbers, *, labels=None, rate=None, delay=None):
        lib, ffi = core.require(), core.ffi
        numbers = np.ascontiguousarray(numbers, dtype=np.float64).reshape(-1)
        handle = lib.kalman_sensor_create(kind.encode(), str(name).encode(), int(n), core.cdptr(numbers), numbers.size)
        if handle == ffi.NULL:
            raise ValueError(core.last_error())
        handle = ffi.gc(handle, lib.kalman_sensor_free)
        self.n = int(n)
        k = core.check(lib.kalman_sensor_channels(handle), "sensor_channels")
        R = np.zeros((k, k))
        core.check(lib.kalman_sensor_R(handle, core.dptr(R)), "sensor_R")
        G = np.zeros((k, self.n))
        if core.check(lib.kalman_sensor_G(handle, core.dptr(G)), "sensor_G"):
            super().__init__(name, G=G, R=R, labels=labels, rate=rate, delay=delay)
        else:
            super().__init__(name, self._read, R=R, jacobian=self._jacobian, labels=labels, rate=rate, delay=delay)
        self.native = handle
        self._k = k

    def _state(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        if x.size != self.n:
            raise ValueError(f"{type(self).__name__} {self.name!r} reads a state of {self.n} components, got {x.size}.")
        return x

    def _read(self, x, t=0.0):
        z = np.zeros(self._k)
        core.check(core.lib.kalman_sensor_read(self.native, core.cdptr(self._state(x)), float(t or 0.0), core.dptr(z)),
                   "sensor_read")
        return z

    def _jacobian(self, x, t=0.0):
        J = np.zeros((self._k, self.n))
        core.check(core.lib.kalman_sensor_jacobian(self.native, core.cdptr(self._state(x)), float(t or 0.0),
                                                   core.dptr(J)), "sensor_jacobian")
        return J

    @property
    def settings(self):
        """The numbers its readings depend on, by name, read and written in the C++
        sensor; ``None`` when it has none. A model that took the sensor shares them."""
        settings = _Settings(self.native)
        return settings if len(settings) else None

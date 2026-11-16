"""
A sensor: what acquires a measurement.

The Python face of ``Kalman::Sensor`` (``include/Kalman/Sensor.h``), with the same
fields. A sensor reads ``k`` channels of a model's state,

    z_sensor = g(x, t) + v,      v ~ N(0, R)

through a function ``g`` or a matrix ``G`` (``g(x) = G x``). Its channels are
named: each is an *observable*, a quantity that can be measured, with its unit.

    doppler = KalmanSensor("doppler", radial_velocity, R=1.0, labels={"V_RADIAL": "m/s"})
    radar   = shell.add_sensor(doppler)        # a new model: the shell, read by one more sensor
    radar.sensors                              # [position, doppler]

The words, used the same way in both languages: an **observable** is a quantity
that can be measured; a **sensor** measures one or more of them; a
**measurement** is the numbers that come out.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


class KalmanSensor:
    """One sensor.

    Parameters
    ----------
    name : what the sensor is called.
    g : ``g(x)`` or ``g(x, t=...)``, returning its ``k`` readings of the state ``x``
        (a column). Give this or ``G``.
    G : a ``k x n`` matrix, for readings linear in the state.
    R : its noise: a ``k x k`` covariance, ``k`` variances, or one variance for
        every channel.
    labels : its observables: ``{"X": "m", "Y": "m"}`` (name and unit), a list of
        names, or a ``KalmanObservable`` enum. Named after the sensor by default
        (numbered when ``k > 1``).
    jacobian : ``dg/dx`` as ``jacobian(x)`` or ``jacobian(x, t=...)``, ``k x n``;
        differenced when not given.
    rate : how often it reports [Hz, or per unit of the model's time]; ``None``
        for every step of the model. Between two of its reports its channels are
        missing (NaN), which the engines skip: see ``model.measurements``.
    """

    def __init__(self, name, g=None, *, G=None, R=None, labels=None, jacobian=None, rate=None):
        if not isinstance(name, str) or not name:
            raise ValueError("KalmanSensor: a sensor needs a name.")
        if (g is None) == (G is None):
            raise ValueError(f"KalmanSensor: the sensor {name!r} needs a function g or a matrix G=, one of them.")
        if R is None:
            raise ValueError(f"KalmanSensor: the sensor {name!r} needs R=, its noise covariance.")
        self.name = name
        self.g = g
        self.G = None if G is None else np.ascontiguousarray(np.asarray(G, dtype=np.float64))
        if self.G is not None and self.G.ndim != 2:
            raise ValueError(f"KalmanSensor: the G of {name!r} must be a k x n matrix.")
        self.R = R
        self.jacobian = jacobian
        if rate is not None and not float(rate) > 0.0:
            raise ValueError(f"KalmanSensor: the rate of {name!r} must be positive.")
        self.rate = None if rate is None else float(rate)
        self.labels = self._labels(labels)

    # ------------------------------------------------------------------
    @staticmethod
    def _labels(labels):
        """``[(name, unit), ...]`` or ``None``."""
        if labels is None:
            return None
        if isinstance(labels, Mapping):
            return [(str(k), "" if v is None else str(v)) for k, v in labels.items()]
        members = getattr(labels, "__members__", None)
        if members is not None:                                  # a KalmanObservable enum
            return [(k, v._value_ if isinstance(v._value_, str) else "") for k, v in members.items()]
        return [(str(k), "") for k in labels]

    @property
    def is_linear(self) -> bool:
        """True when the readings are a matrix."""
        return self.G is not None

    @property
    def channels(self):
        """The number of channels ``k``, or ``None`` while only ``g`` can tell
        (a function with one variance for ``R`` and no labels)."""
        if self.G is not None:
            return int(self.G.shape[0])
        R = np.asarray(self.R, dtype=np.float64)
        if R.ndim >= 1:
            return int(R.shape[0])
        return len(self.labels) if self.labels is not None else None

    def resolved(self, n, x0=None, t0=0.0):
        """``(k, R, labels)`` against a state of ``n`` components: the number of
        channels (asked of ``g`` at ``x0`` when nothing else says), the ``k x k``
        covariance, and the ``(name, unit)`` of each observable it was given
        (``None`` when it names none: the C++ sensor then names them after itself)."""
        k = self.channels
        if k is None:
            x = np.zeros((n, 1)) if x0 is None else np.asarray(x0, dtype=float).reshape(n, 1)
            k = int(np.asarray(self.measure(x, t0)).size)
        R = np.asarray(self.R, dtype=np.float64)
        R = np.ascontiguousarray(R * np.eye(k) if R.ndim == 0 else np.diag(R) if R.ndim == 1 else R)
        return k, R, self.labels

    def measure(self, x, t=None):
        """The readings at state ``x``."""
        if self.G is not None:
            return self.G @ np.asarray(x, dtype=float).reshape(-1, 1)
        return _timed(self.g)(x, t)

    @property
    def observables(self):
        """Its observables as ``[(name, unit), ...]``; ``None`` for a sensor that
        names none and has not been added to a model yet (``model.sensors`` gives
        the names it took)."""
        return self.labels

    def __repr__(self):
        k = self.channels
        what = "G" if self.is_linear else getattr(self.g, "__name__", "g")
        return f"KalmanSensor({self.name!r}, {what}, channels={'?' if k is None else k})"


def _timed(fn):
    """``fn(x, t)`` whether the caller's function takes the time or not."""
    from .engine import KalmanEngine
    if KalmanEngine._accepts_param(fn, "t"):
        return lambda x, t: fn(x, t=t)
    return lambda x, t: fn(x)

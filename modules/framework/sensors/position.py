"""A sensor that reads some components of the state (``Kalman::Sensors::Position``)."""
from __future__ import annotations

import numpy as np

from .base import NativeSensor


class PositionSensor(NativeSensor):
    """Reads some components of the state: ``z = rotation @ x[axes]``.

    Parameters
    ----------
    name : what the sensor is called.
    n : the number of state components of the model it will read.
    axes : the indices of the components it reads.
    sigma : the standard deviation of its noise, the same on every channel and
        given in the sensor's own axes.
    rotation : ``k x k``, for a sensor not aligned with the state's axes: its rows
        are the sensor's axes. ``None``: aligned.
    labels, rate, delay : as for any :class:`~kalman.framework.sensor.KalmanSensor`.

    Its readings are a matrix, so a model of matrices that takes it stays one.
    """

    def __init__(self, name, n, axes, sigma, rotation=None, *, labels=None, rate=None, delay=None):
        axes = np.atleast_1d(axes).astype(float)
        turn = () if rotation is None else np.asarray(rotation, dtype=float).ravel()
        super().__init__("position", name, n, [axes.size, *axes, float(sigma), *turn],
                         labels=labels, rate=rate, delay=delay)

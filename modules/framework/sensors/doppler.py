"""A sensor that reads the velocity along the line of sight (``Kalman::Sensors::Doppler``)."""
from __future__ import annotations

import numpy as np

from .base import NativeSensor


class DopplerSensor(NativeSensor):
    """The velocity along the line of sight, one channel,

        v_r = v . (p - at) / |p - at|

    Parameters
    ----------
    name : what the sensor is called.
    n : the number of state components of the model it will read.
    position, velocity : the indices of the position and of the velocity in the
        state, as many of each (1, 2 or 3 dimensions).
    sigma : the standard deviation of the reading.
    at : where the sensor is; the origin when ``None``.
    labels, rate, delay : as for any :class:`~kalman.framework.sensor.KalmanSensor`.

    Where it is are its settings ``x``, ``y``, ``z`` (as many as dimensions): the
    model that takes it has the parameters ``"<name>.x"``, ..., which ``joint`` can
    estimate. At the sensor itself the line of sight is not defined: the reading
    is 0.
    """

    def __init__(self, name, n, position, velocity, sigma, at=None, *, labels=None, rate=None, delay=None):
        position = np.atleast_1d(position).astype(float)
        velocity = np.atleast_1d(velocity).astype(float)
        where = () if at is None else np.asarray(at, dtype=float).ravel()
        if velocity.size != position.size or (len(where) and len(where) != position.size):
            raise ValueError(f"DopplerSensor {name!r}: position, velocity and where the sensor is take as many "
                             f"numbers each, got {position.size}, {velocity.size} and {len(where)}.")
        super().__init__("doppler", name, n, [position.size, *position, *velocity, float(sigma), *where],
                         labels=labels, rate=rate, delay=delay)

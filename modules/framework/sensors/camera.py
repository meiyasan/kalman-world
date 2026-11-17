"""A sensor that reads the two pixel coordinates of a point in space (``Kalman::Sensors::Camera``)."""
from __future__ import annotations

import numpy as np

from .base import NativeSensor


class CameraSensor(NativeSensor):
    """A pinhole projection, two channels,

        q = rotation @ (p - at)        u = focal * q_x / q_z        v = focal * q_y / q_z

    Parameters
    ----------
    name : what the sensor is called.
    n : the number of state components of the model it will read.
    position : the indices of the point's three coordinates in the state.
    focal : the focal length, in pixels.
    sigma : the standard deviation of each pixel coordinate.
    at : where the camera is; the origin when ``None``.
    rotation : ``3 x 3``, world to camera: its rows are the camera's axes, the
        third the direction it looks in. ``None``: the identity.
    labels, rate, delay : as for any :class:`~kalman.framework.sensor.KalmanSensor`.

    The focal length and where the camera is are its settings ``focal``, ``x``,
    ``y``, ``z``: the model that takes it has the parameters ``"<name>.focal"``,
    ..., which ``joint`` can estimate. A point at or behind the camera has no
    image; its depth is then taken as 1e-6, so the reading stays finite.
    """

    def __init__(self, name, n, position, focal, sigma, at=None, rotation=None, *, labels=None, rate=None,
                 delay=None):
        position = np.atleast_1d(position).astype(float)
        where = np.zeros(3) if at is None else np.asarray(at, dtype=float).ravel()
        turn = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if position.size != 3 or where.size != 3 or turn.shape != (3, 3):
            raise ValueError(f"CameraSensor {name!r}: a point in space has three coordinates, where the camera is "
                             "takes 3 numbers and its rotation is 3 x 3.")
        placed = at is not None or rotation is not None
        super().__init__("camera", name, n, [*position, float(focal), float(sigma),
                                             *((*where, *turn.ravel()) if placed else ())],
                         labels=labels, rate=rate, delay=delay)

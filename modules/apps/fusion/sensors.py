"""Generic sensors: a position reading, a Doppler radar, a camera.

They know a geometry and a noise, and nothing of any product. A given piece of
hardware is an *adapter*: a subclass that fixes what its data sheet and its
mounting say (noise, rate, delay, where it is, its calibration). None is shipped
here; an adapter belongs with whoever uses that hardware.

:class:`~kalman.framework.sensor.KalmanSensor` is the contract; these fill it in
for three readings the applications kept writing by hand. Each says which
components of the state it reads by their indices, so it fits any model that
has them::

    from kalman.apps import PositionSensor, DopplerRadar, Camera

    # a shell with state (px, vx, ax, py, vy, ay, pz, vz, az)
    tracked = shell.add_sensor(PositionSensor("gps", 9, [0, 3, 6], sigma=10.0))
    radar   = tracked.add_instrument(DopplerRadar("doppler", 9, [0, 3, 6], [1, 4, 7], sigma=1.0))
    seen    = radar.add_instrument(Camera("camera", 9, [0, 3, 6], focal=800.0, sigma=1.5))

The radar and the camera are instruments of one sensor, so each can carry a
calibration: ``DopplerRadar(...).calibrate(1.0, 0.0)`` gives the model the
parameters ``doppler.gain`` and ``doppler.offset``, which ``joint`` can estimate.

The readings, their Jacobians and their noise are computed in C++
(``Kalman::Apps::Fusion``, ``apps/include/Kalman/Apps/Fusion/Sensors.h``):
nothing is called back into Python during a pass. The functions here are the
reference the tests compare against, and what runs without libKalmanApps.
"""
from __future__ import annotations

import numpy as np

from ...framework.instrument import KalmanInstrument
from ...framework.sensor import KalmanSensor

__all__ = ["PositionSensor", "DopplerRadar", "Camera"]


def _indices(who, indices, n, count=None):
    out = [int(i) for i in np.atleast_1d(indices)]
    if not out:
        raise ValueError(f"{who}: no state component to read.")
    for i, given in zip(out, np.atleast_1d(indices)):
        if i != given or not 0 <= i < n:
            raise ValueError(f"{who}: component {given} is not in a state of {n}.")
    if count is not None and len(out) != count:
        raise ValueError(f"{who}: {count} state components are expected, got {len(out)}.")
    return out


def _sigma(who, sigma):
    if not float(sigma) > 0.0:
        raise ValueError(f"{who}: sigma must be positive.")
    return float(sigma)


class PositionSensor(KalmanSensor):
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
    labels, rate, delay : as for any :class:`KalmanSensor`.

    Its readings are a matrix, so a model of matrices that takes it stays one.
    """

    def __init__(self, name, n, axes, sigma, rotation=None, *, labels=None, rate=None, delay=None):
        who = f"PositionSensor {name!r}"
        n = int(n)
        axes = _indices(who, axes, n)
        k = len(axes)
        turn = np.eye(k) if rotation is None else np.asarray(rotation, dtype=float)
        if turn.shape != (k, k):
            raise ValueError(f"{who}: the rotation must be {k} x {k}.")
        G = np.zeros((k, n))
        G[:, axes] = turn
        super().__init__(name, G=G, R=[_sigma(who, sigma) * float(sigma)] * k, labels=labels, rate=rate, delay=delay)
        self.axes, self.sigma = axes, float(sigma)
        self.native = ("position", [k, *axes, float(sigma), *(() if rotation is None else turn.ravel())])


class DopplerRadar(KalmanInstrument):
    """A Doppler radar: the velocity along the line of sight, one channel,

        v_r = v . (p - at) / |p - at|

    Parameters
    ----------
    name : what the radar is called.
    n : the number of state components of the model it will read.
    position, velocity : the indices of the position and of the velocity in the
        state, as many of each (1, 2 or 3 dimensions).
    sigma : the standard deviation of the reading.
    at : where the radar is; the origin when ``None``.
    labels, rate, delay : as for any :class:`KalmanSensor`.

    At the radar itself the line of sight is not defined: the reading is 0.
    """

    def __init__(self, name, n, position, velocity, sigma, at=None, *, labels=None, rate=None, delay=None):
        who = f"DopplerRadar {name!r}"
        n = int(n)
        p = _indices(who, position, n)
        v = _indices(who, velocity, n, count=len(p))
        origin = np.zeros(len(p)) if at is None else np.asarray(at, dtype=float).reshape(-1)
        if origin.size != len(p):
            raise ValueError(f"{who}: where the radar is takes {len(p)} numbers.")

        def geometry(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            line = x[p] - origin
            r = float(np.linalg.norm(line))
            u = line / r if r > 0.0 else np.zeros(len(p))
            return x, u, r, float(x[v] @ u)

        def reading(x):
            return [geometry(x)[3]]

        def jacobian(x):
            x, u, r, vr = geometry(x)
            J = np.zeros((1, n))
            if r > 0.0:
                np.add.at(J[0], v, u)
                np.add.at(J[0], p, (x[v] - vr * u) / r)
            return J

        sensor = KalmanSensor(name, reading, R=_sigma(who, sigma) * float(sigma), jacobian=jacobian, labels=labels, delay=delay)
        sensor.native = ("doppler", [len(p), *p, *v, float(sigma), *(() if at is None else origin)])
        super().__init__(name, [sensor], rate=rate)
        self.position, self.velocity, self.sigma, self.at = p, v, float(sigma), origin


class Camera(KalmanInstrument):
    """A pinhole camera: two pixel coordinates of a point in space,

        q = rotation @ (p - at)        u = focal * q_x / q_z        v = focal * q_y / q_z

    Parameters
    ----------
    name : what the camera is called.
    n : the number of state components of the model it will read.
    position : the indices of the point's three coordinates in the state.
    focal : the focal length, in pixels.
    sigma : the standard deviation of each pixel coordinate.
    at : where the camera is; the origin when ``None``.
    rotation : ``3 x 3``, world to camera: its rows are the camera's axes, the
        third the direction it looks in. ``None``: the identity.
    labels, rate, delay : as for any :class:`KalmanSensor`.

    A point at or behind the camera has no image; its depth is then taken as
    :attr:`NEAR`, so the reading stays finite.
    """

    NEAR = 1e-6    #: the smallest depth a point is given

    def __init__(self, name, n, position, focal, sigma, at=None, rotation=None, *, labels=None, rate=None,
                 delay=None):
        who = f"Camera {name!r}"
        n = int(n)
        p = _indices(who, position, n)
        if len(p) != 3:
            raise ValueError(f"{who}: a point in space has three coordinates.")
        if not float(focal) > 0.0:
            raise ValueError(f"{who}: the focal length must be positive.")
        origin = np.zeros(3) if at is None else np.asarray(at, dtype=float).reshape(-1)
        W = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if origin.size != 3:
            raise ValueError(f"{who}: where the camera is takes 3 numbers.")
        if W.shape != (3, 3):
            raise ValueError(f"{who}: the rotation must be 3 x 3.")
        focal = float(focal)

        def seen(x):
            q = W @ (np.asarray(x, dtype=float).reshape(-1)[p] - origin)
            q[2] = max(q[2], self.NEAR)
            return q

        def reading(x):
            q = seen(x)
            return [focal * q[0] / q[2], focal * q[1] / q[2]]

        def jacobian(x):
            q = seen(x)
            J = np.zeros((2, n))
            projection = (focal / q[2]) * np.array([[1.0, 0.0, -q[0] / q[2]], [0.0, 1.0, -q[1] / q[2]]])
            np.add.at(J, (slice(None), p), projection @ W)
            return J

        sensor = KalmanSensor(name, reading, R=[_sigma(who, sigma) * float(sigma)] * 2, jacobian=jacobian, labels=labels,
                              delay=delay)
        placed = at is not None or rotation is not None
        sensor.native = ("camera", [*p, focal, float(sigma), *((*origin, *W.ravel()) if placed else ())])
        super().__init__(name, [sensor], rate=rate)
        self.position, self.focal, self.sigma, self.at, self.rotation = p, focal, float(sigma), origin, W

"""Generic sensors: readings that fit any model.

:class:`~kalman.framework.sensor.KalmanSensor` is the contract; these fill it in
for readings that come back in every field. Each knows a geometry and a noise,
and nothing of any product. It is told which components of the state it reads
by their indices::

    from kalman.framework import PositionSensor, DopplerSensor, CameraSensor

    # a shell with state (px, vx, ax, py, vy, ay, pz, vz, az)
    P, V = [0, 3, 6], [1, 4, 7]
    seen = (shell.add_sensor(PositionSensor("gps", 9, P, sigma=10.0))
                 .add_sensor(DopplerSensor("doppler", 9, P, V, sigma=1.0))
                 .add_sensor(CameraSensor("camera", 9, P, focal=800.0, sigma=1.5)))

    PositionSensor   z = rotation @ x[axes]                       (a matrix)
    DopplerSensor    the velocity along the line of sight         settings x, y, z
    CameraSensor     two pixel coordinates of a point in space    settings focal, x, y, z

A sensor's **settings** (where it is, a focal length) are parameters of the
model that takes it, ``"<name>.<setting>"``, so ``joint`` can estimate them.

A given piece of hardware is an **adapter**: an instrument made of these, with
what its data sheet and its mounting say (``kalman.adapters``). The framework
ships the generic sensors only.

Each is one C++ class (``Kalman::Sensors::Position``, ``Doppler``, ``Camera``,
``include/Kalman/Sensors/``) with its Jacobian written out. The classes here
hold the C++ sensor and compute nothing: ``sensor.g(x)``, its Jacobian, its
noise and its settings are asked of it, and a model takes it as it is, so
nothing is called back into Python during a pass.
"""
from .base     import NativeSensor
from .position import PositionSensor
from .doppler  import DopplerSensor
from .camera   import CameraSensor

__all__ = ["NativeSensor", "PositionSensor", "DopplerSensor", "CameraSensor"]

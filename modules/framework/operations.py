"""
The model operations as functions.

Each builds a model from other models, and is also a method of
:class:`~kalman.framework.model.KalmanModel` and a class of
``kalman.framework.models``: ``joint(m, p)``, ``m.joint(p)`` and ``Joint(m, p)``
are the same model, and likewise ``add_sensor`` and ``Fusion``.
"""
from __future__ import annotations


def joint(model, parameters, **kwargs):
    """``model`` with some of its ``parameters`` estimated along with its state:
    ``joint(shell, {"c_drag": (sigma, drift)})``. See :class:`~.models.joint.Joint`."""
    return model.joint(parameters, **kwargs)


def add_sensor(model, sensor):
    """``model`` read by one more ``sensor`` (a :class:`~.sensor.KalmanSensor`), as a new
    model. See :class:`~.models.fusion.Fusion`."""
    return model.add_sensor(sensor)


__all__ = ["joint", "add_sensor"]

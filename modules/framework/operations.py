"""
The model operations as functions.

Each builds a model from other models, and is also a method of
:class:`~kalman.framework.model.KalmanModel` and a class of
``kalman.framework.models``: ``joint(m, p)``, ``m.joint(p)`` and ``Joint(m, p)``
are the same model.
"""
from __future__ import annotations


def joint(model, parameters, **kwargs):
    """``model`` with some of its ``parameters`` estimated along with its state:
    ``joint(shell, {"c_drag": (sigma, drift)})``. See :class:`~.models.joint.Joint`."""
    return model.joint(parameters, **kwargs)


__all__ = ["joint"]

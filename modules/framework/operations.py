"""
The model operations as functions.

Each builds a model from other models, and is also a method of
:class:`~kalman.framework.model.KalmanModel` and a class of
``kalman.framework.models``: ``joint(m, p)``, ``m.joint(p)`` and ``Joint(m, p)``
are the same model, and likewise ``add_sensor`` and ``Fusion``.

``compose`` puts things side by side: sensors into an instrument
(:class:`~.instrument.KalmanInstrument`).
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


def add_instrument(model, instrument):
    """``model`` read by every sensor of ``instrument`` (a
    :class:`~.instrument.KalmanInstrument`), as a new model."""
    return model.add_instrument(instrument)


def compose(*parts, name=None, rate=None):
    """Sensors as one instrument: ``compose(thermometer, co2, name="station", rate=0.5)``.
    Without a name it is named after its sensors (``"thermometer+co2"``)."""
    from .instrument import KalmanInstrument
    from .sensor import KalmanSensor
    if len(parts) == 1 and isinstance(parts[0], (list, tuple)):
        parts = tuple(parts[0])
    if not parts or not all(isinstance(part, KalmanSensor) for part in parts):
        raise TypeError("compose: KalmanSensor objects are expected, got "
                        f"{[type(part).__name__ for part in parts]}.")
    return KalmanInstrument(name or "+".join(part.name for part in parts), parts, rate=rate)


__all__ = ["joint", "add_sensor", "add_instrument", "compose"]

"""
The model operations as functions.

Each builds a model from other models, and is also a method of
:class:`~kalman.framework.model.KalmanModel` and a class of
``kalman.framework.models``: ``joint(m, p)``, ``m.joint(p)`` and ``Joint(m, p)``
are the same model, and likewise ``add_sensor`` and ``Fusion``, ``lag`` and ``Lag``.

``compose`` puts things side by side: models into a composite model
(``compose(a, b)``, ``a.compose(b)``, ``Composite(a, b)``), sensors into an
instrument (:class:`~.instrument.KalmanInstrument`).
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


def lag(model, depth):
    """``model`` carrying its last ``depth`` states, for sensors that report late and
    fixed-lag smoothing. See :class:`~.models.lag.Lag`."""
    return model.lag(depth)


def add_instrument(model, instrument):
    """``model`` read by every sensor of ``instrument`` (a
    :class:`~.instrument.KalmanInstrument`), as a new model."""
    return model.add_instrument(instrument)


def compose(*parts, name=None, rate=None, readout=None, names=None, **kwargs):
    """Things side by side.

    Models, each with its own state, as one model (:class:`~.models.compose.Composite`):
    ``compose(a, b)``, or ``compose(a, b, readout=KalmanReadout.SUMMED)`` when they
    share one measurement; ``names=`` calls the parts.

    Sensors as one instrument: ``compose(thermometer, co2, name="station", rate=0.5)``.
    Without a name it is named after its sensors (``"thermometer+co2"``)."""
    from .instrument import KalmanInstrument
    from .model import KalmanModel
    from .sensor import KalmanSensor
    if len(parts) == 1 and isinstance(parts[0], (list, tuple)):
        parts = tuple(parts[0])
    if parts and all(isinstance(part, KalmanModel) for part in parts):
        if name is not None or rate is not None:
            raise TypeError("compose: name= and rate= are an instrument's; models take readout= and names=.")
        from .constants import KalmanReadout
        from .models.compose import Composite
        return Composite(*parts, readout=readout or KalmanReadout.STACKED, names=names, **kwargs)
    if readout is not None or names is not None or kwargs:
        raise TypeError("compose: readout= and names= are for models; sensors take name= and rate=.")
    if not parts or not all(isinstance(part, KalmanSensor) for part in parts):
        raise TypeError("compose: models, or sensors, are expected, got "
                        f"{[type(part).__name__ for part in parts]}.")
    return KalmanInstrument(name or "+".join(part.name for part in parts), parts, rate=rate)


__all__ = ["joint", "add_sensor", "add_instrument", "compose", "lag"]

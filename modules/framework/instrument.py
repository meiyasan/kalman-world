"""
An instrument: a set of sensors, and what they share.

The Python face of ``Kalman::Instrument`` (``include/Kalman/Instrument.h``), with
the same fields. A sensor is one reading of the state; an instrument is the
piece of hardware that carries one or more of them: a weather station, a
receiver, a camera. Its sensors share

- a **name**, under which the model groups them (``model.instruments``);
- a **rate**, for the sensors that state none of their own;
- a **calibration**: every reading goes through one gain and one offset,

      z = gain * g(x, t) + offset

  which the model that takes the instrument declares as two parameters,
  ``"<name>.gain"`` and ``"<name>.offset"``. Being parameters, they can be
  estimated with the state: ``model.joint({"station.offset": (sigma, drift)})``
  calibrates the instrument while it measures.

An instrument is made by composing sensors::

    station = compose(thermometer, co2, name="station", rate=0.5)
    station = thermometer | co2                      # named after both
    room    = model.add_instrument(station)          # a new model, read by both

The arithmetic is the C++ model's (``Kalman::Models::Fusion``): nothing is
computed here.
"""
from __future__ import annotations

from .sensor import KalmanSensor


class KalmanInstrument:
    """A set of sensors.

    Parameters
    ----------
    name : what the instrument is called.
    sensors : its :class:`~.sensor.KalmanSensor` objects, in the order of their channels.
    rate : how often it reports, for its sensors that state no rate of their
        own; ``None`` for every step of the model.
    gain, offset : give either and its readings go through
        ``z = gain * g(x, t) + offset`` (see :meth:`calibrate`).
    """

    def __init__(self, name, sensors, *, rate=None, gain=None, offset=None):
        if not isinstance(name, str) or not name:
            raise ValueError("KalmanInstrument: an instrument needs a name.")
        sensors = list(sensors)
        if not sensors:
            raise ValueError(f"KalmanInstrument: {name!r} has no sensor.")
        for sensor in sensors:
            if not isinstance(sensor, KalmanSensor):
                raise TypeError(f"KalmanInstrument: {name!r} is made of KalmanSensor objects, "
                                f"got {type(sensor).__name__}.")
        names = [sensor.name for sensor in sensors]
        if len(set(names)) != len(names):
            raise ValueError(f"KalmanInstrument: {name!r} has two sensors of one name ({names}).")
        if rate is not None and not float(rate) > 0.0:
            raise ValueError(f"KalmanInstrument: the rate of {name!r} must be positive.")
        self.name = name
        self.sensors = sensors
        self.rate = None if rate is None else float(rate)
        self.calibrated, self.gain, self.offset = False, 1.0, 0.0
        if gain is not None or offset is not None:
            self.calibrate(1.0 if gain is None else gain, 0.0 if offset is None else offset)

    def calibrate(self, gain=1.0, offset=0.0):
        """Give the instrument a calibration to carry, ``z = gain * g + offset``: the
        parameters ``"<name>.gain"`` and ``"<name>.offset"`` of the model that takes
        it. Returns the instrument."""
        self.calibrated, self.gain, self.offset = True, float(gain), float(offset)
        return self

    @property
    def channels(self):
        """The total number of channels, or ``None`` while a sensor cannot tell."""
        counts = [sensor.channels for sensor in self.sensors]
        return None if any(k is None for k in counts) else sum(counts)

    def __or__(self, sensor):
        """One more sensor on the instrument (a new instrument)."""
        if not isinstance(sensor, KalmanSensor):
            return NotImplemented
        out = KalmanInstrument(self.name, [*self.sensors, sensor], rate=self.rate)
        if self.calibrated:
            out.calibrate(self.gain, self.offset)
        return out

    def __iter__(self):
        return iter(self.sensors)

    def __len__(self):
        return len(self.sensors)

    def __repr__(self):
        extra = f", gain={self.gain:g}, offset={self.offset:g}" if self.calibrated else ""
        rate = "" if self.rate is None else f", rate={self.rate:g}"
        return f"KalmanInstrument({self.name!r}, {[sensor.name for sensor in self.sensors]}{rate}{extra})"

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

Two more things an instrument can say:

- where its data comes from, ``provider=``: a provider (or any function) that
  returns the instrument's own readings, one column per channel.
  ``model.measurements()`` then fetches them and lays them on the model's steps;
- ``sequential=True``: its sensors are read one after the other, one per step,
  and not stacked into one measurement. This is a detector: its layers, each at
  its ``station``. A model that is a detector reports itself this way
  (``chamber.instruments``); such an instrument is not added to another model.

A given piece of hardware is an **adapter**: a subclass of this class made of
generic sensors (``kalman.framework.sensors``), with what its data sheet and
its mounting say, and its provider. See ``kalman.adapters``.

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
    provider : where its readings come from: a provider (``fetch()`` returning
        ``{"zk": ...}``) or a function returning the array, ``(N, channels)`` at
        the instrument's own rate. See :meth:`read`.
    sequential : ``True`` when its sensors are read one after the other, one per
        step (a detector's layers), and not stacked into one measurement.
    """

    def __init__(self, name, sensors, *, rate=None, gain=None, offset=None, provider=None, sequential=False):
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
        self.provider = provider
        self.sequential = bool(sequential)
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
        """The number of channels of its measurement, or ``None`` while a sensor
        cannot tell: the sum over its sensors, or one sensor's when they are read
        one after the other."""
        counts = [sensor.channels for sensor in self.sensors]
        if any(k is None for k in counts):
            return None
        return counts[0] if self.sequential else sum(counts)

    def read(self, **fetch):
        """Its own readings, from its provider: ``{sensor name: (N, k) array}``, one
        entry per sensor, as :meth:`KalmanModel.measurements` takes them. ``fetch``
        goes to the provider."""
        import numpy as np
        if self.provider is None:
            raise ValueError(f"KalmanInstrument {self.name!r} has no provider: give provider= to read it.")
        if self.sequential:
            raise ValueError(f"KalmanInstrument {self.name!r} is a detector: its readings are hits per layer, "
                             "handed to track finding, not rows of a measurement.")
        source = self.provider
        data = source.fetch(**fetch) if hasattr(source, "fetch") else source(**fetch)
        zk = np.asarray(data["zk"] if isinstance(data, dict) else data, dtype=np.float64)
        widths = [sensor.channels for sensor in self.sensors]
        if any(k is None for k in widths):
            raise ValueError(f"KalmanInstrument {self.name!r}: a sensor does not say how many channels it has.")
        zk = zk.reshape(len(zk), -1) if zk.size else zk.reshape(0, sum(widths))
        if zk.shape[1] != sum(widths):
            raise ValueError(f"KalmanInstrument {self.name!r}: its provider returned {zk.shape[1]} columns, its "
                             f"sensors have {sum(widths)} channels ({dict(zip([s.name for s in self.sensors], widths))}).")
        out, first = {}, 0
        for sensor, k in zip(self.sensors, widths):
            out[sensor.name] = zk[:, first:first + k]
            first += k
        return out

    def __or__(self, sensor):
        """One more sensor on the instrument (a new instrument)."""
        if not isinstance(sensor, KalmanSensor):
            return NotImplemented
        out = KalmanInstrument(self.name, [*self.sensors, sensor], rate=self.rate, provider=self.provider,
                               sequential=self.sequential)
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
        if self.sequential:
            extra += ", sequential"
        return f"{type(self).__name__}({self.name!r}, {[sensor.name for sensor in self.sensors]}{rate}{extra})"

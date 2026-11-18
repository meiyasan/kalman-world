"""Adapters: one piece of hardware each.

The framework ships generic sensors (``kalman.framework.sensors``): a geometry
and a noise, nothing of any product. An **adapter** is what a given device
adds to them: the numbers of its data sheet (noise, rate, delay), where and how
it is mounted, its calibration, and where its data comes from. It is a
:class:`~kalman.framework.instrument.KalmanInstrument` made of generic sensors::

    from kalman.framework import KalmanInstrument, DopplerSensor, PositionSensor

    class BenchRadar(KalmanInstrument):
        \"\"\"The radar of one test bench.\"\"\"

        def __init__(self, n, position, velocity, provider=None):
            super().__init__("bench-radar", [
                PositionSensor("range-fix", n, position, sigma=0.5),         # m, from the data sheet
                DopplerSensor("doppler", n, position, velocity, sigma=0.15,  # m/s
                              at=[12.0, -3.5, 1.8]),                         # where it is mounted [m]
            ], rate=20.0, provider=provider)                                 # Hz; where its data comes from
            self.calibrate(gain=1.0, offset=0.0)                             # then estimated with joint

    tracked = model.add_instrument(BenchRadar(6, [0, 2, 4], [1, 3, 5], provider=stream))
    zk = tracked.measurements()                                              # fetched from the radar
    tracked.joint({"bench-radar.offset": (1.0, 0.0)}).filter(zk)

In C++ an adapter is a class deriving from ``Kalman::Instrument``, made of
``Kalman::Sensors``.

This package is where adapters live, beside ``kalman.apps``: the applications
are specific models, the adapters specific hardware, and the framework knows
neither. One module per device, named after it:

- ``netatmo``: Netatmo indoor modules (CO2), from the Netatmo web API or files.
"""
from .netatmo import NetatmoCloud, NetatmoCSV, NetatmoModule

__all__: list[str] = ["NetatmoCloud", "NetatmoCSV", "NetatmoModule"]

"""What an instrument says beyond its sensors: where its data comes from, and
whether its sensors are read one after the other (a detector)."""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework import DopplerSensor, KalmanInstrument, KalmanModel, KalmanSensor, PositionSensor, compose
from modules.framework.native import core
from modules.providers.sensor import SensorProvider

DT = 0.05
P, V = [0, 2, 4], [1, 3, 5]


def target():
    F, Q, H = np.eye(6), np.zeros((6, 6)), np.zeros((3, 6))
    for axis in range(3):
        p, v = 2 * axis, 2 * axis + 1
        F[p, v] = DT
        H[axis, p] = 1.0
        Q[p, p], Q[p, v], Q[v, p], Q[v, v] = DT**4 / 4, DT**3 / 2, DT**3 / 2, DT**2
    return KalmanModel("gps", fs=1 / DT, F=F, H=H, Q=Q, R=25.0 * np.eye(3),
                       x0=np.array([[100.0], [30.0], [50.0], [-10.0], [200.0], [5.0]]), P0=25.0 * np.eye(6))


class BenchRadar(KalmanInstrument):
    """An adapter: one piece of hardware, an instrument of generic sensors."""

    def __init__(self, provider=None):
        super().__init__("bench-radar", [
            PositionSensor("fix", 6, P, sigma=0.5),
            DopplerSensor("doppler", 6, P, V, sigma=0.15, at=[12.0, -3.5, 1.8]),
        ], rate=10.0, provider=provider)
        self.calibrate(gain=1.0, offset=0.0)


@unittest.skipUnless(core.HAS_CXX_CORE, "needs the C++ core")
class TestProvider(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        rng = np.random.default_rng(3)
        self.rows = rng.normal(size=(40, 4))                    # 4 s at 10 Hz: 3 position channels, 1 Doppler

    def test_read_splits_the_columns_by_sensor(self):
        radar = BenchRadar(provider=SensorProvider(self.rows))
        readings = radar.read()
        self.assertEqual(list(readings), ["fix", "doppler"])
        np.testing.assert_array_equal(readings["fix"], self.rows[:, :3])
        np.testing.assert_array_equal(readings["doppler"], self.rows[:, 3:])

    def test_a_function_is_a_provider_too(self):
        radar = BenchRadar(provider=lambda: self.rows)
        np.testing.assert_array_equal(radar.read()["doppler"], self.rows[:, 3:])

    def test_the_model_fetches_its_instruments_readings(self):
        model = target().add_instrument(BenchRadar(provider=SensorProvider(self.rows)))
        zk = model.measurements()
        self.assertEqual(zk.shape, (80, 7))                      # 4 s at the model's 20 Hz; 3 + 3 + 1 channels
        self.assertTrue(np.isnan(zk[:, :3]).all(), "the model's own sensor has no provider")
        np.testing.assert_array_equal(zk[0::2, 3:], self.rows)   # the radar reports every other step
        self.assertTrue(np.isnan(zk[1::2, 3:]).all())
        np.testing.assert_array_equal(zk, model.measurements(BenchRadar(provider=lambda: self.rows).read()))
        record = model.filter(zk, reset=True)
        self.assertTrue(np.all(np.isfinite(record.xk)))

    def test_the_provider_is_kept_through_other_operations(self):
        model = target().add_instrument(BenchRadar(provider=lambda: self.rows)).joint({"bench-radar.offset": (1.0, 0.0)})
        self.assertIsNotNone({i.name: i for i in model.instruments}["bench-radar"].provider)
        self.assertEqual(model.measurements().shape, (80, 7))

    def test_refusals(self):
        with self.assertRaises(ValueError):
            BenchRadar().read()                                  # no provider
        with self.assertRaises(ValueError):
            BenchRadar(provider=lambda: self.rows[:, :2]).read()  # columns that do not add up
        with self.assertRaises(ValueError):
            target().measurements()                              # nothing to fetch from


@unittest.skipUnless(core.HAS_CXX_CORE, "needs the C++ core")
class TestADetectorIsAnInstrument(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        from modules.apps import DriftChamberModel
        self.chamber = DriftChamberModel(n_layers=8)

    def test_its_layers_are_one_sequential_instrument(self):
        instruments = self.chamber.instruments
        self.assertEqual(len(instruments), 1)
        detector = instruments[0]
        self.assertTrue(detector.sequential)
        self.assertEqual(detector.name, self.chamber.sensors[0].name)
        self.assertEqual([layer.name for layer in detector], [f"layer {i}" for i in range(8)])
        np.testing.assert_array_equal([layer.station for layer in detector], self.chamber.radii)
        self.assertEqual(detector.channels, 2, "a step reads one layer")
        self.assertEqual(len(self.chamber.sensors), 1, "the measurement is still one sensor's")

    def test_the_cxx_chamber_says_the_same(self):
        from modules.framework.models import native
        lib, handle = core.lib, native.view_of(self.chamber)
        self.assertEqual(lib.kalman_model_instrument_count(handle), 1)
        self.assertEqual(lib.kalman_model_instrument_sequential(handle, 0), 1)
        detector = self.chamber.instruments[0]
        self.assertEqual(lib.kalman_model_instrument_sensor_count(handle, 0), len(detector))
        station = core.ffi.new("double *")
        for k, layer in enumerate(detector):
            self.assertEqual(core._s(lib.kalman_model_instrument_sensor_name(handle, 0, k)), layer.name)
            core.check(lib.kalman_model_instrument_sensor_station(handle, 0, k, station), "station")
            self.assertEqual(station[0], layer.station)
        self.assertEqual(lib.kalman_model_sensor_instrument(handle, 0), 0)

    def test_it_is_not_added_to_another_model(self):
        with self.assertRaises(ValueError):
            target().add_instrument(self.chamber.instruments[0])
        with self.assertRaises(ValueError):
            self.chamber.instruments[0].read()

    def test_an_ordinary_instrument_is_not_sequential(self):
        station = compose(KalmanSensor("a", G=[[1.0, 0, 0, 0, 0, 0]], R=1.0), name="station")
        self.assertFalse(station.sequential)
        self.assertFalse(target().add_instrument(station).instruments[1].sequential)


if __name__ == "__main__":
    unittest.main()

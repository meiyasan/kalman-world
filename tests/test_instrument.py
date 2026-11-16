"""Instruments: a set of sensors with one name, one rate and one calibration.

``compose(s1, s2, name=...)`` (``s1 | s2``) makes a ``KalmanInstrument``, the
Python face of ``Kalman::Instrument``; ``model.add_instrument(instrument)`` gives
the model read by all its sensors. Pinned here: the ways to make one, what its
sensors share, the calibration parameters the model then declares, their
estimation with ``joint`` (calibrating while measuring), and what observability
says about it. The same scenario is in ``tests/cxx/25_instrument.cc``.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework import (KalmanInstrument, KalmanModel, KalmanSensor, add_instrument,
                               compose)
from modules.framework.constants import KalmanStructureKind
from modules.framework.native import core

needs_core = unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")


def room():
    """Temperature and CO2, both slowly wandering; the model's own sensor reads the CO2."""
    return KalmanModel("room", fs=1.0, F=np.eye(2), H=np.array([[0.0, 1.0]]), Q=np.diag([1e-4, 1e-2]),
                       R=np.array([[4.0]]), x0=np.array([[21.0], [450.0]]), P0=np.diag([4.0, 100.0]))


def reads(name, component, variance, **kw):
    G = np.zeros((1, 2))
    G[0, component] = 1.0
    return KalmanSensor(name, G=G, R=variance, **kw)


probe = lambda **kw: reads("probe", 0, 0.04, **kw)
humidity = lambda **kw: reads("humidity", 1, 1.0, **kw)


class TestCompose(unittest.TestCase):

    def test_three_ways(self):
        a, b = probe(), humidity()
        for station in (compose(a, b), a | b, a.compose(b), compose([a, b])):
            self.assertIsInstance(station, KalmanInstrument)
            self.assertEqual(station.name, "probe+humidity")
            self.assertEqual([s.name for s in station], ["probe", "humidity"])
        station = compose(a, b, name="station", rate=0.5)
        self.assertEqual((station.name, station.rate, station.channels, len(station)), ("station", 0.5, 2, 2))
        self.assertEqual(len((a | b) | reads("third", 0, 1.0)), 3)

    def test_calibration(self):
        station = compose(probe(), name="station")
        self.assertFalse(station.calibrated)
        self.assertIs(station.calibrate(offset=1.5), station)
        self.assertEqual((station.calibrated, station.gain, station.offset), (True, 1.0, 1.5))
        self.assertTrue(KalmanInstrument("s", [probe()], gain=2.0).calibrated)
        self.assertTrue((station | humidity()).calibrated)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            compose(probe(), probe(), name="twice")
        with self.assertRaises(ValueError):
            KalmanInstrument("", [probe()])
        with self.assertRaises(ValueError):
            KalmanInstrument("empty", [])
        with self.assertRaises(ValueError):
            compose(probe(), name="s", rate=-1.0)
        with self.assertRaises(TypeError):
            compose(probe(), "humidity")
        with self.assertRaises(TypeError):
            room().add_instrument(probe())
        with self.assertRaises(TypeError):
            room().add_sensor(compose(probe()))


@needs_core
class TestAddInstrument(unittest.TestCase):

    def test_uncalibrated_is_still_matrices(self):
        model = room().add_instrument(compose(probe(), humidity(), name="station"))
        self.assertEqual((model.n, model.m), (2, 3))
        self.assertIs(model.structure, KalmanStructureKind.MATRICES)
        self.assertEqual([s.name for s in model.sensors], ["room", "probe", "humidity"])
        self.assertEqual([s.instrument for s in model.sensors], [None, "station", "station"])
        groups = model.instruments
        self.assertEqual([(g.name, [s.name for s in g]) for g in groups],
                         [("room", ["room"]), ("station", ["probe", "humidity"])])
        self.assertNotIn("station.offset", model.parameters())

    def test_function_and_method_are_the_same(self):
        station = compose(probe(), humidity(), name="station").calibrate(2.0, 1.5)
        a, b = room().add_instrument(station), add_instrument(room(), station)
        x = np.array([[20.0], [400.0]])
        np.testing.assert_array_equal(a.h(x), b.h(x))

    def test_calibration_is_two_parameters_of_the_model(self):
        model = room().add_instrument(compose(probe(), humidity(), name="station").calibrate(2.0, 1.5))
        self.assertEqual(model.parameters()["station.gain"], 2.0)
        self.assertEqual(model.parameters()["station.offset"], 1.5)
        self.assertIs(model.structure, KalmanStructureKind.MAPS)             # gain * G x + offset: no matrix H
        x = np.array([[20.0], [400.0]])
        np.testing.assert_array_equal(model.h(x).ravel(), [400.0, 41.5, 801.5])
        np.testing.assert_array_equal(model.sense(np.hstack([x, x]))[:, 1], [400.0, 41.5, 801.5])
        model.set_parameter("station.offset", -1.0)                          # one value for all its sensors
        np.testing.assert_array_equal(model.h(x).ravel(), [400.0, 39.0, 799.0])
        sensors = model.sensors
        self.assertEqual([s.offset for s in sensors], [0.0, -1.0, -1.0])
        self.assertEqual([s.calibrated for s in sensors], [False, True, True])
        self.assertEqual(float(sensors[1].measure(x)[0, 0]), 39.0)           # read through the C++ sensor
        station = model.instruments[1]
        self.assertEqual((station.calibrated, station.gain, station.offset), (True, 2.0, -1.0))

    def test_rate_is_shared_unless_a_sensor_has_its_own(self):
        model = room().add_instrument(compose(probe(rate=1.0), humidity(), name="station", rate=0.25))
        self.assertEqual([s.rate for s in model.sensors], [None, 1.0, 0.25])
        reports = model._reports(8)
        self.assertEqual(reports[:, 2].sum(), 2)                             # every fourth step
        self.assertTrue(reports[:, :2].all())

    def test_a_function_sensor(self):
        squared = KalmanSensor("squared", lambda x: [float(x[0, 0]) ** 2], R=1.0)
        model = room().add_instrument(compose(squared, name="odd").calibrate(0.5, 3.0))
        x = np.array([[4.0], [400.0]])
        self.assertEqual(float(model.h(x)[1, 0]), 0.5 * 16.0 + 3.0)
        self.assertAlmostEqual(float(model.H(x)[1, 0]), 0.5 * 2 * 4.0, places=4)          # differenced, with the gain


@needs_core
class TestCalibratingWhileMeasuring(unittest.TestCase):

    OFFSET = 1.5

    @staticmethod
    def thermo(offset=0.0):
        return compose(probe(), name="thermo").calibrate(1.0, offset)

    def test_one_thermometer_cannot_tell_its_offset_from_the_temperature(self):
        blind = room().add_instrument(self.thermo()).joint({"thermo.offset": (2.0, 0.0)})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = blind.observability(50)
        self.assertEqual(report["rank"], 2)                                  # one direction is not determined
        temperature, co2, offset = report["weakness"].values()
        self.assertGreater(min(temperature, offset), 0.1)                    # it mixes these two,
        self.assertLess(co2, 1e-6)                                           # not the CO2
        self.assertIn("thermo.offset", report["weakness"])

    def test_a_reference_thermometer_settles_it(self):
        reference = reads("reference", 0, 0.01)
        checked = room().add_instrument(self.thermo()).add_sensor(reference)
        model = checked.joint({"thermo.offset": (2.0, 0.0)})
        self.assertTrue(model.observability(50)["observable"])
        self.assertEqual(list(model.states.__members__)[2], "thermo.offset")

        truth = room().add_instrument(self.thermo(self.OFFSET)).add_sensor(reference)
        rng = np.random.default_rng(20261117)
        x, zs = truth.x0.copy(), []
        Lq, Lr = np.linalg.cholesky(truth.Q()), np.linalg.cholesky(truth.R())
        for _ in range(400):
            x = truth.f(x) + Lq @ rng.standard_normal((2, 1))
            zs.append((truth.h(x) + Lr @ rng.standard_normal((3, 1))).ravel())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = model.filter(np.array(zs), smooth=False, reset=True)
        estimate, sigma = rec.xk[-1, 2], np.sqrt(rec.Pk[-1, 2, 2])
        self.assertLess(abs(estimate - self.OFFSET), 3 * sigma)
        self.assertLess(sigma, 0.05)
        self.assertEqual(checked.parameters()["thermo.offset"], 0.0)         # the inner model is left as found


if __name__ == "__main__":
    unittest.main()

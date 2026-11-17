"""The sensors of ``kalman.apps``: a position reading, a Doppler radar, a camera.

Each is one C++ class (``Kalman::Apps::Fusion``) with a Python counterpart.
The model takes the C++ sensor, so nothing is called back into Python during a
pass; the Python functions are the reference these tests compare against.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.apps import Camera, DopplerRadar, PositionSensor
from modules.framework import KalmanInstrument, KalmanModel, KalmanSensor
from modules.framework.models import native
from modules.framework.native import core

DT = 0.05
P, V = [0, 2, 4], [1, 3, 5]                     # state (px, vx, py, vy, pz, vz)
X = np.array([3.0, 1.0, 4.0, -2.0, 12.0, 0.5])


def target():
    F, Q, H = np.eye(6), np.zeros((6, 6)), np.zeros((3, 6))
    for axis in range(3):
        p, v = 2 * axis, 2 * axis + 1
        F[p, v] = DT
        H[axis, p] = 1.0
        Q[p, p], Q[p, v], Q[v, p], Q[v, v] = DT**4 / 4, DT**3 / 2, DT**3 / 2, DT**2
    return KalmanModel("gps", fs=1 / DT, F=F, H=H, Q=Q, R=25.0 * np.eye(3),
                       x0=np.array([[100.0], [30.0], [50.0], [-10.0], [200.0], [5.0]]), P0=25.0 * np.eye(6))


def differences(sensor, x, step=1e-6):
    """``dg/dx`` by central differences."""
    columns = []
    for j in range(x.size):
        e = np.zeros(x.size)
        e[j] = step
        columns.append((np.asarray(sensor.g(x + e)) - np.asarray(sensor.g(x - e))) / (2 * step))
    return np.array(columns).T


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class TestReadings(unittest.TestCase):
    """What each reads, in Python (the reference)."""

    def test_position(self):
        gps = PositionSensor("gps", 6, [0, 2], sigma=3.0)
        self.assertIsInstance(gps, KalmanSensor)
        self.assertTrue(gps.is_linear)
        np.testing.assert_array_equal(gps.measure(X).ravel(), [3.0, 4.0])
        np.testing.assert_array_equal(gps.resolved(6)[1], 9.0 * np.eye(2))
        turned = PositionSensor("turned", 6, [0, 2], 3.0, rotation=[[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_array_equal(turned.measure(X).ravel(), [4.0, -3.0])

    def test_doppler(self):
        radar = DopplerRadar("doppler", 6, P, V, sigma=0.5)
        self.assertIsInstance(radar, KalmanInstrument)
        self.assertEqual([s.name for s in radar], ["doppler"])
        sensor = radar.sensors[0]
        self.assertAlmostEqual(sensor.g(X)[0], 1.0 / 13.0, places=15)          # (3 - 8 + 6) / 13
        np.testing.assert_allclose(sensor.jacobian(X), differences(sensor, X), atol=1e-8)
        below = DopplerRadar("below", 6, P, V, 0.5, at=[3.0, 4.0, 0.0]).sensors[0]
        self.assertAlmostEqual(below.g(X)[0], 0.5, places=15)
        on = X.copy()
        on[4] = 0.0
        self.assertEqual(below.g(on)[0], 0.0, "at the radar itself the reading is 0")
        self.assertFalse(below.jacobian(on).any())
        flat = DopplerRadar("flat", 6, [0, 2], [1, 3], 0.5).sensors[0]
        self.assertAlmostEqual(flat.g(X)[0], -1.0, places=15)                  # (3 - 8) / 5

    def test_camera(self):
        image = Camera("camera", 6, P, focal=800.0, sigma=1.5).sensors[0]
        np.testing.assert_allclose(image.g(X), [200.0, 800.0 / 3.0], rtol=1e-14)
        np.testing.assert_allclose(image.jacobian(X), differences(image, X), atol=1e-5)
        side = Camera("side", 6, P, 100.0, 1.0, at=[-5.0, 0.0, 0.0],
                      rotation=[[0, 1, 0], [0, 0, 1], [1, 0, 0]]).sensors[0]      # looks along x
        np.testing.assert_allclose(side.g(X), [50.0, 150.0], rtol=1e-14)
        np.testing.assert_allclose(side.jacobian(X), differences(side, X), atol=1e-6)
        behind = X.copy()
        behind[4] = -1.0
        self.assertTrue(np.all(np.isfinite(image.g(behind))), "a point behind the camera still gives numbers")

    def test_refusals(self):
        for build in (lambda: PositionSensor("s", 6, [7], 1.0), lambda: PositionSensor("s", 6, [0], 0.0),
                      lambda: PositionSensor("s", 6, [], 1.0), lambda: PositionSensor("s", 6, [0], 1.0, np.eye(2)),
                      lambda: DopplerRadar("s", 6, P, [1, 3], 1.0), lambda: DopplerRadar("s", 6, P, V, 1.0, at=[0, 0]),
                      lambda: Camera("s", 6, [0, 2], 800.0, 1.0), lambda: Camera("s", 6, P, 0.0, 1.0),
                      lambda: Camera("s", 6, P, 800.0, 1.0, rotation=np.eye(2))):
            with self.assertRaises(ValueError):
                build()


@unittest.skipUnless(core.HAS_CXX_CORE, "needs the C++ core")
class TestTheModelTakesTheCxxSensor(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        native.apps_library_names()
        if not core.lib.kalman_sensor_kind_exists(b"doppler"):
            self.skipTest("libKalmanApps is not built")

    def instruments(self):
        return (DopplerRadar("doppler", 6, P, V, 0.2, at=[10.0, -20.0, 0.0]),
                Camera("camera", 6, P, 800.0, 1.5, at=[0.0, 0.0, -50.0]))

    def build(self):
        radar, camera = self.instruments()
        return target().add_sensor(PositionSensor("second", 6, P, 2.0)).add_instrument(radar).add_instrument(camera)

    def test_the_three_kinds_are_registered(self):
        for kind in (b"position", b"doppler", b"camera"):
            self.assertEqual(core.lib.kalman_sensor_kind_exists(kind), 1)
        self.assertEqual(core.lib.kalman_sensor_kind_exists(b"sonar"), 0)

    def test_the_cxx_readings_are_the_python_ones(self):
        model = self.build()
        radar, camera = self.instruments()
        z = model.h(X.reshape(-1, 1)).ravel()
        np.testing.assert_allclose(z[3:6], X[P], rtol=0, atol=0)
        np.testing.assert_allclose(z[6], radar.sensors[0].g(X)[0], rtol=1e-14)
        np.testing.assert_allclose(z[7:], camera.sensors[0].g(X), rtol=1e-14)
        H = np.asarray(model.H(X.reshape(-1, 1)))
        np.testing.assert_allclose(H[6:7], radar.sensors[0].jacobian(X), atol=1e-13)
        np.testing.assert_allclose(H[7:], camera.sensors[0].jacobian(X), atol=1e-11)
        np.testing.assert_array_equal(np.diag(model.R())[3:], [4.0, 4.0, 4.0, 0.2 * 0.2, 2.25, 2.25])

    def test_nothing_is_called_back_during_a_pass(self):
        radar, camera = self.instruments()
        calls = [0]
        for sensor in (radar.sensors[0], camera.sensors[0]):
            g = sensor.g

            def counted(x, g=g):
                calls[0] += 1
                return g(x)
            sensor.g = counted
        model = target().add_instrument(radar).add_instrument(camera)
        data = model.sample(5.0, seed=3)
        calls[0] = 0
        model.filter(data.zk, reset=True)
        self.assertEqual(calls[0], 0)

    def test_the_pass_is_the_python_reference_one(self):
        model = self.build()
        data = model.sample(10.0, seed=3)
        record = model.filter(data.zk, reset=True)
        with native.python_reference():
            reference = self.build().filter(data.zk, reset=True)
        np.testing.assert_allclose(record.xk, reference.xk, atol=1e-9)
        np.testing.assert_allclose(record.Pk, reference.Pk, atol=1e-9)

    def test_a_matrix_sensor_keeps_a_model_of_matrices(self):
        from modules.framework.constants import KalmanEngineKind
        model = target().add_sensor(PositionSensor("second", 6, P, 2.0))
        self.assertEqual(model.DEFAULT_ENGINE, KalmanEngineKind.LINEAR)

    def test_the_radar_gives_the_velocity(self):
        radar, _ = self.instruments()
        model = target().add_instrument(radar)
        data = model.sample(30.0, seed=11)
        with_radar = model.filter(data.zk, reset=True)
        without = target().filter(data.zk[:, :3], reset=True)
        self.assertLess(rmse(with_radar.xk[:, 1], data.xk[:, 1]), 0.8 * rmse(without.xk[:, 1], data.xk[:, 1]))

    def test_a_radar_offset_is_estimated_with_the_state(self):
        """A calibrated instrument's offset is a parameter, and ``joint`` estimates it."""
        biased = target().add_instrument(DopplerRadar("doppler", 6, P, V, 0.2).calibrate(1.0, 1.5))
        data = biased.sample(60.0, seed=5)
        assumed = target().add_instrument(DopplerRadar("doppler", 6, P, V, 0.2).calibrate(1.0, 0.0))
        model = assumed.joint({"doppler.offset": (3.0, 0.0)})
        record = model.filter(data.zk, reset=True)
        offset, sigma = record.xk[-1, 6], np.sqrt(record.Pk[-1, 6, 6])
        self.assertLess(abs(offset - 1.5), 4 * sigma)
        self.assertLess(sigma, 0.2)


if __name__ == "__main__":
    unittest.main()

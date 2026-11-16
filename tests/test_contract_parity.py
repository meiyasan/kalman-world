"""One contract, two languages.

What a model says of itself must not depend on the language it is asked in: the
sensors, their channels, the observables, the dimensions and the parameters that
Python reports are those the C++ model reports through the C API. The numbers are
compared elsewhere (``test_native_twins``, ``test_joint``, ``test_add_sensor``);
this file compares the *description*, operation by operation.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core
from modules.framework.sensor import KalmanSensor
import modules.framework.models.native as native
from modules.apps.ballistics import BallisticModel
from tests.test_joint import Oscillator

needs_core = unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
rate = lambda x: [float(np.asarray(x).reshape(-1)[1])]


def cxx_sensors(handle):
    lib = core.lib
    return [(core._s(lib.kalman_model_sensor_name(handle, i)), lib.kalman_model_sensor_channels(handle, i))
            for i in range(lib.kalman_model_sensor_count(handle))]


def cxx_observables(handle, m):
    lib = core.lib
    return [(core._s(lib.kalman_model_measurement_name(handle, i)), core._s(lib.kalman_model_measurement_unit(handle, i)))
            for i in range(m)]


def python_sensors(model):
    return [(s.name, s.channels) for s in model.sensors]


def python_observables(model):
    return [label for s in model.sensors for label in s.observables]


@needs_core
class TestSensors(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")

    def same(self, model):
        self.assertEqual(python_sensors(model), cxx_sensors(model._handle))
        self.assertEqual(python_observables(model), cxx_observables(model._handle, model.m))
        dims = [core.ffi.new("int *") for _ in range(3)]
        core.check(core.lib.kalman_model_dims(model._handle, *dims), "dims")
        self.assertEqual((model.n, model.m, model.l), tuple(int(d[0]) for d in dims))

    def test_a_python_model_with_a_sensor(self):
        model = Oscillator().add_sensor(KalmanSensor("gyro", rate, R=0.04, labels={"RATE": "rad/s"}))
        self.same(model)
        # And it is what was asked for, not only the same on both sides.
        self.assertEqual(python_sensors(model), [("osc", 1), ("gyro", 1)])
        self.assertEqual(python_observables(model), [("Z0", ""), ("RATE", "rad/s")])
        self.assertEqual(list(model.observables.__members__), ["Z0", "RATE"])

    def test_a_plain_python_model_describes_itself_through_cxx(self):
        model = Oscillator()
        from modules.framework.models import native as bridge
        handle = bridge.view_of(model)
        self.assertEqual(python_sensors(model), cxx_sensors(handle))
        self.assertEqual(python_sensors(model), [("osc", 1)])

    def test_the_readings_and_the_noise_are_the_cxx_sensors(self):
        model = Oscillator().add_sensor(KalmanSensor("gyro", rate, R=0.04))
        x = np.ascontiguousarray([0.4, -0.7])
        for i, sensor in enumerate(model.sensors):
            z, R = np.empty(1), np.empty((1, 1))
            core.check(core.lib.kalman_model_sensor_measure(model._handle, i, core.cdptr(x), 0.0, core.dptr(z)), "measure")
            core.check(core.lib.kalman_model_sensor_R(model._handle, i, core.dptr(R)), "R")
            np.testing.assert_array_equal(sensor.measure(x).ravel(), z)
            np.testing.assert_array_equal(sensor.R, R)

    def test_two_sensors_one_flat_list(self):
        model = Oscillator().add_sensor(KalmanSensor("gyro", rate, R=0.04))
        self.same(model.add_sensor(KalmanSensor("pair", lambda x: [x[0, 0], x[1, 0]], R=[1.0, 2.0])))

    def test_a_matrix_sensor(self):
        self.same(Oscillator().add_sensor(KalmanSensor("radar", G=[[0.0, 1.0]], R=0.04, labels={"V": "m/s"})))

    def test_the_ballistic_model_on_its_twin_and_through_callbacks(self):
        make = lambda: BallisticModel("XYZ", order=3, fs=32, m=4.2, c_drag=6e-4, measurement_uncertainty=10,
                                      process_uncertainty=0.5).add_sensor(
            KalmanSensor("doppler", lambda x: [float(x[1, 0])], R=1.0, labels={"V_RADIAL": "m/s"}))
        twin = make()
        self.same(twin)
        self.assertEqual(python_sensors(twin), [("position", 3), ("doppler", 1)])
        with native.python_reference():
            self.same(make())

    def test_a_refusal_is_the_cxx_one(self):
        with self.assertRaisesRegex(ValueError, "Kalman::Sensor: the sensor 'radar' has a G of 1 x 3"):
            Oscillator().add_sensor(KalmanSensor("radar", G=[[0.0, 1.0, 0.0]], R=0.04))


@needs_core
class TestJoint(unittest.TestCase):

    def test_the_initial_state_is_derived_again_in_cxx(self):
        model = Oscillator(damping=0.1).joint({"damping": (0.5, 1e-4)})
        model.initial_state([0.7, 0.2])                      # one the inner model takes
        x0, P0 = np.empty(3), np.empty((3, 3))
        core.check(core.lib.kalman_model_x0(model._handle, core.dptr(x0)), "x0")
        core.check(core.lib.kalman_model_P0(model._handle, core.dptr(P0)), "P0")
        np.testing.assert_array_equal(model.x0.ravel(), x0)
        np.testing.assert_array_equal(model.P0, P0)
        np.testing.assert_array_equal(x0, [0.7, 0.2, 0.1])
        self.assertAlmostEqual(P0[2, 2], 0.25)

    def test_dimensions_parameters_and_sensors(self):
        model = Oscillator().add_sensor(KalmanSensor("gyro", rate, R=0.04)).joint({"damping": (0.5, 1e-4)})
        dims = [core.ffi.new("int *") for _ in range(3)]
        core.check(core.lib.kalman_model_dims(model._handle, *dims), "dims")
        self.assertEqual((model.n, model.m, model.l), tuple(int(d[0]) for d in dims))
        self.assertEqual(core._s(core.lib.kalman_model_state_name(model._handle, 2)), "damping")
        self.assertEqual(list(model.states.__members__)[2], "damping")
        self.assertIn("damping", native.declared_parameters(model._inner_cxx))

    def test_a_refusal_is_the_cxx_one(self):
        with self.assertRaisesRegex(ValueError, "Models::Joint: the parameter 'damping' needs an initial standard"):
            Oscillator().joint({"damping": (0.0, 0.0)})


if __name__ == "__main__":
    unittest.main()

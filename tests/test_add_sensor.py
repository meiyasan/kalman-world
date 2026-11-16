"""Sensors: KalmanSensor, and a model read by one more of them.

``model.add_sensor(sensor)`` (``add_sensor(model, sensor)``, ``Fusion``) is
``Kalman::Models::Fusion`` seen from Python. Pinned here: the three ways in, the
function and the matrix sensors, the list of sensors and the observables derived
from it, both routes of the inner model to C++ (its counterpart, or callbacks), nesting
under ``joint``, and the model's first argument: a name, a sensor, or a list.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework.model import KalmanModel
from modules.framework.constants import KalmanEngineKind, KalmanStructureKind
from modules.framework.models.fusion import Fusion
from modules.framework.operations import add_sensor
from modules.framework.sensor import KalmanSensor
import modules.framework.models.native as native
from modules.apps.ballistics import BallisticModel
from tests.test_joint import Oscillator

needs_apps = unittest.skipUnless(bool(native.apps_library_names()), "libKalmanApps is not built")

rate = lambda x: [float(np.asarray(x).reshape(-1)[1])]          # reads the velocity
gyro = lambda **kw: KalmanSensor("gyro", rate, R=0.04, **kw)


def constant_velocity(H=((1.0, 0.0),), R=((0.25,),)):
    dt = 0.1
    return KalmanModel("cv", fs=1 / dt, F=np.array([[1, dt], [0, 1]]), H=np.array(H, dtype=float),
                       Q=np.diag([1e-4, 1e-2]), R=np.array(R, dtype=float),
                       x0=np.array([[0.0], [1.0]]), P0=np.eye(2))


class TestSensor(unittest.TestCase):

    def test_fields_and_channels(self):
        self.assertEqual(gyro().channels, None)                              # only g can tell
        self.assertEqual(gyro().resolved(2)[0], 1)
        self.assertEqual(KalmanSensor("s", rate, R=[1.0, 2.0]).channels, 2)
        self.assertEqual(KalmanSensor("s", G=[[0.0, 1.0]], R=1.0).channels, 1)
        self.assertTrue(KalmanSensor("s", G=[[0.0, 1.0]], R=1.0).is_linear)

    def test_observables(self):
        self.assertIsNone(gyro().observables)                                # none given: named when added
        self.assertEqual(gyro(labels={"RATE": "rad/s"}).observables, [("RATE", "rad/s")])
        self.assertEqual(gyro(labels=["RATE"]).observables, [("RATE", "")])
        # The C++ sensor names them: after itself, numbered when there are several.
        named = Oscillator().add_sensor(gyro()).sensors[1]
        self.assertEqual(named.observables, [("gyro", "")])
        two = KalmanSensor("imu", lambda x: [x[0, 0], x[1, 0]], R=1.0)
        self.assertEqual(Oscillator().add_sensor(two).sensors[1].observables, [("imu0", ""), ("imu1", "")])

    def test_refusals(self):
        with self.assertRaises(ValueError):
            KalmanSensor("s", R=1.0)                                         # neither g nor G
        with self.assertRaises(ValueError):
            KalmanSensor("s", rate, G=[[0.0, 1.0]], R=1.0)                   # both
        with self.assertRaises(ValueError):
            KalmanSensor("s", rate)                                          # no R
        with self.assertRaises(ValueError):
            KalmanSensor("", rate, R=1.0)                                    # no name


class TestConstruction(unittest.TestCase):

    def test_method_verb_and_class_are_one_model(self):
        for model in (Oscillator().add_sensor(gyro()), add_sensor(Oscillator(), gyro()), Fusion(Oscillator(), gyro())):
            self.assertIsInstance(model, Fusion)
            self.assertEqual((model.n, model.m, model.n_extra), (2, 2, 1))

    def test_a_new_model_and_the_inner_one_unchanged(self):
        inner = Oscillator()
        model = inner.add_sensor(gyro())
        self.assertIsNot(model, inner)
        self.assertEqual((inner.m, len(inner.sensors)), (1, 1))

    def test_only_a_sensor_can_be_added(self):
        with self.assertRaisesRegex(TypeError, "add_sensor takes a KalmanSensor"):
            Oscillator().add_sensor("second")                               # a name is the model's first argument
        for gone in ("sensor", "remove_sensor", "clear_sensors"):
            self.assertFalse(hasattr(Oscillator(), gone), gone)

    def test_function_sensor(self):
        inner = Oscillator()
        model = inner.add_sensor(gyro(labels={"RATE": "rad/s"}))
        x = np.array([[0.4], [-0.7]])
        np.testing.assert_array_equal(model.h(x).ravel(), [0.4, -0.7])
        np.testing.assert_array_equal(model.f(x), inner.f(x))
        np.testing.assert_allclose(model.H(x), [[1.0, 0.0], [0.0, 1.0]], atol=1e-8)
        np.testing.assert_allclose(model.R(), np.diag([1e-2, 0.04]))
        np.testing.assert_allclose(model.Q(), inner.Q(), rtol=1e-9)    # through the differenced noise coupling
        np.testing.assert_array_equal(model.x0, inner.x0)

    def test_sensors_and_observables(self):
        model = Oscillator().add_sensor(gyro(labels={"RATE": "rad/s"}))
        self.assertEqual([s.name for s in model.sensors], ["osc", "gyro"])
        self.assertEqual(list(model.observables.__members__), ["Z0", "RATE"])       # derived from the sensors
        more = model.add_sensor(KalmanSensor("gyro2", rate, R=0.09, labels=["RATE"]))
        self.assertEqual([s.name for s in more.sensors], ["osc", "gyro", "gyro2"])  # one flat list
        # A name the model already has takes its sensor's name in front.
        self.assertEqual(list(more.observables.__members__), ["Z0", "RATE", "gyro2_RATE"])
        self.assertEqual(more.sensors[2].observables, [("gyro2_RATE", "")])

    def test_the_sensors_are_read_from_the_cxx_model(self):
        inner = Oscillator()
        model = inner.add_sensor(gyro(labels={"RATE": "rad/s"}))
        built_in, added = model.sensors
        x = np.array([[0.4], [-0.7]])
        np.testing.assert_array_equal(built_in.measure(x).ravel(), [0.4])   # the inner model's h
        np.testing.assert_array_equal(added.measure(x).ravel(), [-0.7])     # the sensor's g
        np.testing.assert_array_equal(built_in.R, [[1e-2]])
        np.testing.assert_array_equal(added.R, [[0.04]])
        self.assertEqual(inner.sensors[0].name, "osc")                       # a plain model too
        radar = constant_velocity().add_sensor(KalmanSensor("radar", G=[[0.0, 1.0]], R=0.04))
        self.assertTrue(radar.sensors[1].is_linear)
        np.testing.assert_array_equal(radar.sensors[1].G, [[0.0, 1.0]])
        with self.assertRaises(AttributeError):
            model.sensors = []

    def test_a_function_of_time_and_a_given_jacobian(self):
        sensor = KalmanSensor("clock", lambda x, t=None: [x[1, 0] + t], R=0.04,
                              jacobian=lambda x: [[0.0, 2.0]])               # deliberately not dg/dx
        model = Oscillator().add_sensor(sensor)
        self.assertEqual(model.h(np.array([[0.0], [1.0]]), t=3.0)[1], 4.0)
        self.assertEqual(model.H(np.array([[0.0], [1.0]]))[1, 1], 2.0)

    def test_R_as_a_variance_a_list_or_a_matrix(self):
        two = lambda x: [x[0, 0], x[1, 0]]
        for R in (0.5, [0.5, 0.5], 0.5 * np.eye(2)):
            model = Oscillator().add_sensor(KalmanSensor("pair", two, R=R))
            np.testing.assert_allclose(model.R()[1:, 1:], 0.5 * np.eye(2))

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, "G of 1 x 3"):
            Oscillator().add_sensor(KalmanSensor("s", G=[[0.0, 1.0, 0.0]], R=1.0))   # checked in C++
        with self.assertRaises(ValueError):
            Oscillator().add_sensor(KalmanSensor("s", rate, R=np.eye(2)))            # R for two, g gives one
        with self.assertRaises(ValueError):
            Oscillator().add_sensor(KalmanSensor("s", rate, R=1.0, labels=["A", "B"]))
        with self.assertRaises(TypeError):
            Fusion(object(), gyro())
        with self.assertRaises(TypeError):
            Fusion(Oscillator(), "gyro")

    def test_initial_state_is_the_inner_models(self):
        model = Oscillator().add_sensor(gyro())
        model.initial_state([0.3, -0.2])
        np.testing.assert_array_equal(model.x0.ravel(), [0.3, -0.2])
        np.testing.assert_array_equal(model.inner.x0.ravel(), [0.3, -0.2])


class TestFirstArgument(unittest.TestCase):
    """The first argument of a model is its sensor: a name, a KalmanSensor, or a list."""

    KW = dict(fs=10.0, F=np.array([[1, 0.1], [0, 1]]), Q=np.diag([1e-4, 1e-2]),
              x0=np.array([[0.0], [1.0]]), P0=np.eye(2))

    def test_a_name_names_the_models_own_sensor(self):
        model = KalmanModel("radar", H=np.array([[1.0, 0.0]]), R=np.array([[0.25]]), **self.KW)
        self.assertEqual([(s.name, s.channels) for s in model.sensors], [("radar", 1)])

    def test_a_sensor_is_the_measurement(self):
        radar = KalmanSensor("radar", G=[[1.0, 0.0]], R=0.25, labels={"X": "m"})
        model = KalmanModel(radar, **self.KW)
        self.assertEqual((model.m, [s.name for s in model.sensors]), (1, ["radar"]))
        np.testing.assert_array_equal(model.H(), [[1.0, 0.0]])
        np.testing.assert_array_equal(model.R(), [[0.25]])
        self.assertIs(model.structure, KalmanStructureKind.MATRICES)          # still a model of matrices
        written = KalmanModel("radar", H=np.array([[1.0, 0.0]]), R=np.array([[0.25]]), **self.KW)
        np.random.seed(2)
        z = written.sample(20.0).zk
        np.testing.assert_allclose(model.filter(z, smooth=False, reset=True).xk,
                                   written.filter(z, smooth=False, reset=True).xk, rtol=0, atol=1e-12)

    def test_several_sensors(self):
        position = KalmanSensor("gps", G=[[1.0, 0.0]], R=0.25)
        speed = KalmanSensor("wheel", lambda x: [x[1, 0]], R=[0.04])
        model = KalmanModel([position, speed], **self.KW)
        self.assertEqual((model.m, [s.name for s in model.sensors]), (2, ["gps", "wheel"]))
        np.testing.assert_array_equal(model.h(np.array([[3.0], [4.0]])).ravel(), [3.0, 4.0])
        np.testing.assert_allclose(model.R(), np.diag([0.25, 0.04]))

    def test_a_name_in_the_list_keeps_the_models_own_sensor(self):
        model = Oscillator(["probe", gyro()])
        self.assertEqual((model.m, [s.name for s in model.sensors]), (2, ["probe", "gyro"]))
        np.testing.assert_array_equal(model.h(np.array([[0.4], [-0.7]])).ravel(), [0.4, -0.7])
        np.testing.assert_allclose(model.R(), np.diag([1e-2, 0.04]))
        added = Oscillator("probe").add_sensor(gyro())                      # the same model, by add_sensor
        truth = Oscillator(damping=0.3, random_seed=5).add_sensor(gyro()).sample(20.0)
        np.testing.assert_allclose(model.filter(truth.zk, smooth=False, reset=True).xk,
                                   added.filter(truth.zk, smooth=False, reset=True).xk, rtol=0, atol=1e-6)

    def test_refusals(self):
        with self.assertRaises(TypeError):
            KalmanModel(3.5, **self.KW)
        with self.assertRaises(ValueError):
            KalmanModel(["a", "b"], H=np.array([[1.0, 0.0]]), R=np.array([[1.0]]), **self.KW)   # two names
        with self.assertRaises(ValueError):
            KalmanModel([gyro(), gyro()], **self.KW)                           # the same name twice
        with self.assertRaises(ValueError):
            KalmanModel(KalmanSensor("s", rate, R=0.04), **self.KW)            # channels unknown before the model


class TestMatrices(unittest.TestCase):
    """A matrix added to a model of matrices is the model with the stacked H."""

    def test_stays_a_model_of_matrices(self):
        model = constant_velocity().add_sensor(KalmanSensor("radar", G=[[0.0, 1.0]], R=0.04))
        self.assertIs(model.structure, KalmanStructureKind.MATRICES)
        self.assertIs(model.DEFAULT_ENGINE, KalmanEngineKind.LINEAR)
        np.testing.assert_array_equal(model.H(), [[1.0, 0.0], [0.0, 1.0]])

    def test_same_estimate_as_the_model_written_out(self):
        written = constant_velocity(H=((1.0, 0.0), (0.0, 1.0)), R=((0.25, 0.0), (0.0, 0.04)))
        np.random.seed(3)
        sample = written.sample(40.0)
        want = written.filter(sample.zk, smooth=False, reset=True)
        got = constant_velocity().add_sensor(KalmanSensor("radar", G=[[0.0, 1.0]], R=0.04)).filter(sample.zk, smooth=False, reset=True)
        np.testing.assert_allclose(got.xk, want.xk, rtol=0, atol=1e-10)
        np.testing.assert_allclose(got.Pk, want.Pk, rtol=0, atol=1e-10)


class TestEstimation(unittest.TestCase):

    def test_the_second_sensor_lowers_the_velocity_error(self):
        model = Oscillator(damping=0.3, random_seed=5).add_sensor(gyro())
        sample = model.sample(60.0)
        both = model.filter(sample.zk, smooth=False, reset=True)
        alone = Oscillator(damping=0.3).filter(sample.zk[:, :1], smooth=False, reset=True)
        rms = lambda rec: np.sqrt(np.mean((rec.xk[500:, 1] - sample.xk[500:, 1]) ** 2))
        self.assertLess(rms(both), rms(alone))

    def test_a_parameter_of_the_inner_model_is_estimated_through_it(self):
        truth = Oscillator(damping=0.3, random_seed=11).add_sensor(gyro())
        sample = truth.sample(60.0)
        model = Oscillator(damping=0.0).add_sensor(gyro()).joint({"damping": (0.5, 1e-4)})
        self.assertEqual((model.n, model.m), (3, 2))
        rec = model.filter(sample.zk, smooth=False, reset=True)
        estimate, sigma = rec.xk[-1, 2], np.sqrt(rec.Pk[-1, 2, 2])
        self.assertLess(abs(estimate - 0.3), 3 * sigma)
        self.assertLess(sigma, 0.03)
        self.assertEqual(model.inner.inner["damping"], 0.0)


class TestCameras(unittest.TestCase):
    """A camera reads two pixel coordinates, by a projection that divides by the
    depth. One camera gives the direction of a point and not how far it is; two
    at different places give both."""

    FOCAL = 800.0
    TRUTH = np.array([[1.5], [-0.8], [40.0]])
    LEFT, RIGHT = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    def camera(self, name, position):
        def pixels(x):
            p = np.asarray(x, dtype=float).reshape(-1) - position        # looking along +z
            return [self.FOCAL * p[0] / p[2], self.FOCAL * p[1] / p[2]]
        return KalmanSensor(name, pixels, R=0.5 ** 2, labels={"U": "px", "V": "px"})

    def point(self):
        # A point at rest. The first guess is what one look through the left camera
        # gives: on its line of sight, at an assumed depth of 30 m (the truth is 40 m),
        # with 20 m of doubt on the depth.
        guess = self.LEFT + 0.75 * (self.TRUTH.ravel() - self.LEFT)
        return KalmanModel("point", fs=10.0, F=np.eye(3), H=np.array([[1.0, 0.0, 0.0]]),
                           Q=1e-10 * np.eye(3), R=np.array([[1e4]]),
                           x0=guess.reshape(3, 1), P0=np.diag([25.0, 25.0, 400.0]))

    def track(self, model, seed, steps=200):
        rng = np.random.default_rng(seed)
        clean = np.asarray(model.h(self.TRUTH), dtype=float).reshape(-1)
        z = clean + np.sqrt(np.diag(model.R())) * rng.standard_normal((steps, model.m))
        return model.filter(z, smooth=False, reset=True)

    def test_observables_of_two_cameras(self):
        both = self.point().add_sensor(self.camera("left", self.LEFT)).add_sensor(self.camera("right", self.RIGHT))
        self.assertEqual(list(both.observables.__members__), ["Z0", "U", "V", "right_U", "right_V"])

    def test_one_camera_leaves_the_depth_open_and_two_give_it(self):
        left, right = self.camera("left", self.LEFT), self.camera("right", self.RIGHT)
        for seed in (4, 5, 6):
            one = self.track(self.point().add_sensor(left), seed)
            two = self.track(self.point().add_sensor(left).add_sensor(right), seed)
            depth_one, depth_two = np.sqrt(one.Pk[-1, 2, 2]), np.sqrt(two.Pk[-1, 2, 2])
            # The error bars say it, and they are honest ones.
            self.assertGreater(depth_one, 5.0)                               # one camera: metres of doubt left
            self.assertLess(abs(one.xk[-1, 2] - 40.0), 3 * depth_one)
            self.assertLess(depth_two, 0.1)                                  # two: triangulated to centimetres
            self.assertLess(abs(two.xk[-1, 2] - 40.0), 3 * depth_two)
            self.assertLess(abs(two.xk[-1, 0] - 1.5), 0.05)


@needs_apps
class TestBallisticDoppler(unittest.TestCase):
    """The case of notebook 01, Step 6: a Doppler radar on a tracked shell."""

    INIT = [[100, 100, 0], [200, 100, 0], [300, 100, -9.81]]
    COMMON = dict(order=3, fs=32, m=4.2, c_drag=6e-4, measurement_uncertainty=10, process_uncertainty=0.5)

    @staticmethod
    def doppler(x):
        x = np.asarray(x).reshape(-1)
        p, vel = x[[0, 3, 6]], x[[1, 4, 7]]
        r = float(np.linalg.norm(p))
        return [float(vel @ (p / r if r > 1.0 else np.array([0.0, 0.0, 1.0])))]

    def make(self, **kw):
        model = BallisticModel("XYZ", **self.COMMON, **kw).add_sensor(
            KalmanSensor("doppler", self.doppler, R=1.0, labels={"V_RADIAL": "m/s"}))
        model.initial_state(self.INIT)
        return model

    def setUp(self):
        warnings.simplefilter("ignore")
        self.sample = self.make(random_seed=7).sample(12)

    def test_the_counterpart_and_the_callbacks_agree(self):
        with native.python_reference():                 # no counterpart: the shell is called back
            py = self.make().filter(self.sample.zk, smooth=False, reset=True)
        cx = self.make().filter(self.sample.zk, smooth=False, reset=True)
        np.testing.assert_array_less(np.abs(py.xk - cx.xk).max(axis=0), 1e-6 * np.maximum(1.0, np.abs(py.xk).max(axis=0)))

    def test_one_position_sensor_of_three_channels_then_doppler(self):
        model = self.make()
        self.assertEqual([(s.name, s.channels) for s in model.sensors], [("XYZ", 3), ("doppler", 1)])
        self.assertEqual(model.sensors[0].observables, [("X", "m"), ("Y", "m"), ("Z", "m")])
        self.assertEqual(model.sensors[1].observables, [("V_RADIAL", "m/s")])
        self.assertEqual(list(model.observables.__members__), ["X", "Y", "Z", "V_RADIAL"])

    def test_doppler_lowers_the_velocity_error(self):
        both = self.make().filter(self.sample.zk, smooth=False, reset=True)
        position = BallisticModel("XYZ", **self.COMMON)
        position.initial_state(self.INIT)
        alone = position.filter(self.sample.zk[:, :3], smooth=False, reset=True)
        for dim in (1, 4, 7):
            rms = lambda rec: np.sqrt(np.mean((rec.xk[:, dim] - self.sample.xk[:, dim]) ** 2))
            self.assertLess(rms(both), rms(alone))


if __name__ == "__main__":
    unittest.main()

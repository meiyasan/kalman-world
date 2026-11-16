"""joint: parameters of a model estimated with its state.

``Joint`` (``model.joint(...)``, ``joint(model, ...)``) is ``Kalman::Models::Joint`` seen from Python. Its
inner model reaches C++ as its counterpart (libKalmanApps) or as a callback model whose
parameters are set back on the Python side. Both routes are pinned here, against
each other and against the analytic Jacobian of a small oscillator.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework.model import KalmanModel
from modules.framework.models.joint import Joint
from modules.framework.operations import joint
from modules.framework.native import core
import modules.framework.models.native as native
from modules.apps.ballistics import BallisticModel

HAVE_APPS = bool(native.apps_library_names())
needs_apps = unittest.skipUnless(HAVE_APPS, "libKalmanApps is not built")
needs_core = unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")


class Oscillator(KalmanModel):
    """A damped oscillator stepped by explicit Euler; the damping is a parameter
    read at every step. Only the position is measured."""

    NAME = "Test Oscillator"

    def __init__(self, sensor="osc", *, omega=2.0, damping=0.0, fs=50.0, **kwargs):
        self.n, self.m, self.l = 2, 1, 0
        self["omega"], self["damping"] = omega, damping
        super().__init__(sensor, fs=fs, Q=np.diag([1e-6, 1e-4]), R=np.array([[1e-2]]),
                         x0=np.array([[1.0], [0.0]]), P0=np.eye(2), **kwargs)

    def f(self, x, u=None, w=None, t=None):
        x = np.asarray(x, dtype=float).reshape(2, 1)
        dt = 1.0 / self.fs
        out = np.array([[x[0, 0] + dt * x[1, 0]],
                        [x[1, 0] + dt * (-self["omega"] ** 2 * x[0, 0] - self["damping"] * x[1, 0])]])
        return out if w is None else out + np.asarray(w, dtype=float).reshape(2, 1)

    def h(self, x, v=None, t=None):
        z = np.asarray(x, dtype=float).reshape(-1, 1)[:1]
        return z if v is None else z + np.asarray(v, dtype=float).reshape(1, 1)


class Frozen(Oscillator):
    """Reads its damping once, when built: a parameter the filter cannot move."""

    def __init__(self, *args, damping=0.0, **kwargs):
        self._damping = damping
        super().__init__(*args, damping=damping, **kwargs)

    def f(self, x, u=None, w=None, t=None):
        x = np.asarray(x, dtype=float).reshape(2, 1)
        dt = 1.0 / self.fs
        return np.array([[x[0, 0] + dt * x[1, 0]],
                         [x[1, 0] + dt * (-self["omega"] ** 2 * x[0, 0] - self._damping * x[1, 0])]])


class TestConstruction(unittest.TestCase):

    def test_dimensions_and_initial_state(self):
        model = Joint(Oscillator(damping=0.1), {"damping": (0.5, 1e-4)})
        self.assertEqual((model.n, model.m, model.l, model.n_inner), (3, 1, 0, 2))
        np.testing.assert_array_equal(model.x0.ravel(), [1.0, 0.0, 0.1])     # starts from model["damping"]
        self.assertAlmostEqual(model.P0[2, 2], 0.25)
        self.assertEqual(model.P0[0, 2], 0.0)
        self.assertEqual(model.fs, 50.0)
        self.assertEqual(list(model.states.__members__), ["X0", "X1", "damping"])

    def test_method_verb_and_class_are_one_model(self):
        spec = {"damping": (0.5, 1e-4)}
        for model in (Oscillator().joint(spec), joint(Oscillator(), spec), Joint(Oscillator(), spec)):
            self.assertIsInstance(model, Joint)
            self.assertEqual((model.n, model.names), (3, ["damping"]))

    def test_noise_is_block_diagonal(self):
        model = Joint(Oscillator(), {"damping": {"sigma": 0.5, "drift": 1e-3}})
        np.testing.assert_allclose(model.Q(), np.diag([1e-6, 1e-4, 1e-6]))
        np.testing.assert_array_equal(model.R(), [[1e-2]])

    def test_refusals(self):
        with self.assertRaises(KeyError):
            Joint(Oscillator(), {"mass": (1.0, 0.0)})
        with self.assertRaises(ValueError):
            Joint(Oscillator(), {})
        with self.assertRaises(ValueError):
            Joint(Oscillator(), {"damping": (0.0, 0.0)})
        with self.assertRaises(ValueError):
            Joint(Oscillator(), {"damping": (1.0, -1.0)})
        with self.assertRaises(ValueError):
            Joint(Oscillator(), {"damping": 1.0})
        with self.assertRaises(ValueError):
            Joint(Oscillator(), {"damping": {"sigma": 1.0, "rate": 0.1}})
        with self.assertRaises(TypeError):
            Joint(object(), {"damping": (1.0, 0.0)})

    def test_a_parameter_read_only_at_construction_is_refused(self):
        with self.assertRaisesRegex(ValueError, "changes neither f nor h"):
            Joint(Frozen(), {"damping": (0.5, 0.0)})

    def test_initial_state_of_either_size(self):
        model = Joint(Oscillator(), {"damping": (0.5, 0.0)})
        model.initial_state([0.3, -0.2, 0.05])                      # the joint state
        np.testing.assert_array_equal(model.x0.ravel(), [0.3, -0.2, 0.05])
        model.initial_state([0.7, 0.1])                             # the inner one
        np.testing.assert_array_equal(model.x0.ravel(), [0.7, 0.1, 0.0])
        np.testing.assert_array_equal(model.inner.x0.ravel(), [0.7, 0.1])
        self.assertAlmostEqual(model.P0[2, 2], 0.25)


class TestMaps(unittest.TestCase):

    def setUp(self):
        self.inner = Oscillator(damping=0.0)
        self.model = Joint(self.inner, {"damping": (0.5, 1e-4)})
        self.x = np.array([[0.4], [-0.7], [0.3]])

    def test_f_uses_the_carried_parameter_and_restores_it(self):
        out = self.model.f(self.x)
        want = Oscillator(damping=0.3).f(self.x[:2])
        np.testing.assert_array_equal(out[:2], want)
        self.assertEqual(out[2, 0], 0.3)
        self.assertEqual(self.inner["damping"], 0.0)

    def test_noise_through_f(self):
        w = np.array([[0.1], [0.2], [0.05]])
        out = self.model.f(self.x, w=w)
        np.testing.assert_allclose(out.ravel(), (self.model.f(self.x) + w).ravel())

    def test_jacobian_blocks(self):
        dt = 1.0 / 50.0
        J = self.model.F(self.x)
        analytic = np.array([[1.0, dt, 0.0],
                             [-dt * 4.0, 1.0 - dt * 0.3, -dt * -0.7],
                             [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(J, analytic, atol=1e-7)
        self.assertEqual(J[2, 2], 1.0)
        self.assertEqual(self.inner["damping"], 0.0)
        H = self.model.H(self.x)
        np.testing.assert_allclose(H, [[1.0, 0.0, 0.0]], atol=1e-8)


class TestEstimation(unittest.TestCase):

    def test_damping_recovered_through_callbacks(self):
        truth = Oscillator(damping=0.3, random_seed=11)
        sample = truth.sample(60.0)
        model = Joint(Oscillator(damping=0.0), {"damping": (0.5, 1e-4)})
        self.assertIsNotNone(model.native_handle(), "the pass runs on the C++ model")
        rec = model.filter(sample.zk, smooth=False, reset=True)
        estimate, sigma = rec.xk[-1, 2], np.sqrt(rec.Pk[-1, 2, 2])
        self.assertLess(abs(estimate - 0.3), 3 * sigma)
        self.assertLess(sigma, 0.05)
        self.assertEqual(model.inner["damping"], 0.0)


@needs_apps
class TestBallisticCounterpart(unittest.TestCase):
    """The case of notebook 01, Step 7: the drag coefficient of a shell."""

    INIT = [[100, 100, 0], [200, 100, 0], [300, 100, -9.81]]
    COMMON = dict(order=3, fs=32, m=4.2, measurement_uncertainty=10, process_uncertainty=0.3)

    def make(self):
        model = Joint(BallisticModel("XYZ", c_drag=0.0, **self.COMMON), {"c_drag": (1e-3, 1e-6)})
        model.initial_state(self.INIT)
        return model

    def setUp(self):
        warnings.simplefilter("ignore")
        truth = BallisticModel("XYZ", c_drag=6e-4, random_seed=42, **self.COMMON)
        truth.initial_state(self.INIT)
        self.sample = truth.sample(12)

    def test_the_counterpart_and_the_callbacks_agree(self):
        with native.python_reference():                 # no counterpart: the inner model is called back
            python = self.make()
            self.assertIsNone(python.native_handle())
            py = python.filter(self.sample.zk, smooth=False, reset=True)
        model = self.make()
        self.assertIsNotNone(model.native_handle(), "the counterpart was expected to be used")
        cx = model.filter(self.sample.zk, smooth=False, reset=True)
        scale = np.abs(py.xk).max(axis=0)
        np.testing.assert_array_less(np.abs(py.xk - cx.xk).max(axis=0), 1e-7 * np.maximum(1.0, scale))
        self.assertLess(np.abs(py.Pk - cx.Pk).max() / np.abs(py.Pk).max(), 1e-6)

    def test_drag_recovered(self):
        rec = self.make().filter(self.sample.zk, smooth=False, reset=True)
        estimate, sigma = rec.xk[-1, 9], np.sqrt(rec.Pk[-1, 9, 9])
        self.assertLess(abs(estimate - 6e-4), 3 * sigma)
        self.assertLess(sigma, 1e-4)

    def test_a_changed_wrapper_is_not_handed_to_cxx(self):
        model = self.make()
        model.h = lambda x, v=None, t=None: Joint.h(model, x, v, t)
        self.assertIsNone(model.native_handle())

    def test_the_parameter_is_set_back_on_the_python_model(self):
        with native.python_reference():
            model = self.make()
        x = np.asarray(model.x0, dtype=float).copy()
        x[1] = x[4] = x[7] = 200.0                      # moving: drag acts
        x[9] = 6e-4
        slow = model.f(x)
        x[9] = 0.0
        fast = model.f(x)
        self.assertLess(slow[7], fast[7])
        self.assertEqual(model.inner["c_drag"], 0.0)


@needs_core
class TestCAPI(unittest.TestCase):

    def test_refusals_are_reported(self):
        lib, ffi = core.lib, core.ffi
        inner = lib.kalman_model_create_matrices(b"matrices", 2, 1, 0)     # declares no parameter
        self.assertNotEqual(inner, ffi.NULL)
        try:
            encoded = [ffi.new("char[]", b"damping")]
            names = ffi.new("const char *[]", encoded)
            sigma, drift = ffi.new("double[]", [0.5]), ffi.new("double[]", [0.0])
            handle = lib.kalman_model_create_joint(inner, 1, names, sigma, drift)
            self.assertEqual(handle, ffi.NULL)
            self.assertIn("damping", core.last_error())
            self.assertEqual(lib.kalman_model_create_joint(ffi.NULL, 1, names, sigma, drift), ffi.NULL)
            self.assertEqual(lib.kalman_model_create_joint(inner, 0, names, sigma, drift), ffi.NULL)
        finally:
            lib.kalman_model_destroy(inner)


if __name__ == "__main__":
    unittest.main()

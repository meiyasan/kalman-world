"""A model written in Python, filtered by the C++ engine (Kalman::Models::Callback).

The C++ filter calls back into the Python model for ``f``, ``h`` and their
Jacobians. These tests pin that the result is the Python loop's, that missing
measurements are handled alike, and that a Python exception inside a callback
comes back as that exception.

Skipped when libKalman is not built.
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.models.native import run_callback_filter
from modules.framework.models.extended import ExtendedKalmanModel
from modules.apps.weather.storms import StormModel, synthetic_track
from modules.apps.gps import GPSModel


class Pendulum(ExtendedKalmanModel):
    """Damped pendulum written in Python only; the angle is observed."""

    def __init__(self, **kw):
        self.n, self.m, self.l = 2, 1, 0
        self.dtype = np.float64
        super().__init__("pendulum", Q=np.diag([1e-6, 1e-4]), R=np.array([[1e-2]]), L=np.eye(2),
                         fs=50.0, x0=np.array([[0.5], [0.0]]), P0=np.eye(2), **kw)

    def f(self, x, u=None, w=None, t=None):
        th, om = np.asarray(x, float).reshape(-1)
        dt = 1.0 / self.fs
        return np.array([[th + dt * om], [om + dt * (-9.81 * np.sin(th) - 0.1 * om)]])

    def h(self, x, v=None, t=None):
        return np.asarray(x, float).reshape(-1)[:1].reshape(1, 1)


def python_loop(make, z):
    was, native.USE_NATIVE_TWINS = native.USE_NATIVE_TWINS, False
    try:
        record = make().filter(z)
    finally:
        native.USE_NATIVE_TWINS = was
    return np.asarray(record.xk), np.asarray(record.Pk)


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestPythonModelOnTheCxxEngine(unittest.TestCase):

    def agree(self, make, z, rtol=1e-9):
        x, P = python_loop(make, z)
        out = run_callback_filter(make(), z)
        scale_x, scale_P = max(1.0, np.nanmax(np.abs(x))), max(1.0, np.nanmax(np.abs(P)))
        self.assertLess(np.nanmax(np.abs(x - out["x_post"])) / scale_x, rtol)
        self.assertLess(np.nanmax(np.abs(P - out["P_post"])) / scale_P, 100 * rtol)

    def test_pendulum_matches_the_python_loop(self):
        rng = np.random.default_rng(0)
        t = np.arange(400) / 50.0
        z = (0.5 * np.cos(np.sqrt(9.81) * t) * np.exp(-0.05 * t) + rng.normal(0, 0.1, 400)).reshape(-1, 1)
        self.agree(Pendulum, z)

    def test_missing_measurements_are_skipped_alike(self):
        _, fixes = synthetic_track(n_steps=200, seed=3)
        fixes[50:60] = np.nan
        fixes[120, 2] = np.nan
        self.agree(lambda: StormModel("TC", omega=-0.022), fixes, rtol=1e-7)

    def test_analytic_jacobian_and_time_dependent_measurement(self):
        """GPS: h and H depend on t (the satellites move) and H is analytic."""
        make = lambda: GPSModel("rx", apply_relativistic=False)
        z = np.asarray(make().simulate(120.0, seed=0))
        self.agree(make, z, rtol=1e-9)

    def test_a_python_exception_comes_back_as_itself(self):
        class Broken(Pendulum):
            def h(self, x, v=None, t=None):
                raise ZeroDivisionError("the sensor fell off")
        with self.assertRaises(ZeroDivisionError) as caught:
            run_callback_filter(Broken(), np.zeros((5, 1)))
        self.assertIn("sensor fell off", str(caught.exception))

    def test_the_model_object_is_not_kept_alive_by_the_library(self):
        import gc, weakref
        model = Pendulum()
        ref = weakref.ref(model)
        run_callback_filter(model, np.zeros((5, 1)))
        del model
        gc.collect()
        self.assertIsNone(ref(), "the callback closures were released with the handle")


if __name__ == "__main__":
    unittest.main()

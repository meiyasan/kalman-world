"""An application model and its C++ twin (libKalmanApps) are one model.

A Python model that mixes in :class:`NativeTwin` hands its filtering pass to
the C++ class registered under :meth:`native_name`. These tests pin the two
things that makes safe: the twin gives the Python answer, and every
configuration without a C++ counterpart keeps the Python physics.

Skipped when libKalmanApps is not built (``make build``).
"""
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.models.native import NativeTwin, apps_library_names, native_model_classes
from modules.framework.filters import UnscentedKalmanFilter
from modules.apps.weather.storms import StormModel, synthetic_track, analytic_steering
from modules.apps.ballistics import BallisticModel

HAVE_APPS = bool(apps_library_names())
needs_apps = unittest.skipUnless(HAVE_APPS, "libKalmanApps is not built")


def both(make, z, **kw):
    """(record with the Python physics, record on the C++ twin)."""
    native.USE_NATIVE_TWINS = False
    try:
        python = make().filter(z, **kw)
    finally:
        native.USE_NATIVE_TWINS = True
    model = make()
    assert model.native_handle() is not None, "the twin was expected to be used"
    return python, model.filter(z, **kw)


# How close "the same answer" is. One call of f agrees to the last digit or two
# (numpy and the C math library round a cosine differently), and both sides then
# difference f with a step of 1e-6, which turns that last digit into ~1e-9 of
# noise in each Jacobian. Over a few hundred steps that is ~1e-8 on the state and
# ~1e-7 on the covariance, relative to the largest entry.
STATE_RTOL = 1e-7
COVARIANCE_RTOL = 1e-6


def close(test, a, b, rtol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(1.0, float(np.nanmax(np.abs(a))))
    test.assertLess(float(np.nanmax(np.abs(a - b))) / scale, rtol, what)


def trajectory(n=400, fs=100.0, seed=0, drop=17):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    z = np.c_[30 * t, 10 * t, 120 * t - 0.5 * 9.81 * t**2] + rng.normal(0, 0.5, (n, 3))
    if drop:
        z[::drop] = np.nan
    return z


@needs_apps
class TestStormTwin(unittest.TestCase):

    def test_twin_matches_python_in_both_hemispheres(self):
        for hemisphere, omega in (("N", 0.0), ("N", -0.022), ("S", 0.0), ("S", 0.022)):
            _, fixes = synthetic_track(n_steps=300, hemisphere=hemisphere, seed=2)
            py, cx = both(lambda: StormModel("TC", hemisphere=hemisphere, omega=omega), fixes)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, {hemisphere} omega={omega}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {hemisphere} omega={omega}")

    def test_track_only_model_has_its_own_twin(self):
        _, fixes = synthetic_track(n_steps=200, seed=4)
        model = StormModel("TC", observe_intensity=False)
        self.assertEqual(model.native_name(), "storm-track-only")
        py, cx = both(lambda: StormModel("TC", observe_intensity=False), fixes[:, :2])
        close(self, py.xk, cx.xk, STATE_RTOL, "state")

    def test_missing_fixes_are_skipped_alike(self):
        _, fixes = synthetic_track(n_steps=200, seed=5)
        fixes[40:55] = np.nan                       # a satellite gap
        fixes[100, 2] = np.nan                      # one channel missing
        py, cx = both(lambda: StormModel("TC"), fixes)
        close(self, py.xk, cx.xk, STATE_RTOL, "state through the gap")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, "covariance through the gap")

    def test_smoothing_runs_on_the_twin_forward_pass(self):
        _, fixes = synthetic_track(n_steps=200, seed=6)
        py, cx = both(lambda: StormModel("TC", omega=-0.022), fixes, smooth=True)
        close(self, py.xk, cx.xk, COVARIANCE_RTOL, "smoothed state")

    def test_a_change_after_construction_reaches_the_twin(self):
        _, fixes = synthetic_track(n_steps=200, seed=7)

        def make():
            model = StormModel("TC")
            model.omega = -0.03                     # set after __init__
            model._Q = model._Q * 4.0
            return model
        py, cx = both(make, fixes)
        close(self, py.xk, cx.xk, STATE_RTOL, "state after changing omega and Q")
        straight = StormModel("TC").filter(fixes)
        self.assertGreater(float(np.max(np.abs(np.asarray(straight.xk) - np.asarray(cx.xk)))), 1e-3)

    def test_a_second_run_starts_from_the_model_again(self):
        _, fixes = synthetic_track(n_steps=120, seed=8)
        model = StormModel("TC")
        first = np.asarray(model.filter(fixes).xk).copy()
        second = np.asarray(model.filter(fixes).xk)
        np.testing.assert_array_equal(first, second)

    def test_python_callbacks_keep_the_python_physics(self):
        self.assertIsNone(StormModel("TC", steering=analytic_steering).native_handle())
        self.assertIsNone(StormModel("TC", intensity_fn=lambda lon, lat, I, t: 0.0).native_handle())

    def test_a_subclass_that_changes_the_dynamics_is_not_the_twin(self):
        class Stalled(StormModel):
            def f(self, x, u=None, w=None, t=None):
                return np.asarray(x, float).reshape(-1, 1)
        self.assertIsNone(Stalled("TC").native_handle())

        class Renamed(StormModel):                  # same physics, another name: still the twin
            NAME = "Renamed"
        self.assertIsNotNone(Renamed("TC").native_handle())

    def test_another_engine_keeps_the_python_loop(self):
        _, fixes = synthetic_track(n_steps=60, seed=9)
        model = StormModel("TC")
        flt = UnscentedKalmanFilter(model)
        self.assertIsNone(flt._native_twin(None))
        self.assertEqual(len(flt.forward(fixes).xk), 60)

    def test_switch_and_environment_turn_twins_off(self):
        model = StormModel("TC")
        native.USE_NATIVE_TWINS = False
        try:
            self.assertIsNone(model.native_handle())
        finally:
            native.USE_NATIVE_TWINS = True
        os.environ["KALMAN_NATIVE_TWINS"] = "0"
        try:
            self.assertIsNone(model.native_handle())
        finally:
            del os.environ["KALMAN_NATIVE_TWINS"]
        self.assertIsNotNone(model.native_handle())


class TestSmootherUsesEachStepsTransition(unittest.TestCase):
    """The fused RTS kernel takes one F for the whole run, so it may only be used
    when F is constant. A nonlinear model's Jacobian is not: smoothing it with the
    first step's F moved the smoothed storm track by more than a degree."""

    def smoothed(self, make, z, accel):
        import modules.framework.native as accelerated
        was, native.USE_NATIVE_TWINS = native.USE_NATIVE_TWINS, False
        had, accelerated.HAS_C_ACCEL = accelerated.HAS_C_ACCEL, accel and accelerated.HAS_C_ACCEL
        try:
            return np.asarray(make().filter(z, smooth=True).xk)
        finally:
            native.USE_NATIVE_TWINS, accelerated.HAS_C_ACCEL = was, had

    def test_nonlinear_models_smooth_the_same_with_and_without_the_kernels(self):
        _, fixes = synthetic_track(n_steps=200, seed=6)
        cases = ((lambda: StormModel("TC", omega=-0.022), fixes),
                 (lambda: BallisticModel("b", fs=100.0, c_drag=0.05), trajectory(drop=0)))
        for make, z in cases:
            close(self, self.smoothed(make, z, True), self.smoothed(make, z, False), 1e-6,
                  "smoothed state, C++ kernels vs pure Python")

    def test_a_constant_transition_still_takes_the_fused_smoother(self):
        from modules.apps.vision import DetectionModel
        from modules.framework.filters import LinearKalmanFilter
        rng = np.random.default_rng(0)
        z = np.cumsum(rng.normal(0, 1, (200, 2)), 0)
        flt = LinearKalmanFilter(DetectionModel("b", fs=25, frame=(320, 240)))
        flt.forward(z)
        import modules.framework.native as accelerated
        self.assertEqual(flt._native_backward_eligible(), accelerated.HAS_C_ACCEL)


@needs_apps
class TestBallisticTwin(unittest.TestCase):

    def test_twin_matches_python_with_and_without_drag(self):
        z = trajectory()
        for kw in ({}, {"c_drag": 0.01}, {"c_drag": 0.05, "m": 2.0}, {"order": 2}, {"order": 2, "c_drag": 0.01}):
            py, cx = both(lambda: BallisticModel("b", fs=100.0, **kw), z)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, {kw}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {kw}")

    def test_orbital_gravity(self):
        rng = np.random.default_rng(1)
        r0 = 6.771e6
        v0 = np.sqrt(6.674e-11 * 5.972e24 / r0)
        t = np.arange(300) / 10.0
        w = v0 / r0
        z = np.c_[r0 * np.cos(w * t), r0 * np.sin(w * t), 0 * t] + rng.normal(0, 50.0, (300, 3))

        def make():
            model = BallisticModel("o", fs=10.0, g=None, measurement_uncertainty=50.0)
            model.initial_state([r0, 0, 0, 0, v0, 0, 0, 0, 0])
            return model
        self.assertEqual(make().native_parameters()["surface_gravity"], 0.0)
        py, cx = both(make, z)
        close(self, py.xk, cx.xk, STATE_RTOL, "state on a circular orbit")

    def test_features_without_a_twin_keep_the_python_physics(self):
        self.assertIsNone(BallisticModel("b", drag_uncertainty=0.3, c_drag=0.01).native_handle())
        custom = BallisticModel("b", f=lambda x, u=None, w=None, t=None: x)
        self.assertIsNone(custom.native_handle())
        self.assertIsNotNone(BallisticModel("b").native_handle())


@needs_apps
class TestTwinContract(unittest.TestCase):

    def test_twins_stay_out_of_the_plugin_catalogue(self):
        """StormModel is the catalogue entry; its twin is not listed a second time."""
        StormModel("TC").native_handle()            # make sure the library is loaded
        listed = set(native_model_classes())
        self.assertFalse(listed & apps_library_names(), listed & apps_library_names())
        self.assertTrue({"storm", "ballistic"} <= apps_library_names())

    def test_parameters_cross_by_name(self):
        model = StormModel("TC", omega=-0.02, beta_drift=4.0, hemisphere="S", dt_hours=3.0)
        handle = model.native_handle()
        lib = core.lib
        names = [core._s(lib.kalman_model_parameter_name(handle, i))
                 for i in range(lib.kalman_model_parameter_count(handle))]
        self.assertEqual(names, ["beta_drift", "dt_hours", "hemisphere", "omega"])
        value = core.ffi.new("double *")
        for key, want in model.native_parameters().items():
            core.check(lib.kalman_model_parameter(handle, key.encode(), value), key)
            self.assertEqual(value[0], want)
        self.assertEqual(lib.kalman_model_set_parameter(handle, b"basin", 1.0), -1)
        self.assertIn("basin", core.last_error())

    def test_a_model_without_native_name_is_never_a_twin(self):
        class Plain(NativeTwin):
            pass
        self.assertIsNone(Plain().native_name())


if __name__ == "__main__":
    unittest.main()

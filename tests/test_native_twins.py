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
from modules.apps.weather.storms import (StormModel, synthetic_track, analytic_steering, LearnedIntensityModel,
                                         synthetic_steered_track,
                                         TornadoEnvironmentModel, synthetic_environment)
from modules.apps.ballistics import BallisticModel
from modules.apps.vehicles import CarSpeedModel
from modules.framework.constants import KalmanIntegratorKind
from modules.apps.astro.space import TransLunarModel, InterplanetaryModel

HAVE_APPS = bool(apps_library_names())
needs_apps = unittest.skipUnless(HAVE_APPS, "libKalmanApps is not built")


def both(make, z, **kw):
    """(record with the Python physics, record on the C++ twin)."""
    with native.python_reference():
        python = make().filter(z, **kw)
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

    def test_a_learned_intensity_tendency_runs_on_the_twin(self):
        # notebook 14's regression, fitted on steered tracks, then used without steering
        tracks = [synthetic_steered_track(n_steps=40, lon0=150 + 3 * k, lat0=8 + k, seed=k)[0] for k in range(6)]
        learned = LearnedIntensityModel().fit(tracks)
        _, fixes = synthetic_track(n_steps=200, seed=9)
        fixes[60:70] = np.nan
        make = lambda: StormModel("TC", intensity_fn=learned.as_intensity_fn())
        self.assertEqual(make().native_name(), "storm")
        py, cx = both(make, fixes)
        close(self, py.xk, cx.xk, STATE_RTOL, "state with a learned intensity")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, "covariance with a learned intensity")
        persistence = StormModel("TC").filter(fixes)
        self.assertGreater(float(np.max(np.abs(np.asarray(persistence.xk)[:, 4] - np.asarray(cx.xk)[:, 4]))), 1.0)
        model = make()                              # and the twin forgets it again
        model.native_handle()
        model.intensity_fn = None
        close(self, np.asarray(model.filter(fixes).xk), np.asarray(persistence.xk), STATE_RTOL, "persistence again")

    def test_another_intensity_environment_keeps_the_python_physics(self):
        tracks = [synthetic_steered_track(n_steps=30, seed=k)[0] for k in range(3)]
        learned = LearnedIntensityModel(env=lambda lon, lat, t=None: (1.0, 2.0)).fit(tracks)
        self.assertIsNone(StormModel("TC", intensity_fn=learned.as_intensity_fn()).native_handle())

        class Wider(LearnedIntensityModel):
            @staticmethod
            def _features(sst_warmth, shear, I):
                return np.array([sst_warmth, shear, I ** 0.5])
        wider = Wider().fit(tracks)
        self.assertIsNone(StormModel("TC", intensity_fn=wider.as_intensity_fn()).native_handle())

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
        had, accelerated.HAS_C_ACCEL = accelerated.HAS_C_ACCEL, accel and accelerated.HAS_C_ACCEL
        try:
            with native.python_reference():
                return np.asarray(make().filter(z, smooth=True).xk)
        finally:
            accelerated.HAS_C_ACCEL = had

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
        from modules.framework.integrators import DormandPrinceIntegrator, RK4Integrator
        self.assertIsNone(BallisticModel("b", integrator=DormandPrinceIntegrator(rtol=1e-9)).native_handle())
        class Mine(RK4Integrator):
            def _step(self, f, x, u=None, t=None, dt=1.0):
                return super()._step(f, x, u=u, t=t, dt=dt)
        self.assertIsNone(BallisticModel("b", integrator=Mine()).native_handle())
        self.assertIsNone(BallisticModel("b", g=None, substeps="auto").native_handle())
        custom = BallisticModel("b", f=lambda x, u=None, w=None, t=None: x)
        self.assertIsNone(custom.native_handle())
        self.assertIsNotNone(BallisticModel("b").native_handle())

    def test_substeps_and_integrator_reach_the_twin(self):
        from modules.framework.constants import KalmanIntegratorKind
        for kw in (dict(substeps=4), dict(integrator=KalmanIntegratorKind.EULER),
                   dict(integrator=KalmanIntegratorKind.DORMAND_PRINCE), dict(substeps=3, c_drag=0.02)):
            def make():
                model = BallisticModel("b", fs=20.0, **kw)
                model.initial_state([0, 30, 0, 0, 20, 0, 100, 40, -9.81])
                return model
            self.assertIsNotNone(make().native_handle(), str(kw))
            z = make().sample(200, seed=3).zk
            py, cx = both(make, z)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state with {kw}")


@needs_apps
class TestTornadoTwin(unittest.TestCase):

    def test_twin_matches_python_in_both_regimes(self):
        obs, _ = synthetic_environment(n_steps=300, onsets=((90, 160),), seed=3)
        for mu, persistence in ((0.3, 0.85), (3.0, 0.85), (3.0, 0.5)):
            py, cx = both(lambda: TornadoEnvironmentModel("ENV", mu=mu, persistence=persistence), obs)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, mu={mu} a={persistence}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, mu={mu} a={persistence}")

    def test_the_bank_runs_on_twins(self):
        models, _, _ = TornadoEnvironmentModel.ikf_bank(season="peak")
        self.assertTrue(all(m.native_handle() is not None for m in models))


@needs_apps
class TestCarSpeedTwin(unittest.TestCase):

    def run_of(self, **kw):
        car = CarSpeedModel("CAR", **kw)
        return np.asarray(car.sample(15.0, seed=1).zk)

    def test_twin_matches_python_through_the_grip_to_power_handoff(self):
        for kw in ({}, {"power": 300_000.0, "mass": 1200.0}, {"grade": 0.05, "mu": 0.6}, {"fs": 50.0}):
            z = self.run_of(**kw)
            py, cx = both(lambda: CarSpeedModel("CAR", **kw), z)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, {kw}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {kw}")
            self.assertGreater(float(np.max(np.asarray(cx.xk)[:, 1])), 20.0, "the car got up to speed")

    def test_another_integrator_keeps_the_python_physics(self):
        class Eulered(CarSpeedModel):
            DEFAULT_INTEGRATOR = KalmanIntegratorKind.EULER
        self.assertIsNone(Eulered("CAR").native_handle())

        class Substepped(CarSpeedModel):
            SOLVER_SUBSTEPS = 4
        self.assertIsNone(Substepped("CAR").native_handle())
        self.assertIsNotNone(CarSpeedModel("CAR").native_handle())


# --- the lane-keeping car and the rocket family ----------------------------------
from modules.apps.vehicles import CarLaneModel                          # noqa: E402
from modules.apps.astro.rocket import (RocketLiftoffModel, RocketAscentModel,  # noqa: E402
                                       RocketDescentModel, RocketLandingModel)
from modules.runtime.astro import R_EARTH, KM                           # noqa: E402


def steering(n, fs=50.0):
    """A lane-change steering program, past the actuator limit for half a second."""
    t = np.arange(n) / fs
    u = 0.01 * np.sin(0.8 * t) + 0.15 * ((t > 3.0) & (t < 3.5)) - 0.15 * ((t > 3.5) & (t < 4.0))
    return u.reshape(-1, 1)


@needs_apps
class TestCarLaneTwin(unittest.TestCase):

    def make(self, **kw):
        return lambda: CarLaneModel("camera", fs=50, x0=[[0.3], [0.0], [15.0]], lateral_uncertainty=0.05, **kw)

    def test_twin_matches_python_with_the_steering_program(self):
        uk = steering(500)
        for kw in ({}, {"q_lat": 2.5e-3, "q_head": 1e-4}, {"wheelbase": 3.5, "steer_limit": 0.05}):
            z = self.make(**kw)().sample(10.0, uk=uk, seed=1).zk
            py, cx = both(self.make(**kw), z, uk=uk)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, {kw}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {kw}")
            self.assertGreater(float(np.ptp(np.asarray(cx.xk)[:, 1])), 0.01, "the car steered")

    def test_integrator_and_substeps_reach_the_twin(self):
        uk = steering(300)
        for kw in (dict(integrator=KalmanIntegratorKind.RK4), dict(substeps=4)):
            def make():
                model = self.make()()
                if "integrator" in kw:
                    model._integrator_override, model._integrator_inst = kw["integrator"], None
                return model
            if "substeps" in kw:
                class Substepped(CarLaneModel):
                    SOLVER_SUBSTEPS = 4
                make = lambda: Substepped("camera", fs=50, x0=[[0.3], [0.0], [15.0]])
            self.assertIsNotNone(make().native_handle(), str(kw))
            z = make().sample(6.0, uk=uk, seed=2).zk
            py, cx = both(make, z, uk=uk)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state with {kw}")

    def test_what_stays_in_python(self):
        class Wider(CarLaneModel):                      # another actuator limit
            def clip_u(self, u):
                return np.asarray(u, float).reshape(-1, 1)
        self.assertIsNone(Wider("camera").native_handle())

        class Drifting(CarLaneModel):                   # another derivative
            def f(self, x, u=None, w=None, t=None):
                return np.zeros((3, 1))
        self.assertIsNone(Drifting("camera").native_handle())

        from modules.framework.integrators import RK4Integrator
        class Mine(RK4Integrator):
            def _step(self, f, x, u=None, t=None, dt=1.0):
                return super()._step(f, x, u=u, t=t, dt=dt)
        model = CarLaneModel("camera")
        model._integrator_override, model._integrator_inst = Mine(), None
        self.assertIsNone(model.native_handle())
        self.assertIsNotNone(CarLaneModel("camera").native_handle())

    def test_a_controller_function_keeps_the_python_model(self):
        from modules.framework.filters import ExtendedKalmanFilter
        flt = ExtendedKalmanFilter(CarLaneModel("camera"))
        self.assertIsNone(flt._native_twin(lambda x, P, t, nu=None: np.zeros((1, 1))))


def liftoff_model():
    model = RocketLiftoffModel("LIFTOFF", m0=549_000.0, m_dry=138_000.0, ve=2900.0, mdot=2600.0,
                               drag_area=8.0, length=70.0, width=3.7, fs=10.0, track_m=50.0, q_mass=0.0)
    model.initial_state([0.0, 0.0, model.m0], P0=np.diag([50.0**2, 10.0**2, 50.0**2]))
    return model


def ascent_program(n=240):
    uk = np.zeros((n, 2))
    for i in range(n):
        pitch = np.deg2rad(89.0 - 111.0 * (i / (n - 1)) ** 0.57)
        uk[i] = [19.4 * np.sin(pitch), 19.4 * np.cos(pitch)]
    return uk


def ascent_model(**kw):
    model = RocketAscentModel(dt_s=2.0, track_km=1.0, **kw)
    model.initial_state(np.array([[R_EARTH], [0], [0], [0]]),
                        P0=np.diag([(1 * KM)**2, (1 * KM)**2, 0.01**2, 0.01**2]))
    return model


BOOSTER = dict(m0=30_000.0, m_dry=22_000.0, thrust_max=845e3, drag_area=10.0)
BOOSTER_X0 = np.array([350.0, 2000.0, -20.0, -60.0, 0.05, 0.0, 30_000.0])


def descent_model(**kw):
    model = RocketDescentModel(fs=20.0, **BOOSTER, **kw)
    model.initial_state(BOOSTER_X0)
    return model


def landing_model(**kw):
    model = RocketLandingModel("LANDING", **kw)
    model.initial_state([3000.0, -150.0, 1000.0], P0=np.diag([20.0**2, 5.0**2, 10.0**2]))
    return model


@needs_apps
class TestRocketTwins(unittest.TestCase):
    """The scenarios of share/notebooks/06_RocketScience.ipynb, seeded."""

    def check(self, make, z, what, **kw):
        py, cx = both(make, z, **kw)
        close(self, py.xk, cx.xk, STATE_RTOL, f"state, {what}")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {what}")
        return cx

    def test_liftoff_through_burnout_and_coast(self):
        z = liftoff_model().sample(420.0, seed=3).zk
        cx = self.check(liftoff_model, z, "liftoff")
        self.assertEqual(float(np.asarray(cx.xk)[-1, 2]), 138_000.0, "burnout: the dry mass")
        z = RocketLiftoffModel("L", drag=False).sample(120.0, seed=4).zk
        self.check(lambda: RocketLiftoffModel("L", drag=False), z, "liftoff without drag")

    def test_ascent_with_the_thrust_program(self):
        uk = ascent_program()
        for kw in ({}, {"drag": True}):
            z = ascent_model(**kw).sample(240 / 500.0, uk=uk, seed=3).zk
            self.check(lambda: ascent_model(**kw), z, f"ascent {kw}", uk=uk)

    def test_descent_guided_and_driven(self):
        z = descent_model().sample(70.0, seed=3).zk
        cx = self.check(descent_model, z, "descent, guidance")
        self.assertLess(abs(float(np.asarray(cx.xk)[-1, 1])), 0.5, "the booster is on the pad")
        ctl = lambda: descent_model(control=True)
        uk, _ = ctl().autopilot(BOOSTER_X0, 1400)
        z = ctl().sample(70.0, uk=uk, process_noise=False, seed=3).zk
        self.assertEqual(ctl().native_name(), "rocket-descent-control")
        self.check(ctl, z, "descent, throttle and gimbal as the control", uk=uk)

    def test_landing_burn_to_the_pad(self):
        for kw in ({}, {"drag": False, "kp": 2.0}):
            z = landing_model(**kw).sample(40.0, seed=3).zk
            cx = self.check(lambda: landing_model(**kw), z, f"landing {kw}")
            self.assertLess(abs(float(np.asarray(cx.xk)[-1, 0])), 0.5, "landed")

    def test_substeps_and_integrator_reach_the_twins(self):
        class Fine(RocketLandingModel):
            SOLVER_SUBSTEPS = 3
        class Euler(RocketLiftoffModel):
            DEFAULT_INTEGRATOR = KalmanIntegratorKind.EULER
        for make in (lambda: Fine("LANDING", x0=[[3000.0], [-150.0], [1000.0]]), lambda: Euler("L")):
            self.assertIsNotNone(make().native_handle())
            z = make().sample(30.0, seed=5).zk
            self.check(make, z, type(make()).__name__)

    def test_what_stays_in_python(self):
        from modules.framework.integrators import RK4Integrator
        class Mine(RK4Integrator):
            def _step(self, f, x, u=None, t=None, dt=1.0):
                return super()._step(f, x, u=u, t=t, dt=dt)
        for cls in (RocketLiftoffModel, RocketLandingModel, RocketDescentModel):
            model = cls()
            model._integrator_override, model._integrator_inst = Mine(), None
            self.assertIsNone(model.native_handle(), cls.__name__)
        ascent = RocketAscentModel()
        ascent._integrator_override, ascent._integrator_inst = Mine(), None
        self.assertIsNone(ascent.native_handle())

        class Harder(RocketLandingModel):               # another guidance law
            def guidance(self, x):
                return 0.0
        self.assertIsNone(Harder().native_handle())

        class Stiff(RocketDescentModel):                # another inertia
            def inertia(self, m):
                return m
        self.assertIsNone(Stiff().native_handle())

        class Bounce(RocketLiftoffModel):               # another closing step
            def after_step(self, x, u=None, t=None):
                return x
        self.assertIsNone(Bounce().native_handle())

        class Drag(RocketAscentModel):                  # another acceleration
            def acceleration(self, r, v, u):
                return np.zeros(2)
        self.assertIsNone(Drag().native_handle())
        for cls in (RocketLiftoffModel, RocketLandingModel, RocketDescentModel, RocketAscentModel):
            self.assertIsNotNone(cls().native_handle(), cls.__name__)


@needs_apps
class TestTwinContract(unittest.TestCase):

    def test_twins_stay_out_of_the_plugin_catalogue(self):
        """StormModel is the catalogue entry; its twin is not listed a second time."""
        StormModel("TC").native_handle()            # make sure the library is loaded
        listed = set(native_model_classes())
        self.assertFalse(listed & apps_library_names(), listed & apps_library_names())
        self.assertTrue({"storm", "ballistic", "tornado-environment", "car-speed"} <= apps_library_names())
        self.assertTrue({"car-lane", "rocket-liftoff", "rocket-ascent", "rocket-descent",
                         "rocket-descent-control", "rocket-landing"} <= apps_library_names())

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


# --- the space models (kalman.apps.astro.space) ----------------------------------
def free_return(**kw):
    """The notebook's Apollo-style free return: a 400 km parking orbit, then TLI."""
    from modules.runtime.astro import R_EARTH, KM, D_MOON, GM_EARTH
    r0 = R_EARTH + 400 * KM
    v = np.sqrt(GM_EARTH * (2.0 / r0 - 1.0 / (0.5 * (r0 + D_MOON)))) + 0.034
    model = TransLunarModel(**{"moon_phase_deg": 133.0, "sun_phase_deg": 60.0, "dt_s": 180.0,
                               "track_km": 5.0, "n_sub": 4, **kw})
    model.initial_state(np.array([[r0], [0], [0], [0], [v], [0]]),
                        P0=np.diag([(50 * KM) ** 2] * 3 + [0.01 ** 2] * 3))
    return model


def mars_cruise(**kw):
    """The notebook's Earth-Mars cruise, seeded by a Lambert arc of the 2026 window."""
    from astropy.time import Time, TimeDelta
    from modules.apps.astro.space.transfers import lambert
    from modules.runtime.astro import body_state, DAY
    dep, tof = Time("2026-10-28"), 311.0
    r1, _ = body_state("earth", dep)
    r2, _ = body_state("mars", dep + TimeDelta(tof * 86400, format="sec"))
    v1, _ = lambert(r1, r2, tof * DAY)
    model = InterplanetaryModel(**{"departure": "earth", "target": "mars", "epoch": dep,
                                   "span_days": tof + 20, "dt_s": 43200.0, "track_Mm": 0.02, **kw})
    model.initial_state(np.r_[r1, v1].reshape(-1, 1), P0=np.diag([1.0] * 3 + [1e-4] * 3))
    return model, tof * DAY


@needs_apps
class TestTransLunarTwin(unittest.TestCase):

    def test_twin_matches_python_through_the_lunar_flyby(self):
        from modules.runtime.astro import DAY
        np.random.seed(8)
        z = free_return().sample(4.5 * DAY, process_noise=False).zk      # out past the Moon
        for kw in ({}, {"bodies": ("earth", "moon"), "n_sub": 1}):
            py, cx = both(lambda: free_return(**kw), z)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, {kw}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {kw}")

    def test_ephemeris_phases_and_integrators_reach_the_twin(self):
        from modules.runtime.astro import DAY
        # (Dormand-Prince is left out of these runs for its time, not its result:
        # the framework probes f at x = 0, where |r| = 0 makes it step to its limit.)
        for kw in ({"use_ephemeris": True, "epoch": "2026-11-06"},
                   {"integrator": KalmanIntegratorKind.EULER, "n_sub": 8}):
            self.assertEqual(free_return(**kw).native_name(), "translunar", str(kw))
            z = free_return(**kw).sample(1.0 * DAY, seed=2).zk
            py, cx = both(lambda: free_return(**kw), z)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, {kw}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {kw}")

    def test_configurations_without_a_twin_keep_the_python_physics(self):
        from modules.framework.integrators import RK4Integrator

        class Mine(RK4Integrator):
            def _step(self, f, x, u=None, t=None, dt=1.0):
                return super()._step(f, x, u=u, t=t, dt=dt)
        self.assertIsNone(free_return(integrator=Mine()).native_handle())

        class Dragged(TransLunarModel):
            def acceleration(self, r, t):
                return super().acceleration(r, t) * 0.999
        self.assertIsNone(Dragged().native_handle())
        self.assertIsNotNone(free_return().native_handle())


@needs_apps
class TestInterplanetaryTwin(unittest.TestCase):

    def test_twin_matches_python_on_the_mars_cruise(self):
        for kw in ({}, {"perturbers": ("venus", "jupiter")}, {"epoch": None}, {"n_sub": 3}):
            model, span = mars_cruise(**kw)
            self.assertEqual(model.native_name(), "interplanetary", str(kw))
            np.random.seed(8)
            z = model.sample(span, process_noise=False).zk
            py, cx = both(lambda: mars_cruise(**kw)[0], z)
            close(self, py.xk, cx.xk, STATE_RTOL, f"state, {kw}")
            close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, f"covariance, {kw}")

    def test_the_table_is_held_at_its_ends(self):
        # A cruise longer than the span the perturbers were tabulated over.
        model, span = mars_cruise(span_days=100.0)
        z = model.sample(span, seed=4).zk
        py, cx = both(lambda: mars_cruise(span_days=100.0)[0], z)
        close(self, py.xk, cx.xk, STATE_RTOL, "state past the table")

    def test_configurations_without_a_twin_keep_the_python_physics(self):
        self.assertIsNone(InterplanetaryModel("earth", "mars", control=True).native_handle())
        model, _ = mars_cruise()
        model._field.t_grid = model._field.t_grid * 1.0000001     # not numpy.arange's grid
        self.assertIsNone(model.native_handle())

        class Thrusting(InterplanetaryModel):
            def acceleration(self, r, t):
                return super().acceleration(r, t) + 1e-9
        self.assertIsNone(Thrusting("earth", "mars").native_handle())
        self.assertIsNotNone(InterplanetaryModel("earth", "mars").native_handle())


if __name__ == "__main__":
    unittest.main()

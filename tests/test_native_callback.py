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
    with native.python_reference():
        record = make().filter(z)
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


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestFilterTakesTheCxxLoop(unittest.TestCase):
    """``model.filter()`` itself: a Python-written model with constant, additive
    noise runs its loop in C++; anything the C++ path does not reproduce keeps
    the Python loop."""

    def path(self, model, controller=None):
        from modules.framework.filter import KalmanFilter
        return KalmanFilter(model)._callback_eligible(controller)

    def test_filter_gives_the_reference_result_through_the_cxx_loop(self):
        rng = np.random.default_rng(1)
        z = rng.normal(0, 0.2, (300, 1))
        self.assertTrue(self.path(Pendulum()))
        x_ref, P_ref = python_loop(Pendulum, z)
        record = Pendulum().filter(z)
        self.assertLess(np.max(np.abs(np.asarray(record.xk) - x_ref)), 1e-9)
        self.assertLess(np.max(np.abs(np.asarray(record.Pk) - P_ref)), 1e-9)

    def test_smoothing_after_the_cxx_loop(self):
        rng = np.random.default_rng(2)
        z = rng.normal(0, 0.2, (200, 1))
        with native.python_reference():
            ref = np.asarray(Pendulum().filter(z, smooth=True).xk)
        got = np.asarray(Pendulum().filter(z, smooth=True).xk)
        self.assertLess(np.max(np.abs(got - ref)), 1e-8)

    def test_noise_that_is_not_a_constant_matrix_is_asked_for_each_step(self):
        """A state-dependent Q, a time-dependent R and a coupling left to be
        differenced from f all run on the C++ loop, and give the Python result."""
        class SpeedDependentQ(Pendulum):
            def Q(self, t=None, x=None):
                if x is None:
                    return self._Q
                return self._Q * (1.0 + float(np.asarray(x).reshape(-1)[1]) ** 2)

        class GrowingR(Pendulum):
            def R(self, t=None):
                return self._R * (1.0 + (t or 0.0))

        class NoCoupling(Pendulum):
            def __init__(self):
                super().__init__()
                self._L = None

            def f(self, x, u=None, w=None, t=None):
                out = super().f(x)
                return out if w is None else out + np.asarray(w, float).reshape(-1, 1)

        rng = np.random.default_rng(4)
        t = np.arange(300) / 50.0
        z = (0.5 * np.cos(3.1 * t) + rng.normal(0, 0.1, 300)).reshape(-1, 1)
        for make in (SpeedDependentQ, GrowingR, NoCoupling):
            self.assertFalse(native.constant_noise(make()), make.__name__)
            self.assertTrue(self.path(make()), make.__name__)
            x_ref, P_ref = python_loop(make, z)
            record = make().filter(z)
            self.assertLess(np.max(np.abs(np.asarray(record.xk) - x_ref)), 1e-9, make.__name__)
            self.assertLess(np.max(np.abs(np.asarray(record.Pk) - P_ref)), 1e-9, make.__name__)
        self.assertGreater(np.max(np.abs(np.asarray(SpeedDependentQ().filter(z).Pk)
                                         - np.asarray(Pendulum().filter(z).Pk))), 1e-6,
                           "the state-dependent Q is really used")

    def test_what_keeps_the_python_loop(self):
        self.assertFalse(self.path(Pendulum(), controller=lambda *a, **k: None), "a controller callback")
        with native.python_reference():
            self.assertFalse(self.path(Pendulum()), "the reference switch")
        from modules.framework.filters import EnsembleKalmanFilter

        class Field(Pendulum):
            def forecast(self, X, t=None):
                return X
        self.assertFalse(EnsembleKalmanFilter(Field())._callback_eligible(None),
                         "an ensemble model with a whole-ensemble hook")


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestUnscentedOnTheCxxEngine(unittest.TestCase):
    """The unscented filter on a Python-written model runs in C++ and gives the
    Python engine's result. A framework model's f takes w and its h takes v, so
    the noise is carried through the maps: sigma points over state and noise."""

    def both(self, make, z, **engine):
        from modules.framework.filters import UnscentedKalmanFilter
        with native.python_reference():
            ref = UnscentedKalmanFilter(make(), **engine).forward(z)
        flt = UnscentedKalmanFilter(make(), **engine)
        self.assertTrue(flt._callback_eligible(None))
        got = flt.forward(z)
        return (np.asarray(ref.xk), np.asarray(ref.Pk)), (np.asarray(got.xk), np.asarray(got.Pk))

    def close(self, a, b, rtol, what):
        self.assertLess(np.nanmax(np.abs(a - b)) / max(1.0, np.nanmax(np.abs(a))), rtol, what)

    def data(self):
        rng = np.random.default_rng(5)
        t = np.arange(300) / 50.0
        z = (0.5 * np.cos(3.1 * t) + rng.normal(0, 0.1, 300)).reshape(-1, 1)
        z[40:48] = np.nan
        return z

    def test_noise_carried_through_the_maps(self):
        (x0, P0), (x1, P1) = self.both(_NoisyPendulum, self.data(), alpha=0.5)
        self.close(x0, x1, 1e-7, "state")
        self.close(P0, P1, 1e-7, "covariance")

    def test_default_spread_on_application_models(self):
        """alpha = 1e-3, the default, has weights of order 1e6, so the last digits
        of f are amplified: agreement is to 1e-6 here, not to 1e-9."""
        _, fixes = synthetic_track(n_steps=200, seed=1)
        (x0, P0), (x1, P1) = self.both(lambda: StormModel("TC", omega=-0.022), fixes)
        self.close(x0, x1, 1e-6, "storm state")
        self.close(P0, P1, 1e-6, "storm covariance")


class _NoisyPendulum(Pendulum):
    """The pendulum with its noise inside the maps: a random torque, and a
    sensor error that grows with the angle."""

    def f(self, x, u=None, w=None, t=None):
        out = Pendulum.f(self, x)
        if w is not None:
            w = np.asarray(w, float).reshape(-1)
            out[1, 0] += w[1] * np.cos(float(np.asarray(x).reshape(-1)[0]))
            out[0, 0] += w[0]
        return out

    def h(self, x, v=None, t=None):
        out = Pendulum.h(self, x)
        if v is not None:
            out = out + float(np.asarray(v).reshape(-1)[0]) * (1.0 + 0.5 * out)
        return out


def _pendulum_run(seed, steps=150):
    """A pendulum trajectory with process noise, and noisy readings of its angle."""
    rng = np.random.default_rng(seed)
    model = Pendulum()
    x = np.array([[0.5], [0.0]])
    truth, z = [], []
    for _ in range(steps):
        x = model.f(x) + rng.multivariate_normal(np.zeros(2), model._Q).reshape(2, 1)
        truth.append(x.ravel())
        z.append(x[0, 0] + rng.normal(0, 0.1))
    return np.array(truth), np.array(z).reshape(-1, 1)


class TestDeclaredCouplingMeansAddedNoise(unittest.TestCase):
    """A model that declares L has said its process noise is added. Pendulum's f
    takes w and ignores it, as most application models do: reading the signature
    alone, the particle and unscented engines passed the noise into f, where it
    vanished, and the particle cloud collapsed onto one point."""

    def test_which_models_carry_noise_through_f(self):
        from modules.framework.engine import KalmanEngine
        self.assertFalse(KalmanEngine._noise_through_f(Pendulum()), "L given: added")
        undeclared = Pendulum()
        undeclared._L = None
        self.assertTrue(KalmanEngine._noise_through_f(undeclared), "no L: through f")

    def test_particle_cloud_keeps_its_spread(self):
        from modules.framework.filters import ParticleFilter
        truth, z = _pendulum_run(0)
        for reference in (True, False):
            np.random.seed(3)
            if reference:
                with native.python_reference():
                    record = ParticleFilter(Pendulum(), n_particles=400).forward(z)
            else:
                if not core.HAS_CXX_CORE:
                    continue
                record = ParticleFilter(Pendulum(), n_particles=400).forward(z)
            sigma = np.sqrt(np.asarray(record.Pk)[50:, 0, 0])
            error = np.asarray(record.xk)[50:, 0] - truth[50:, 0]
            self.assertGreater(float(sigma.min()), 5e-3, "the cloud did not collapse")
            self.assertGreater(float(np.mean(np.abs(error) < 3 * sigma)), 0.9, "errors inside 3 sigma")

    def test_unscented_covariance_feels_the_process_noise(self):
        from modules.framework.filters import UnscentedKalmanFilter

        class Noisier(Pendulum):
            def __init__(self):
                super().__init__()
                self._Q = self._Q * 100.0
        _, z = _pendulum_run(1)
        with native.python_reference():
            quiet = np.asarray(UnscentedKalmanFilter(Pendulum(), alpha=0.5).forward(z).Pk)[-1]
            noisy = np.asarray(UnscentedKalmanFilter(Noisier(), alpha=0.5).forward(z).Pk)[-1]
        self.assertGreater(noisy[1, 1], 5 * quiet[1, 1], "a hundred times the Q widens the estimate")


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestParticleOnTheCxxEngine(unittest.TestCase):
    """The particle filter on a Python-written model runs in C++. The two
    engines draw from different generators, so they are compared over many
    runs, not digit for digit."""

    def test_same_accuracy_and_spread_as_the_python_engine(self):
        from modules.framework.filters import ParticleFilter
        errors = {True: [], False: []}
        sigmas = {True: [], False: []}
        for run in range(24):
            truth, z = _pendulum_run(run)
            for reference in (True, False):
                np.random.seed(50 + run)
                if reference:
                    with native.python_reference():
                        record = ParticleFilter(Pendulum(), n_particles=400).forward(z)
                else:
                    flt = ParticleFilter(Pendulum(), n_particles=400)
                    self.assertTrue(flt._callback_eligible(None))
                    record = flt.forward(z)
                x, P = np.asarray(record.xk), np.asarray(record.Pk)
                errors[reference].append(np.sqrt(np.mean((x[40:, 0] - truth[40:, 0]) ** 2)))
                sigmas[reference].append(np.sqrt(np.mean(P[40:, 0, 0])))
        # The error of one run varies by about 40% of its mean, so the two means
        # are compared against their own standard errors (4 of them), and the
        # reported sigma, which hardly varies, to 10%.
        python, cxx = np.array(errors[True]), np.array(errors[False])
        spread = np.sqrt((python.var() + cxx.var()) / len(python))
        self.assertLess(abs(python.mean() - cxx.mean()), 4 * spread,
                        f"RMS error: Python {python.mean():.4f}, C++ {cxx.mean():.4f}")
        python, cxx = np.mean(sigmas[True]), np.mean(sigmas[False])
        self.assertLess(abs(python - cxx) / python, 0.1, f"reported sigma: Python {python:.4f}, C++ {cxx:.4f}")

    def test_a_seeded_run_repeats(self):
        from modules.framework.filters import ParticleFilter
        _, z = _pendulum_run(0)
        np.random.seed(11); first = np.asarray(ParticleFilter(Pendulum(), n_particles=200).forward(z).xk).copy()
        np.random.seed(11); again = np.asarray(ParticleFilter(Pendulum(), n_particles=200).forward(z).xk)
        np.testing.assert_array_equal(first, again)


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestEnsembleOnTheCxxEngine(unittest.TestCase):
    """The ensemble filter on a Python model with per-member f and h runs in C++.
    Different random generators, so the two engines are compared over many runs."""

    def test_same_accuracy_and_spread_as_the_python_engine(self):
        from modules.framework.filters import EnsembleKalmanFilter
        errors = {True: [], False: []}
        sigmas = {True: [], False: []}
        for run in range(24):
            truth, z = _pendulum_run(run)
            for reference in (True, False):
                if reference:
                    with native.python_reference():
                        record = EnsembleKalmanFilter(Pendulum(), n_members=60, seed=run).forward(z)
                else:
                    flt = EnsembleKalmanFilter(Pendulum(), n_members=60, seed=run)
                    self.assertTrue(flt._callback_eligible(None))
                    record = flt.forward(z)
                x, P = np.asarray(record.xk), np.asarray(record.Pk)
                errors[reference].append(np.sqrt(np.mean((x[40:, 0] - truth[40:, 0]) ** 2)))
                sigmas[reference].append(np.sqrt(np.mean(P[40:, 0, 0])))
        python, cxx = np.array(errors[True]), np.array(errors[False])
        spread = np.sqrt((python.var() + cxx.var()) / len(python))
        self.assertLess(abs(python.mean() - cxx.mean()), 4 * spread,
                        f"RMS error: Python {python.mean():.4f}, C++ {cxx.mean():.4f}")
        python, cxx = np.mean(sigmas[True]), np.mean(sigmas[False])
        self.assertLess(abs(python - cxx) / python, 0.1, f"reported sigma: Python {python:.4f}, C++ {cxx:.4f}")

    def test_a_seeded_run_repeats(self):
        from modules.framework.filters import EnsembleKalmanFilter
        _, z = _pendulum_run(0)
        first = np.asarray(EnsembleKalmanFilter(Pendulum(), n_members=40, seed=5).forward(z).xk).copy()
        again = np.asarray(EnsembleKalmanFilter(Pendulum(), n_members=40, seed=5).forward(z).xk)
        np.testing.assert_array_equal(first, again)


if __name__ == "__main__":
    unittest.main()

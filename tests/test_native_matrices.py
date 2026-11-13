"""Linear models run wholly in C++ whatever their matrices are.

A model of constant matrices has gone through the C++ kernel for long. One
whose matrices are functions of time is tabulated over the samples of the run
and filtered on a C++ model of tables (``KalmanFilter._forward_matrices``); a
model that is itself C++ (a ``NativeModel``) hands the whole pass over.

Skipped when libKalman is not built.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

import modules.framework.models.native as native
from modules.framework.native import core
from modules.framework.filter import BayesFilter, KalmanFilter
from modules.framework.model import KalmanModel
from modules.framework.constants import KalmanStructureKind

needs_core = unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
PATHS = ("_forward_twin", "_forward_matrices", "_forward_native", "_forward_callback")


class path_taken:
    """Which of the filter's C++ passes ran inside the block (``None``: the Python loop)."""

    def __enter__(self):
        self.taken, self._orig = [], {}
        for name in PATHS:
            self._orig[name] = getattr(BayesFilter, name)

            def wrap(name=name):
                def run(filt, *args, **kwargs):
                    self.taken.append(name)
                    return self._orig[name](filt, *args, **kwargs)
                return run
            setattr(BayesFilter, name, wrap())
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(BayesFilter, name, fn)

    @property
    def last(self):
        return self.taken[-1] if self.taken else None


def apart(a, b):
    xa, xb = np.asarray(a.xk, float), np.asarray(b.xk, float)
    Pa, Pb = np.asarray(a.Pk, float), np.asarray(b.Pk, float)
    return (float(np.nanmax(np.abs(xa - xb)) / max(1.0, np.nanmax(np.abs(xa)))),
            float(np.nanmax(np.abs(Pa - Pb)) / max(1.0, np.nanmax(np.abs(Pa)))))


def both(make, z):
    with native.python_reference():
        reference = make().filter(z)
    with path_taken() as path:
        record = make().filter(z)
    return reference, record, path.last


DT = 0.1
KW = dict(fs=10, H=np.array([[1.0, 0.0]]), x0=np.zeros((2, 1)), P0=np.eye(2))
F0 = np.array([[1.0, DT], [0.0, 1.0]])
Q0 = np.diag([1e-4, 1e-3])
Z = np.cumsum(np.random.default_rng(1).normal(size=(200, 1)), axis=0)


@needs_core
class TestTimeVaryingMatrices(unittest.TestCase):

    def check(self, make, z=Z, tolerance=1e-12):
        reference, record, path = both(make, z)
        self.assertEqual(path, "_forward_matrices")
        state, covariance = apart(reference, record)
        self.assertLess(state, tolerance)
        self.assertLess(covariance, tolerance)

    def test_a_measurement_noise_that_grows(self):
        self.check(lambda: KalmanModel("s", F=F0, Q=Q0, R=lambda t: np.array([[0.04 * (1 + (t or 0))]]), **KW))

    def test_every_matrix_a_function_of_time(self):
        def make():
            return KalmanModel(
                "s", fs=10, x0=np.zeros((2, 1)), P0=np.eye(2),
                F=lambda t: np.array([[1.0, DT * (1 + 0.1 * np.sin(t or 0))], [0.0, 1.0]]),
                H=lambda t: np.array([[np.cos(0.3 * (t or 0)), 0.2]]),
                Q=lambda t: Q0 * (1 + 0.5 * (t or 0)), R=lambda t: np.array([[0.04 + 0.01 * (t or 0)]]))
        self.assertIs(make().structure, KalmanStructureKind.TIME_VARYING)
        self.check(make)

    def test_missing_measurements_are_skipped(self):
        z = Z.copy()
        z[::7] = np.nan
        self.check(lambda: KalmanModel("s", F=F0, Q=Q0, R=lambda t: np.array([[0.04 * (1 + (t or 0))]]), **KW), z)

    def test_a_function_that_counts_its_calls_sees_the_loop_sequence(self):
        """Not what a matrix function should do, but models did: the layer of a
        calorimeter was a count kept in R(). Each function is called once per
        sample, in the loop's order, so such a model still gets its sequence."""
        def make():
            state = {"k": 0}

            def R(t=None):
                state["k"] += 1
                return np.array([[0.04 * state["k"]]])
            return KalmanModel("s", F=lambda t: F0 * 1.0 if state["k"] % 2 else F0, Q=Q0, R=R, **KW)
        self.check(make)

    def test_power_grid(self):
        from modules.apps.resonators.powergrids.harmonics import PowerGridStateModel
        make = lambda: PowerGridStateModel("s")
        z = np.asarray(make().sample(1.0, seed=1).zk)
        self.check(make, z)

    def test_a_controller_or_another_engine_keeps_its_own_path(self):
        from modules.framework.filters import UnscentedKalmanFilter
        model = KalmanModel("s", F=F0, Q=Q0, R=lambda t: np.array([[0.04]]), **KW)
        with path_taken() as path:
            UnscentedKalmanFilter(model).forward(Z[:20])
        self.assertNotEqual(path.last, "_forward_matrices")


@needs_core
class TestCalorimeter(unittest.TestCase):
    """The layers of a calorimeter are the samples: its matrices are functions of
    the layer, read from t. Forward and smoother in C++, same reconstruction."""

    def reconstruct(self, cls, geometry, particle, seed, **entry):
        from modules.apps.calorimeter import CalorimeterSampler
        hits = CalorimeterSampler(geometry).sample(10.0, particle, seed=seed, **entry)
        with native.python_reference():
            reference = cls.from_geometry(0, hits, geometry).reconstruct()
        with path_taken() as path:
            result = cls.from_geometry(0, hits, geometry).reconstruct()
        self.assertEqual(path.last, "_forward_matrices")
        for key in ("x0", "y0", "tx", "ty", "E", "chi2_dof"):
            self.assertLess(abs(result[key] - reference[key]), 1e-10 * max(1.0, abs(reference[key])), key)
        np.testing.assert_allclose(result["P"], reference["P"], rtol=1e-9, atol=1e-18)
        return result

    def test_electromagnetic_and_hadronic(self):
        from modules.apps.calorimeter import (CaloLayer, CalorimeterGeometry, CalorimeterElectromagneticModel,
                                              CalorimeterHadronicModel)
        ecal = CalorimeterGeometry([CaloLayer("pwo", 23., is_active=True, cell_size_mm=22.)] * 10,
                                   z_start=0., aperture_mm=200.)
        layers = []
        for _ in range(40):
            layers += [CaloLayer("fe", 20., is_active=False),
                       CaloLayer("scint_tile", 3., is_active=True, cell_size_mm=53., sigma_sampling=0.50)]
        hcal = CalorimeterGeometry(layers, z_start=0., aperture_mm=600.)
        for seed in (42, 3):
            photon = self.reconstruct(CalorimeterElectromagneticModel, ecal, "gamma", seed,
                                      x0=20., y0=-15., tx=0.05, ty=0.03)
            self.assertLess(abs(photon["E"] - 10.0), 1.5)
        self.reconstruct(CalorimeterHadronicModel, hcal, "neutral", 7, x0=-10., y0=30., tx=-0.02, ty=0.04)

    def test_the_layer_is_read_from_the_time(self):
        from modules.apps.calorimeter import CalorimeterLinearModel
        z = np.arange(5) * 10.0
        model = CalorimeterLinearModel(0, z, [np.eye(2) * (k + 1) for k in range(5)])
        self.assertIs(model.structure, KalmanStructureKind.TIME_VARYING)
        self.assertEqual(model.R(t=3.0)[0, 0], model.R(t=3.0)[0, 0])       # asking twice gives the same
        self.assertGreater(model.R(t=3.0)[0, 0], model.R(t=1.0)[0, 0])


@needs_core
class TestNativeModelPass(unittest.TestCase):
    """A model that is C++ hands the whole pass to Kalman::Filter."""

    def setUp(self):
        try:
            core.load_plugin(core.resolve_plugin("KalmanPlugin_example"))
        except (FileNotFoundError, PermissionError):
            self.skipTest("the example plugin is not built (make examples)")

    def test_linear_and_extended(self):
        for name, engine in (("constant-velocity", "linear"), ("fading-sensor", "linear"), ("pendulum", "extended")):
            model = native.NativeModel(name)
            z = np.cumsum(np.random.default_rng(2).normal(size=(120, model.m)), axis=0) * 0.01
            with path_taken() as path:
                record = model.filter(z)
            self.assertEqual(path.last, "_forward_twin", name)
            whole = native.NativeModel(name).filter_native(z, engine=engine)
            np.testing.assert_array_equal(np.asarray(record.xk).reshape(120, -1), np.asarray(whole.xk).reshape(120, -1))

    def test_a_subclass_that_rewrites_the_model_is_not_handed_over(self):
        class Rewritten(native.NativeModel):
            def h(self, x, v=None, t=None):
                return super().h(x, v=v, t=t) * 2.0
        self.assertIsNone(Rewritten("pendulum").native_handle())
        self.assertIsNotNone(native.NativeModel("pendulum").native_handle())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless("ballistic" in native.apps_library_names(), "libKalmanApps is not built")
class TestTwinsOnTheSigmaPointEngines(unittest.TestCase):
    """A model with a C++ twin runs wholly in C++ on the unscented and particle
    engines too. The Python classes take their noise as an argument of f and h,
    so the points are drawn over the noise as well: the twin is told to do the same."""

    def cases(self):
        from modules.apps.ballistics import BallisticModel
        from modules.apps.vehicles import CarSpeedModel
        from modules.apps.gps import GPSModel
        from modules.apps.weather.storms import StormModel
        for make in (lambda: BallisticModel("s", fs=100.0), lambda: CarSpeedModel("s"), lambda: StormModel("s")):
            yield make, np.asarray(make().sample(200 / make().fs, seed=1).zk)
        yield (lambda: GPSModel("s")), GPSModel("s").simulate(120.0, seed=1)

    def test_unscented(self):
        from modules.framework.filters import UnscentedKalmanFilter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for make, z in self.cases():
                with native.python_reference():
                    reference = UnscentedKalmanFilter(make()).forward(z)
                with path_taken() as path:
                    record = UnscentedKalmanFilter(make()).forward(z)
                self.assertEqual(path.last, "_forward_twin", type(make()).__name__)
                state, covariance = apart(reference, record)
                self.assertLess(state, 1e-7, type(make()).__name__)
                # With alpha = 1e-3 the points sit 1e-3 sigma from the mean and their
                # weights are of order 1e6: the last digit of h comes back that much larger.
                self.assertLess(covariance, 1e-4, type(make()).__name__)

    def test_particle_runs_on_the_twin(self):
        from modules.framework.filters import ParticleFilter
        from modules.apps.vehicles import CarSpeedModel
        z = np.asarray(CarSpeedModel("s").sample(5.0, seed=1).zk)
        np.random.seed(3)
        with path_taken() as path:
            record = ParticleFilter(CarSpeedModel("s"), n_particles=400).forward(z)
        self.assertEqual(path.last, "_forward_twin")
        self.assertTrue(np.all(np.isfinite(np.asarray(record.xk, float))))

    def test_the_extended_pass_after_a_sigma_pass_is_unchanged(self):
        from modules.framework.filters import UnscentedKalmanFilter
        from modules.apps.vehicles import CarSpeedModel
        z = np.asarray(CarSpeedModel("s").sample(5.0, seed=1).zk)
        model = CarSpeedModel("s")
        before = np.asarray(model.filter(z).xk)
        UnscentedKalmanFilter(model).forward(z)
        np.testing.assert_array_equal(np.asarray(model.filter(z).xk), before)


class TestFilterFamilies(unittest.TestCase):
    """Gaussian beliefs are Kalman filters; a particle filter is not one."""

    def test_hierarchy(self):
        from modules.framework.filters import (ParticleFilter, LinearKalmanFilter, ExtendedKalmanFilter,
                                               UnscentedKalmanFilter, EnsembleKalmanFilter)
        for cls in (LinearKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, EnsembleKalmanFilter):
            self.assertTrue(issubclass(cls, KalmanFilter) and issubclass(cls, BayesFilter), cls.__name__)
        self.assertTrue(issubclass(ParticleFilter, BayesFilter))
        self.assertFalse(issubclass(ParticleFilter, KalmanFilter))


@unittest.skipUnless("ballistic" in native.apps_library_names(), "libKalmanApps is not built")
class TestSmootherInCxx(unittest.TestCase):
    """The RTS sweep runs in C++ after an extended pass too (one Jacobian per step)."""

    def test_after_a_twin_and_after_a_time_varying_model(self):
        from modules.apps.ballistics import BallisticModel
        from modules.apps.gravitational.gravwave.polarization import PolarizationModel
        cases = [(lambda: BallisticModel("s", fs=100.0), None),
                 (lambda: PolarizationModel("s"), None),
                 (lambda: KalmanModel("s", F=F0, Q=Q0, R=lambda t: np.array([[0.04 * (1 + (t or 0))]]), **KW), Z)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for make, z in cases:
                if z is None:
                    z = np.asarray(make().sample(300 / make().fs, seed=2).zk)
                with native.python_reference():
                    reference = make().filter(z, smooth=True)
                calls = []
                original = BayesFilter._backward_native
                BayesFilter._backward_native = lambda self, N: (calls.append(N), original(self, N))[1]
                try:
                    record = make().filter(z, smooth=True)
                finally:
                    BayesFilter._backward_native = original
                self.assertEqual(calls, [len(z)], type(make()).__name__)
                state, covariance = apart(reference, record)
                self.assertLess(state, 1e-6, type(make()).__name__)
                self.assertLess(covariance, 1e-5, type(make()).__name__)


@unittest.skipUnless("ballistic" in native.apps_library_names(), "libKalmanApps is not built")
class TestNoModelFallsBackToThePythonLoop(unittest.TestCase):
    """Every model of the catalogue that builds with its defaults filters in a C++
    pass. A model added later without one fails here instead of running slowly
    unnoticed."""

    def test_catalogue(self):
        import inspect
        from modules import api
        checked, python_loop = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, cls in sorted(api._model_catalog().items()):
                if hasattr(cls, "NATIVE_NAME"):
                    continue
                first = list(inspect.signature(cls.__init__).parameters)[1:2]
                try:
                    model = cls("s") if first in (["sensor_id"], ["name"]) else cls()
                    z = np.asarray(model.sample(20 / (model.fs or 1.0), seed=1).zk, float)
                except Exception:
                    continue                    # needs arguments, or its default state cannot be sampled
                if not np.all(np.isfinite(z)):
                    continue
                with path_taken() as path:
                    model.filter(z)
                checked.append(name)
                if path.last is None and not model.engine.OWNS_PASS:
                    python_loop.append(name)
        self.assertEqual(python_loop, [])
        self.assertGreaterEqual(len(checked), 14, checked)

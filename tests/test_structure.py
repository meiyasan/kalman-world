"""Structural invariants of the framework/apps layout.

These guard the architecture itself rather than any single model:

- every abstract contract + base + registry lives in ``framework/``;
- every app domain is a uniform *package* (variant submodules + an ``__init__``
  that re-exports its public names and a ``VARIANTS`` registry);
- the plug-in registries (engine / integrator / solver / provider) cover their
  full kind enum and resolve by name;
- the public ``kalman.apps`` surface stays importable after reorganisation;
- ``apps/`` holds packages only (no flat domain modules).
"""
import importlib
import inspect
import os
import pkgutil
import sys
import unittest

sys.path.insert(0, ".")
import modules.apps as apps
from modules.framework.model import KalmanModel
from modules.framework.constants import (ALL_ENGINES, ALL_INTEGRATORS,
                                         ALL_SOLVERS, ALL_PROVIDERS)

APPS_DIR = os.path.dirname(apps.__file__)

# apps/ contains packages only — no flat domain modules, no shims.
SHIMS: set[str] = set()


# Family packages group related domains; their members are the real domains.
FAMILIES = {"astro", "gravitational", "resonators"}


def _domain_packages():
    """(import_path, name) for every leaf domain package, families expanded."""
    out = []
    for m in pkgutil.iter_modules([APPS_DIR]):
        if not m.ispkg:
            continue
        if m.name in FAMILIES:
            fam_dir = os.path.join(APPS_DIR, m.name)
            for sub in pkgutil.iter_modules([fam_dir]):
                if sub.ispkg:
                    out.append((f"{m.name}.{sub.name}", sub.name))
        else:
            out.append((m.name, m.name))
    return sorted(out)


class TestFrameworkOwnsContracts(unittest.TestCase):
    """Bases live in framework/; concretes live outside it."""

    def test_bases_are_in_framework(self):
        from modules.framework import (KalmanModel as M, KalmanEngine, KalmanIntegrator,
                                       KalmanSolver, KalmanProvider, KalmanFilter)
        for base in (M, KalmanEngine, KalmanIntegrator, KalmanSolver, KalmanProvider, KalmanFilter):
            self.assertTrue(base.__module__.startswith("modules.framework"),
                            f"{base.__name__} base must live in framework/, got {base.__module__}")

    def test_estimator_alias(self):
        from modules.framework import KalmanEngine, KalmanEstimator
        self.assertIs(KalmanEstimator, KalmanEngine)


class TestReservedNames(unittest.TestCase):
    """The names of the model contract mean one thing each.

    An operation gives back a model made from this one; an engine hook is a
    method an engine looks up on the model. A name used for both, or redefined by
    an application with another meaning, is found by the wrong caller without any
    error (an ``observe`` operation once hijacked the ensemble engine's hook).
    """

    #: Methods of KalmanModel that return a model made from this one.
    OPERATIONS = ("joint", "add_sensor", "add_instrument", "lag")
    #: Reserved for the operations still to come: nothing may take these names.
    PLANNED = ("compose", "route")
    #: Aliases of f and h, provided by the base class; a model never defines them.
    ALIASES = ("step", "sense")
    #: Looked up on a model by the ensemble engine only.
    ENSEMBLE_HOOKS = ("covariance_mask",)
    #: Ensemble-filter words that are not model methods: the engine's own steps.
    NOT_ON_A_MODEL = ("forecast", "observe", "project", "analyze", "initial_ensemble", "localization")

    @staticmethod
    def _models():
        for _, name, _ in pkgutil.walk_packages(apps.__path__, apps.__name__ + "."):
            try:
                importlib.import_module(name)
            except Exception:
                continue
        seen, stack = [], [KalmanModel]
        while stack:
            for sub in stack.pop().__subclasses__():
                if sub not in seen:
                    seen.append(sub)
                    stack.append(sub)
        return seen

    def test_operations_are_methods_of_the_base_contract(self):
        for name in self.OPERATIONS:
            self.assertTrue(callable(getattr(KalmanModel, name, None)), name)

    def test_f_and_h_take_one_state_or_many(self):
        import numpy as np
        model = KalmanModel("cv", fs=10.0, F=np.array([[1, 0.1], [0, 1]]), H=np.array([[1.0, 0.0]]),
                            Q=1e-3 * np.eye(2), R=np.array([[0.1]]), x0=np.zeros((2, 1)), P0=np.eye(2))
        X = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, -1.0]])
        x = X[:, 1:2]
        np.testing.assert_array_equal(model.step(x), model.f(x))                 # step is f
        np.testing.assert_array_equal(model.sense(x), model.h(x))                # sense is h
        np.testing.assert_array_equal(model.step(X), model.F() @ X)              # many: each column
        np.testing.assert_array_equal(model.sense(X), model.H() @ X)
        self.assertEqual(KalmanModel.BATCHED, ())
        self.assertFalse(model.batched("f") or model.batched("h"))

    def test_a_model_not_written_for_many_states_is_looped_over(self):
        import numpy as np

        class ByPosition(KalmanModel):
            def __init__(self):
                self.n, self.m, self.l = 2, 1, 0
                super().__init__("p", fs=1.0, Q=np.zeros((2, 2)), R=np.eye(1), x0=np.zeros((2, 1)), P0=np.eye(2))

            def f(self, x, u=None, w=None, t=None):
                x = np.asarray(x, dtype=float).reshape(-1)       # picks entries by position
                return np.array([[x[0] + x[1]], [x[1]]])

            def h(self, x, v=None, t=None):
                return np.asarray(x, dtype=float).reshape(-1)[:1].reshape(1, 1)

        model = ByPosition()
        X = np.array([[0.0, 1.0, 2.0], [1.0, 1.0, -1.0]])
        np.testing.assert_array_equal(model.step(X), [[1.0, 2.0, 1.0], [1.0, 1.0, -1.0]])
        np.testing.assert_array_equal(model.sense(X), [[0.0, 1.0, 2.0]])

    def test_a_model_never_defines_the_aliases(self):
        for alias in self.ALIASES:
            with self.assertRaisesRegex(TypeError, "is the name of"):
                type("Wrong", (KalmanModel,), {alias: lambda self, x: x})

    def test_one_initial_state_in_two_forms(self):
        import numpy as np
        model = KalmanModel("cv", fs=10.0, F=np.array([[1, 0.1], [0, 1]]), H=np.array([[1.0, 0.0]]),
                            Q=1e-3 * np.eye(2), R=np.array([[0.1]]), x0=np.array([[1.0], [-2.0]]),
                            P0=np.diag([4.0, 0.25]))
        x0, P0 = model.initial_state()                                   # read
        np.testing.assert_array_equal(x0.ravel(), [1.0, -2.0])
        model.initial_state([0.0, 1.0])                                  # set the mean only
        np.testing.assert_array_equal(model.x0.ravel(), [0.0, 1.0])
        np.testing.assert_array_equal(model.P0, np.diag([4.0, 0.25]))
        X = model.initial_state(samples=4000, rng=3)                     # as samples: columns
        self.assertEqual(X.shape, (2, 4000))
        np.testing.assert_allclose(X.mean(axis=1), [0.0, 1.0], atol=0.1)
        np.testing.assert_allclose(X.std(axis=1), [2.0, 0.5], rtol=0.05)
        np.testing.assert_array_equal(model.initial_state(samples=5, rng=3), model.initial_state(samples=5, rng=3))
        self.assertEqual(model.initial_state(samples=7).shape, (2, 7))   # numpy's global stream
        for gone in ("initial_state_vector", "initial_state_covariance", "get_initial_state",
                     "get_initial_state_vector", "get_initial_state_covariance", "initial_ensemble"):
            self.assertFalse(hasattr(model, gone), gone)
        from modules.framework.filter import KalmanFilter
        kf = KalmanFilter(model)
        np.testing.assert_array_equal(kf.initial_state()[0], model.x0)        # the filter has the same one
        self.assertEqual(kf.initial_state(samples=3, rng=1).shape, (2, 3))
        for gone in ("get_initial_state", "get_initial_state_vector", "get_initial_state_covariance"):
            self.assertFalse(hasattr(kf, gone), gone)

    def test_what_only_one_engine_needs_is_a_class_beside_the_model(self):
        import numpy as np
        from modules.framework.engine_models import KalmanEnsembleModel, KalmanDetectorModel
        from modules.apps.weather.flow import KolmogorovFlowModel
        from modules.apps.tracking import DriftChamberModel
        from modules.framework.filters import CombinatorialKalmanFilter
        self.assertTrue(issubclass(KolmogorovFlowModel, KalmanEnsembleModel))
        self.assertTrue(issubclass(DriftChamberModel, KalmanDetectorModel))
        self.assertIsNone(KalmanEnsembleModel().covariance_mask())           # no mask by default
        for name in ("covariance_mask", "seed_state", "radii"):
            self.assertFalse(hasattr(KalmanModel, name), f"{name} is not part of every model")
        plain = KalmanModel("cv", fs=10.0, F=np.eye(2), H=np.array([[1.0, 0.0]]), Q=1e-3 * np.eye(2),
                            R=np.array([[0.1]]), x0=np.zeros((2, 1)), P0=np.eye(2))
        with self.assertRaisesRegex(TypeError, "not a KalmanDetectorModel"):
            CombinatorialKalmanFilter(plain).forward([np.zeros((1, 1))], layer_radii=[1.0])

    def test_ensemble_words_are_not_model_methods(self):
        for cls in [KalmanModel] + self._models():
            for name in self.NOT_ON_A_MODEL:
                self.assertFalse(callable(cls.__dict__.get(name)), f"{cls.__name__}.{name}: an engine's word on a model")

    def test_the_flow_model_declares_its_maps(self):
        from modules.apps.weather.flow import KolmogorovFlowModel
        self.assertEqual(KolmogorovFlowModel.BATCHED, ("f", "h"))

    def test_no_operation_is_named_like_a_hook(self):
        self.assertFalse(set(self.OPERATIONS + self.PLANNED) & set(self.ENSEMBLE_HOOKS + self.NOT_ON_A_MODEL + self.ALIASES))

    def test_applications_do_not_redefine_them(self):
        for cls in self._models():
            if not cls.__module__.startswith("modules.apps"):
                continue
            for name in self.OPERATIONS + self.PLANNED:
                self.assertNotIn(name, cls.__dict__, f"{cls.__name__}.{name} shadows a model operation")

    def test_planned_names_are_free(self):
        for cls in [KalmanModel] + self._models():
            for name in self.PLANNED:
                self.assertFalse(hasattr(cls, name), f"{cls.__name__}.{name} is taken before its operation exists")


class TestRegistriesCoverTheirKinds(unittest.TestCase):
    """Each plug-in registry maps every member of its kind enum."""

    def test_registry_coverage(self):
        from modules.framework.engines import ENGINE_BY_KIND
        from modules.framework.integrators import INTEGRATOR_BY_KIND
        from modules.framework.solvers import SOLVER_BY_KIND
        from modules.providers import PROVIDER_BY_KIND
        for registry, kinds, label in (
                (ENGINE_BY_KIND, ALL_ENGINES, "engine"),
                (INTEGRATOR_BY_KIND, ALL_INTEGRATORS, "integrator"),
                (SOLVER_BY_KIND, ALL_SOLVERS, "solver"),
                (PROVIDER_BY_KIND, ALL_PROVIDERS, "provider")):
            self.assertEqual(set(registry), set(kinds), f"{label} registry misses kinds")

    def test_concretes_tag_their_kind(self):
        from modules.framework.integrators import INTEGRATOR_BY_KIND
        from modules.framework.solvers import SOLVER_BY_KIND
        for registry in (INTEGRATOR_BY_KIND, SOLVER_BY_KIND):
            for kind, cls in registry.items():
                self.assertEqual(cls.KIND, kind, f"{cls.__name__}.KIND != {kind}")


class TestAppsAreUniformPackages(unittest.TestCase):
    """Every domain is a package with variant submodules and a public surface."""

    def test_no_unexpected_flat_modules(self):
        flat = {m.name for m in pkgutil.iter_modules([APPS_DIR]) if not m.ispkg}
        self.assertEqual(flat - SHIMS, set(),
                         "app domains must be packages (flat modules left: "
                         f"{sorted(flat - SHIMS)})")

    def test_every_domain_has_variants_and_all(self):
        for path, name in _domain_packages():
            mod = importlib.import_module(f"modules.apps.{path}")
            self.assertTrue(hasattr(mod, "__all__"), f"{name} package needs __all__")
            self.assertTrue(hasattr(mod, "VARIANTS"), f"{name} package needs VARIANTS")
            self.assertGreater(len(mod.VARIANTS), 0, f"{name}.VARIANTS is empty")
            for key, cls in mod.VARIANTS.items():
                self.assertTrue(inspect.isclass(cls),
                                f"{name}.VARIANTS[{key!r}] must be a class")

    def test_variants_are_kalman_models(self):
        """VARIANTS entries are models (the estimable object of that domain)."""
        for path, name in _domain_packages():
            mod = importlib.import_module(f"modules.apps.{path}")
            for key, cls in mod.VARIANTS.items():
                self.assertTrue(issubclass(cls, KalmanModel),
                                f"{name}.VARIANTS[{key!r}] = {cls.__name__} is not a KalmanModel")

    def test_public_names_resolve_from_apps(self):
        """Every name a domain exports is reachable and identical via kalman.apps
        when re-exported there (the stable public surface)."""
        for path, name in _domain_packages():
            mod = importlib.import_module(f"modules.apps.{path}")
            for public in mod.__all__:
                self.assertTrue(hasattr(mod, public), f"{name}.{public} missing")
                if hasattr(apps, public):
                    self.assertIs(getattr(apps, public), getattr(mod, public),
                                  f"kalman.apps.{public} differs from {name}.{public}")

    def test_families_expose_their_domains(self):
        """A family package groups domains and exposes them via DOMAINS."""
        for fam in FAMILIES:
            mod = importlib.import_module(f"modules.apps.{fam}")
            self.assertTrue(hasattr(mod, "DOMAINS"), f"{fam} family needs DOMAINS")
            self.assertGreater(len(mod.DOMAINS), 1, f"{fam} should group >1 domain")
            for name, dom in mod.DOMAINS.items():
                self.assertTrue(hasattr(dom, "VARIANTS"), f"{fam}.{name} needs VARIANTS")


if __name__ == "__main__":
    unittest.main()

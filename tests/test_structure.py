"""Structural invariants of the framework/apps layout.

These guard the architecture itself rather than any single model:

- every abstract contract + base + registry lives in ``framework/``;
- every app domain is a uniform *package* (variant submodules + an ``__init__``
  that re-exports its public names and a ``VARIANTS`` registry);
- the plug-in registries (engine / integrator / solver / provider) cover their
  full kind enum and resolve by name;
- the public ``kalman.apps`` surface stays importable after reorganisation.
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

# Flat modules deliberately kept as back-compat shims for the weather move.
SHIMS = {"fluids", "reanalysis", "storms", "flowsurrogate"}


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

    def test_shims_forward_to_weather(self):
        """The back-compat shims still resolve to the weather package objects."""
        from modules.apps import weather
        for shim, canonical in (("fluids", "flow"), ("reanalysis", "geo"),
                                ("storms", "storms"), ("flowsurrogate", "surrogate")):
            m = importlib.import_module(f"modules.apps.{shim}")
            target = importlib.import_module(f"modules.apps.weather.{canonical}")
            shared = [n for n in dir(target)
                      if not n.startswith("_") and hasattr(m, n)
                      and (inspect.isclass(getattr(target, n)) or inspect.isfunction(getattr(target, n)))]
            self.assertTrue(shared, f"shim {shim} forwards nothing")
            for n in shared[:5]:
                self.assertIs(getattr(m, n), getattr(target, n), f"{shim}.{n} is not the canonical object")


if __name__ == "__main__":
    unittest.main()

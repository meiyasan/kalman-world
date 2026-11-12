"""The field solvers forward to C++ (Kalman::Solvers::Vorticity, ::FiniteDifference)."""
import unittest

import numpy as np

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.constants import KalmanIntegratorKind
from modules.framework.solvers.finite_difference import FiniteDifferenceSolver
from modules.framework.solvers.spectral import SpectralSolver
from modules.apps.weather.flow import NS2D
from modules.apps.weather.barotropic import BetaPlaneNS2D


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
class TestSolversInCxx(unittest.TestCase):

    def both(self, solver, w, steps):
        got = solver.advance(w.copy(), steps)
        with native.python_reference():
            want = solver.advance(w.copy(), steps)
        return got, want

    def test_kolmogorov_on_every_kind_of_grid_and_integrator(self):
        for n in (16, (16, 24), 20, (12, 14)):                 # 2^k, {2,3,5}-smooth, and neither
            for kind in (KalmanIntegratorKind.EULER, KalmanIntegratorKind.RK4, KalmanIntegratorKind.DORMAND_PRINCE):
                s = NS2D(n=n, nu=1 / 40., kf=2, chi=2.0, dt=0.005, seed=3, integrator=kind)
                self.assertIsNotNone(s._native_scheme(), f"{n} {kind}")
                w = s.random_ic()
                got, want = self.both(s, w, 6)
                self.assertEqual(got.shape, want.shape)
                np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, err_msg=f"{n} {kind}")
                flat, _ = self.both(s, w.ravel(), 6)
                np.testing.assert_allclose(flat, want.ravel(), rtol=0, atol=1e-12)
                self.assertGreater(np.abs(want - w).max(), 1e-4)     # the field did move

    def test_beta_plane(self):
        s = BetaPlaneNS2D(n=(16, 24), seed=1)
        w = s.random_ic(amp=1e-5)
        got, want = self.both(s, w, 20)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-17)

    def test_diffusion_with_each_boundary(self):
        for bc in ("periodic", "dirichlet", "neumann"):
            s = FiniteDifferenceSolver(n=(20, 32), dt=1e-3, bc=bc, bc_value=0.3, nu=0.05)
            self.assertIsNotNone(s._native_scheme())
            w = s.random_ic()
            got, want = self.both(s, w, 40)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-13, err_msg=bc)
            flat, _ = self.both(s, w.ravel(), 40)
            np.testing.assert_allclose(flat, want.ravel(), rtol=0, atol=1e-13)

    def test_other_physics_stays_in_python(self):
        class Damped(NS2D):                        # changes the equation, keeps the declared terms
            def rhs_hat(self, wh, t=None):
                return super().rhs_hat(wh, t) - 0.5 * wh
        s = Damped(n=16, dt=0.005, seed=3)
        self.assertIsNone(s._native_scheme())
        w = s.random_ic()
        plain = NS2D(n=16, dt=0.005, seed=3).advance(w.copy(), 5)
        self.assertGreater(np.abs(s.advance(w.copy(), 5) - plain).max(), 1e-3)   # its own rhs_hat ran
        with self.assertRaises(NotImplementedError):                               # and libTorch is refused too
            Damped(n=16, dt=0.005, backend="torch", device="cpu").advance(w.copy(), 1)

        class Reaction(FiniteDifferenceSolver):
            def rhs(self, field, t=None):
                return self.nu * self.laplacian(field) - field
        r = Reaction(n=16, dt=1e-3)
        self.assertIsNone(r._native_scheme())
        self.assertTrue(np.all(np.isfinite(r.advance(r.random_ic(), 5))))

        self.assertIsNone(SpectralSolver(n=8)._native_scheme())                    # no physics stated at all


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
class TestEnsembleAnalysisInCxx(unittest.TestCase):
    """A field model's ensemble filter keeps its hooks in Python; the analysis arithmetic is C++."""

    def run_cycles(self, reference, obs_type="vorticity"):
        from modules.apps.weather.flow import KolmogorovFlowModel
        from modules.framework.engines.ensemble import EnsembleKalmanEngine
        model = KolmogorovFlowModel("flow", grid=16, cycle=4, n_obs=24, obs_type=obs_type, seed=2)
        engine = EnsembleKalmanEngine(n_members=12, inflation=1.05, seed=5)
        rng = np.random.default_rng(9)
        # A small ensemble drawn around a spun-up state (the model's own
        # initial_ensemble would spin a 2500-step climatology first).
        truth = model.solver.advance(model.solver.random_ic(), 200).ravel()
        start = truth[:, None] + 0.3 * np.random.default_rng(1).standard_normal((model.n, 12))
        model.initial_ensemble = lambda N, rng=None: start.copy()
        def go():
            engine.reset(model)
            x = truth.copy()
            means = []
            for k in range(4):
                x = np.asarray(model.forecast(x.reshape(-1, 1)), float).ravel()
                z = np.asarray(model.h(x.reshape(-1, 1)), float).ravel() + 0.05 * rng.standard_normal(model.m)
                if k == 2:
                    z[::3] = np.nan                       # some probes silent
                engine.forecast(model)
                means.append(engine.analyze(model, z).copy())
            return np.array(means), engine.members.copy()
        if reference:
            with native.python_reference():
                return go()
        return go()

    def test_analysis_matches_the_python_engine(self):
        for obs_type in ("vorticity", "gradient"):
            self._compare(obs_type)

    def _compare(self, obs_type):
        (m_cxx, X_cxx), (m_py, X_py) = self.run_cycles(False, obs_type), self.run_cycles(True, obs_type)
        self.assertTrue(np.all(np.isfinite(X_cxx)))
        scale = np.abs(X_py).max()
        self.assertLess(np.abs(X_cxx - X_py).max() / scale, 1e-9)
        self.assertLess(np.abs(m_cxx - m_py).max() / scale, 1e-9)

    def test_the_kernel_refuses_a_single_member(self):
        out = np.empty((3, 1))
        a = lambda *shape: np.ascontiguousarray(np.ones(shape))
        rc = core.lib.enkf_analysis(3, 1, 2, core.cdptr(a(3, 1)), core.cdptr(a(2, 1)), core.cdptr(np.ascontiguousarray(np.eye(2))),
                                    core.cdptr(a(2)), core.cdptr(a(2, 1)), core.ffi.NULL, core.ffi.NULL, core.dptr(out))
        self.assertEqual(rc, -1)
        self.assertTrue(np.all(np.isnan(out)))

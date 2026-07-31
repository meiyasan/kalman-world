"""Tests for the KalmanIntegrator (time-stepping) and KalmanSolver (spatial
discretization) concepts, the model f_continuous delegation, and the torch/GPU
SpectralSolver backend."""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.framework import (KalmanModel, KalmanEngine, KalmanEstimator,
                               KalmanIntegrator, KalmanSolver, SpectralSolver,
                               EulerIntegrator, RK4Integrator, DormandPrinceIntegrator)
from modules.framework.solvers import FiniteDifferenceSolver
from modules.framework.integrators import (INTEGRATOR_BY_KIND, resolve_integrator,
                                           register_integrator, unregister_integrator,
                                           available_integrators)
from modules.framework.solvers import (SOLVER_BY_KIND, resolve_solver, register_solver,
                                       unregister_solver, available_solvers)
from modules.framework.engines import resolve_engine
from modules.framework.constants import (KalmanIntegratorKind, ALL_INTEGRATORS,
                                         KalmanSolverKind, ALL_SOLVERS)


# ── Integrators ───────────────────────────────────────────────────────────────

class TestIntegrators(unittest.TestCase):

    def test_decay_accuracy_ordering(self):
        """xdot=-x → x(dt)=x0 e^-dt; RK4 & DP far more accurate than Euler."""
        f = lambda x, u, t: -x
        x0 = np.array([1.0]); dt = 0.1
        exact = np.exp(-dt)
        e = abs(EulerIntegrator().step(f, x0, dt=dt)[0] - exact)
        r = abs(RK4Integrator().step(f, x0, dt=dt)[0] - exact)
        d = abs(DormandPrinceIntegrator().step(f, x0, dt=dt)[0] - exact)
        self.assertLess(r, e / 100)
        self.assertLess(d, r + 1e-12)
        for I in (EulerIntegrator, RK4Integrator, DormandPrinceIntegrator):
            self.assertLess(abs(I().step(f, x0, dt=1e-4)[0] - np.exp(-1e-4)), 1e-3)

    def test_rk4_conserves_shm_energy(self):
        """ẍ=-x (SHM): RK4 conserves energy far better than Euler over many steps."""
        f = lambda s, u, t: np.array([s[1], -s[0]])
        def energy(I, n=2000, dt=0.05):
            s = np.array([1.0, 0.0])
            for _ in range(n):
                s = I.step(f, s, dt=dt)
            return 0.5 * (s[0] ** 2 + s[1] ** 2)
        self.assertLess(abs(energy(RK4Integrator()) - 0.5), 1e-3)
        self.assertGreater(abs(energy(EulerIntegrator()) - 0.5), 0.1)

    def test_integrate_n_steps(self):
        f = lambda x, u, t: -x
        one = RK4Integrator()
        x = one.integrate(f, np.array([1.0]), dt=0.01, n=100)[0]
        self.assertLess(abs(x - np.exp(-1.0)), 1e-6)

    def test_registry_and_resolve(self):
        self.assertEqual(set(INTEGRATOR_BY_KIND), ALL_INTEGRATORS)
        self.assertIsInstance(resolve_integrator(KalmanIntegratorKind.RK4), RK4Integrator)
        self.assertIsInstance(resolve_integrator(None), RK4Integrator)   # default
        with self.assertRaises(NotImplementedError):
            resolve_integrator(KalmanIntegratorKind.IMPLICIT_EULER).step(lambda x, u, t: x, np.zeros(1))

    def test_dtype_preserving(self):
        z = np.array([1.0 + 1j])                          # complex (spectral use)
        out = RK4Integrator().step(lambda x, u, t: -x, z, dt=0.1)
        self.assertTrue(np.iscomplexobj(out))


# ── Solvers ───────────────────────────────────────────────────────────────────

class TestSolvers(unittest.TestCase):

    def test_registry_and_reserved(self):
        self.assertEqual(set(SOLVER_BY_KIND), ALL_SOLVERS)
        self.assertIsInstance(resolve_solver(KalmanSolverKind.SPECTRAL), SpectralSolver)
        self.assertIsInstance(resolve_solver(KalmanSolverKind.FINITE_DIFFERENCE),
                              FiniteDifferenceSolver)
        for k in (KalmanSolverKind.FINITE_VOLUME, KalmanSolverKind.FINITE_ELEMENT):
            with self.assertRaises(NotImplementedError):
                resolve_solver(k).solve(np.zeros((4, 4)))

    def test_spectral_rectangular_grid(self):
        """A solver resolves a given 2-D gridding: square or rectangular, with
        physical domain lengths; a mismatched field is rejected clearly."""
        from modules.apps.weather.flow import NS2D
        r = NS2D(n=(16, 32), nu=1/40., kf=4, chi=2.0, dt=0.01, seed=1)
        self.assertEqual(r.shape, (16, 32))
        self.assertIsNone(r.n)                              # not square
        w = r.random_ic()
        self.assertEqual(w.shape, (16, 32))
        self.assertEqual(r.advance(w.copy(), 3).shape, (16, 32))
        gx, gy = r.gradient(w)
        self.assertEqual(gx.shape, (16, 32))
        ens = np.stack([r.random_ic() for _ in range(3)])
        self.assertEqual(r.advance(ens.copy(), 2).shape, (3, 16, 32))
        sq = NS2D(n=16, dt=0.01, seed=1)
        with self.assertRaises(ValueError):
            sq.advance(np.zeros((8, 8)), 1)                  # grid mismatch

    def test_spectral_physical_length_scaling(self):
        """Wavenumbers scale as 2π/L, so gradients are right on any domain size."""
        n, Lx = 32, 4.0
        s = SpectralSolver(n=(n, n), length=(2 * np.pi, Lx))
        x = Lx * np.arange(n) / n
        X = np.tile(x, (n, 1))
        gx, _ = s.gradient(np.sin(2 * np.pi * X / Lx))
        np.testing.assert_allclose(gx, (2 * np.pi / Lx) * np.cos(2 * np.pi * X / Lx), atol=1e-10)


class TestFiniteDifferenceSolver(unittest.TestCase):
    """The first non-spectral solver — arbitrary/bounded grids via stencils."""

    def test_second_order_operators(self):
        n, L = 64, 2 * np.pi
        s = FiniteDifferenceSolver(n=n, length=L, bc="periodic")
        x = L * np.arange(n) / n
        X, Y = np.meshgrid(x, x)
        f = np.sin(X) * np.cos(Y)
        fx, fy = s.gradient(f)
        np.testing.assert_allclose(fx, np.cos(X) * np.cos(Y), atol=5e-3)
        np.testing.assert_allclose(fy, -np.sin(X) * np.sin(Y), atol=5e-3)
        np.testing.assert_allclose(s.laplacian(f), -2 * f, atol=5e-3)

    def test_diffusion_matches_analytic_decay(self):
        n, L, nu = 64, 2 * np.pi, 0.1
        s = FiniteDifferenceSolver(n=n, length=L, dt=1e-4, nu=nu, bc="periodic")
        x = L * np.arange(n) / n
        X, Y = np.meshgrid(x, x)
        f = np.sin(X) * np.cos(Y)                            # eigenfunction: decays e^{-2νt}
        t_end = 0.5
        out = s.advance(f.copy(), int(t_end / s.dt))
        np.testing.assert_allclose(out, np.exp(-2 * nu * t_end) * f, atol=1e-3)

    def test_boundary_conditions(self):
        f = np.ones((16, 16))
        per = FiniteDifferenceSolver(n=16, bc="periodic", dt=1e-3, nu=0.5).advance(f.copy(), 20)
        neu = FiniteDifferenceSolver(n=16, bc="neumann", dt=1e-3, nu=0.5).advance(f.copy(), 20)
        dir_ = FiniteDifferenceSolver(n=16, bc="dirichlet", bc_value=0.0,
                                      dt=1e-3, nu=0.5).advance(f.copy(), 20)
        np.testing.assert_allclose(per, f, atol=1e-9)        # uniform field is stationary
        np.testing.assert_allclose(neu, f, atol=1e-9)
        self.assertLess(dir_[0, 0], 0.99)                    # boundary pulled toward 0
        self.assertAlmostEqual(dir_[8, 8], 1.0, places=6)    # interior still untouched
        with self.assertRaises(ValueError):
            FiniteDifferenceSolver(n=8, bc="nope")

    def test_rectangular_and_integrator_swap(self):
        r = FiniteDifferenceSolver(n=(8, 16), dt=1e-3)
        self.assertEqual(r.shape, (8, 16))
        self.assertEqual(r.advance(r.random_ic(), 3).shape, (8, 16))
        e = FiniteDifferenceSolver(n=16, dt=1e-3, integrator=KalmanIntegratorKind.EULER)
        self.assertIsInstance(e.integrator, EulerIntegrator)
        self.assertEqual(e.solve(np.zeros((16, 16)), 2).shape, (16, 16))

    def test_spectral_composes_integrator(self):
        from modules.apps.weather.flow import NS2D
        s = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=0.01, seed=1)
        self.assertIsInstance(s, SpectralSolver)
        self.assertIsInstance(s.integrator, RK4Integrator)
        w = s.random_ic()
        np.testing.assert_allclose(s.solve(w.copy(), 3), s.advance(w.copy(), 3))

    def test_torch_backend_matches_numpy(self):
        try:
            import torch  # noqa
        except ImportError:
            self.skipTest("torch not installed")
        from modules.apps.weather.flow import NS2D
        nps = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=0.01, seed=2)
        ts = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=0.01, seed=2, backend="torch", device="cpu")
        w = nps.random_ic()
        np.testing.assert_allclose(nps.advance(w.copy(), 4), ts.advance(w.copy(), 4), atol=1e-9)
        # batched (N,g,g) ensemble advance equals per-member
        ens = np.stack([nps.random_ic() for _ in range(3)])
        per = np.stack([nps.advance(ens[i].copy(), 4) for i in range(3)])
        np.testing.assert_allclose(per, ts.advance(ens.copy(), 4), atol=1e-9)

    def test_spectral_integrator_is_swappable(self):
        """Any explicit integrator drives the spectral solver — uniformly on
        numpy and torch (the composed integrator, via its Butcher tableau)."""
        from modules.apps.weather.flow import NS2D
        w0 = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=0.01, seed=1).random_ic()
        fine = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=5e-4, seed=1)
        truth = fine.advance(w0.copy(), 20)                # 0.01 total
        errs = {}
        for kind in (KalmanIntegratorKind.EULER, KalmanIntegratorKind.RK4,
                     KalmanIntegratorKind.DORMAND_PRINCE):
            s = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=0.01, seed=1, integrator=kind)
            self.assertEqual(s.integrator.KIND, kind)
            errs[kind] = np.abs(s.advance(w0.copy(), 1) - truth).max()
        self.assertLess(errs[KalmanIntegratorKind.RK4], errs[KalmanIntegratorKind.EULER])
        self.assertLess(errs[KalmanIntegratorKind.DORMAND_PRINCE], errs[KalmanIntegratorKind.EULER])
        try:
            import torch  # noqa
        except ImportError:
            return
        for kind in (KalmanIntegratorKind.EULER, KalmanIntegratorKind.DORMAND_PRINCE):
            a = NS2D(n=16, dt=0.01, seed=1, integrator=kind).advance(w0.copy(), 4)
            b = NS2D(n=16, dt=0.01, seed=1, integrator=kind, backend="torch",
                     device="cpu").advance(w0.copy(), 4)
            np.testing.assert_allclose(a, b, atol=1e-9)

    def test_device_cuda_mps_guarded(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from modules.apps.weather.flow import NS2D
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else None)
        if dev is None:
            self.skipTest("no GPU/MPS device")
        s = NS2D(n=16, dt=0.01, backend="torch", device=dev, seed=0)
        out = s.advance(s.random_ic(), 2)
        self.assertEqual(out.shape, (16, 16))
        self.assertTrue(np.all(np.isfinite(out)))


# ── Model f_continuous delegation + aliases ──────────────────────────────────

class _Decay(KalmanModel):
    NAME = "Decay"
    def __init__(self, **kw):
        self.n = 1
        super().__init__("d", H=np.eye(1), Q=np.zeros((1, 1)), R=np.eye(1),
                         fs=10, x0=np.array([[2.0]]), **kw)
    def f_continuous(self, x, u=None, t=None):
        return -np.asarray(x, float)


class TestModelDelegation(unittest.TestCase):

    def test_f_matches_hand_rk4(self):
        m = _Decay()
        hand = RK4Integrator().step(lambda x, u, t: -x, m.x0.astype(float), dt=0.1)
        np.testing.assert_allclose(m.f(m.x0), hand)
        self.assertTrue(np.all(np.isfinite(m.F())))       # numerical Jacobian of f

    def test_check_shapes_passes_and_filters(self):
        from modules.filters import ExtendedKalmanFilter
        m = _Decay()
        rec = ExtendedKalmanFilter(m).forward(np.array([[1.9], [1.7], [1.5]]))
        self.assertEqual(np.asarray(rec.xk).shape, (3, 1))

    def test_integrator_override(self):
        rk4 = _Decay(); eul = _Decay(integrator=KalmanIntegratorKind.EULER)
        self.assertFalse(np.allclose(rk4.f(rk4.x0), eul.f(eul.x0)))
        self.assertIsInstance(eul.default_integrator(), EulerIntegrator)

    def test_disallowed_integrator_raises(self):
        class R(_Decay):
            ALLOWED_INTEGRATORS = frozenset({KalmanIntegratorKind.RK4})
        with self.assertRaises(TypeError):
            R(integrator=KalmanIntegratorKind.EULER)

    def test_substeps_advance_full_interval(self):
        class Sub(_Decay):
            SOLVER_SUBSTEPS = 5
        m = Sub()
        # 5 substeps of dt/5 over one 1/fs interval ≈ e^{-1/fs}
        np.testing.assert_allclose(m.f(m.x0).ravel(), m.x0.ravel() * np.exp(-1/m.fs), rtol=1e-4)

    def test_estimator_alias(self):
        self.assertIs(KalmanEstimator, KalmanEngine)


# ── Custom (out-of-enum) schemes ──────────────────────────────────────────────

class _Heun(KalmanIntegrator):
    """A scheme the framework has never heard of: no KalmanIntegratorKind."""
    NAME = "Heun"
    tableau = ([0.0, 1.0], [[], [1.0]], [0.5, 0.5])

    def step(self, f, x, u=None, t=None, dt=1.0):
        k1 = f(x, u, t)
        return x + 0.5 * dt * (k1 + f(x + dt * k1, u, t + dt))


class _Trivial(KalmanSolver):
    NAME = "Trivial"

    def solve(self, field, steps=1, u=None, t=None):
        return np.asarray(field)


class TestCustomSchemes(unittest.TestCase):
    """A downstream scheme must be usable without editing the kind enum."""

    def tearDown(self):
        unregister_integrator("heun")
        unregister_solver("trivial")

    def test_instance_and_class_override_accepted(self):
        self.assertIsNone(_Heun.KIND)                       # not an enum member
        for override in (_Heun(), _Heun):
            self.assertIsInstance(_Decay(integrator=override).default_integrator(), _Heun)

    def test_registered_name_resolves_everywhere(self):
        self.assertIs(register_integrator("heun", _Heun), _Heun)
        self.assertIn("heun", available_integrators())
        self.assertIsInstance(resolve_integrator("heun"), _Heun)
        self.assertIsInstance(_Decay(integrator="heun").default_integrator(), _Heun)
        # shipped kinds keep resolving by name too
        self.assertIsInstance(resolve_integrator("rk4"), RK4Integrator)

    def test_registered_solver(self):
        register_solver("trivial", _Trivial)
        self.assertIn("trivial", available_solvers())
        self.assertIsInstance(resolve_solver("trivial"), _Trivial)

    def test_shipped_kind_needs_the_enum_member_to_rebind(self):
        with self.assertRaises(ValueError):
            register_integrator("rk4", _Heun)               # ambiguous by name
        register_integrator(KalmanIntegratorKind.EULER, _Heun)   # explicit rebind
        try:
            self.assertIsInstance(resolve_integrator(KalmanIntegratorKind.EULER), _Heun)
        finally:
            INTEGRATOR_BY_KIND[KalmanIntegratorKind.EULER] = EulerIntegrator

    def test_non_subclass_rejected(self):
        with self.assertRaises(TypeError):
            register_integrator("nope", object)
        with self.assertRaises(TypeError):
            register_solver("nope", object)

    def test_unknown_kind_raises_instead_of_falling_back(self):
        """A typo must fail, not silently run a different scheme."""
        for bad in ("nonsense", "rk-4"):
            with self.assertRaises(ValueError):
                resolve_integrator(bad)
            with self.assertRaises(ValueError):
                resolve_solver(bad)
        with self.assertRaises(ValueError):
            resolve_engine("nonsense")
        with self.assertRaises(ValueError):
            _Decay(integrator="nonsense")
        # None still means "the default"
        self.assertIsInstance(resolve_integrator(None), RK4Integrator)
        self.assertIsInstance(resolve_solver(None), SpectralSolver)

    def test_allowed_list_still_gates_enum_kinds(self):
        """Custom schemes bypass ALLOWED_*; declared enum kinds still do not."""
        class R(_Decay):
            ALLOWED_INTEGRATORS = frozenset({KalmanIntegratorKind.RK4})
        with self.assertRaises(TypeError):
            R(integrator=KalmanIntegratorKind.EULER)
        self.assertIsInstance(R(integrator=_Heun()).default_integrator(), _Heun)


if __name__ == "__main__":
    unittest.main()

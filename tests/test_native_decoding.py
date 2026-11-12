"""Continuous models are decoded in C++: exp(A dt) and the Van Loan noise covariance."""
import unittest

import numpy as np
from scipy.linalg import expm

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.model import KalmanModel
from modules.framework.constants import KalmanDynamicsKind
from modules.helpers import van_loan_discretize, matrix_exponential


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
class TestDecodingInCxx(unittest.TestCase):

    def test_exponential_matches_scipy(self):
        rng = np.random.default_rng(0)
        for n, scale in ((2, 1.0), (4, 0.1), (6, 5.0), (9, 30.0)):
            A = scale * rng.standard_normal((n, n))
            want = expm(A)
            np.testing.assert_allclose(matrix_exponential(A), want, rtol=1e-9, atol=1e-9 * np.abs(want).max())

    def test_van_loan_matches_the_scipy_reference(self):
        rng = np.random.default_rng(1)
        for n in (2, 3, 6):
            A = rng.standard_normal((n, n))
            L = rng.standard_normal((n, 2))
            Qc = np.diag([0.5, 2.0])
            F, Q = van_loan_discretize(A, L, Qc, 0.05)
            with native.python_reference():
                Fr, Qr = van_loan_discretize(A, L, Qc, 0.05)
            np.testing.assert_allclose(F, Fr, atol=1e-13)
            np.testing.assert_allclose(Q, Qr, atol=1e-13)
            np.testing.assert_allclose(Q, Q.T, atol=0)

    def test_a_singular_generator(self):
        # position, velocity, white acceleration of intensity q: the textbook block
        dt, q = 0.25, 2.0
        F, Q = van_loan_discretize(np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([[0.0], [1.0]]), np.array([[q]]), dt)
        np.testing.assert_allclose(F, [[1, dt], [0, 1]], atol=1e-15)
        np.testing.assert_allclose(Q, q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), atol=1e-15)

    def test_a_linear_model_stated_in_continuous_time(self):
        A = np.array([[0.0, 1.0], [-4.0, -0.2]])
        m = KalmanModel("osc", A=A, Qc=np.array([[0.3]]), Lc=np.array([[0.0], [1.0]]),
                        H=np.array([[1.0, 0.0]]), R=np.eye(1), fs=50.0)
        np.testing.assert_allclose(m.F(), expm(A / 50.0), atol=1e-13)
        self.assertGreater(m.Q()[1, 1], 0.0)

    def test_a_nonlinear_continuous_model_states_an_intensity(self):
        class Pendulum(KalmanModel):
            DYNAMICS = KalmanDynamicsKind.CONTINUOUS
            def __init__(self, **kw):
                self.n = 2
                super().__init__("p", H=np.array([[1.0, 0.0]]), R=np.eye(1) * 1e-2, **kw)
            def f(self, x, u=None, w=None, t=None):
                th, om = np.asarray(x, float).reshape(-1)
                return np.array([[om], [-9.81 * np.sin(th) - 0.1 * om]])

        q = 0.5
        slow, fast = (Pendulum(fs=fs, Qc=np.diag([0.0, q]), x0=np.array([[0.3], [0.0]])) for fs in (50.0, 200.0))
        for m in (slow, fast):
            dt = 1.0 / m.fs
            Q = m.Q()
            self.assertAlmostEqual(Q[1, 1] / (q * dt), 1.0, places=2)         # leading term: intensity x dt
            self.assertAlmostEqual(Q[0, 0] / (q * dt**3 / 3), 1.0, places=1)  # the angle's share, through dx/dt
            np.testing.assert_allclose(Q, Q.T)
        # the same physics at two rates: four times the rate, a quarter of the noise per step
        self.assertAlmostEqual(slow.Q()[1, 1] / fast.Q()[1, 1], 4.0, places=1)

        from modules.framework.filters import ExtendedKalmanFilter
        z = 0.3 * np.cos(3.1 * np.arange(100) / 50.0).reshape(-1, 1)
        rec = ExtendedKalmanFilter(slow).forward(z)
        self.assertTrue(np.all(np.isfinite(np.asarray(rec.xk))))

    def test_an_intensity_on_a_discrete_model_is_refused(self):
        with self.assertRaises(ValueError):
            KalmanModel("d", F=np.eye(2), H=np.eye(2), Qc=np.eye(2), R=np.eye(2), fs=10.0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
class TestDifferentiators(unittest.TestCase):
    """Jacobians come from the scheme a model declares, computed in C++."""

    @staticmethod
    def fn(x):
        a, b = np.asarray(x, float).reshape(-1)
        return np.array([[a * a * b], [np.sin(a)], [3.0 * b]])

    exact = np.array([[2 * 1.3 * -0.7, 1.3 ** 2], [np.cos(1.3), 0.0], [0.0, 3.0]])
    at = np.array([[1.3], [-0.7]])

    def test_schemes_match_their_python_reference(self):
        from modules.helpers import numerical_jacobian
        for scheme, accuracy in (("forward", 1e-5), ("central", 1e-9)):
            J = numerical_jacobian(self.fn, self.at, scheme=scheme)
            with native.python_reference():
                Jr = numerical_jacobian(self.fn, self.at, scheme=scheme)
            np.testing.assert_allclose(J, Jr, rtol=0, atol=1e-9)
            np.testing.assert_allclose(J, self.exact, rtol=0, atol=accuracy)
        with self.assertRaises(ValueError):
            numerical_jacobian(self.fn, self.at, scheme="sideways")

    def test_an_error_in_the_map_comes_back_as_itself(self):
        from modules.helpers import numerical_jacobian
        calls = []
        def breaks(x):
            calls.append(1)
            if len(calls) > 1:
                raise FloatingPointError("out of range")
            return self.fn(x)
        with self.assertRaises(FloatingPointError):
            numerical_jacobian(breaks, self.at)

    def test_a_model_declares_its_differentiator(self):
        from modules.framework.constants import KalmanDifferentiatorKind as Kind

        class Step(KalmanModel):
            def __init__(self, **kw):
                self.n = 2
                super().__init__("s", H=np.array([[1.0, 0.0]]), R=np.eye(1), Q=np.eye(2) * 1e-6, fs=10.0, **kw)
            def f(self, x, u=None, w=None, t=None):
                a, b = np.asarray(x, float).reshape(-1)
                return np.array([[a + 0.1 * b], [b - 0.1 * np.sin(a)]])

        class Central(Step):
            DIFFERENTIATOR = Kind.CENTRAL

        class Formula(Step):
            DIFFERENTIATOR = Kind.ANALYTIC

        x = np.array([[0.8], [0.3]])
        exact = np.array([[1.0, 0.1], [-0.1 * np.cos(0.8), 1.0]])
        forward, central = Step().F(x=x), Central().F(x=x)
        np.testing.assert_allclose(forward, exact, atol=1e-6)
        np.testing.assert_allclose(central, exact, atol=1e-10)
        self.assertGreater(np.abs(forward - exact).max(), np.abs(central - exact).max())
        with self.assertRaises(NotImplementedError) as raised:
            Formula().F(x=x)
        self.assertIn("ANALYTIC", str(raised.exception))

    def test_a_twin_linearises_with_the_declared_scheme(self):
        from modules.apps.ballistics.extended import BallisticModel
        from modules.framework.constants import KalmanDifferentiatorKind as Kind

        class CentralBallistic(BallisticModel):
            DIFFERENTIATOR = Kind.CENTRAL

        def run(cls, reference):
            m = cls("b", fs=20.0, c_drag=0.02)
            m.initial_state([0, 30, 0, 0, 20, 0, 100, 40, -9.81])
            z = BallisticModel("b", fs=20.0, c_drag=0.02).sample(5.0, seed=4).zk
            if reference:
                with native.python_reference():
                    return np.asarray(m.filter(z).Pk)
            return np.asarray(m.filter(z).Pk)

        if CentralBallistic("b").native_handle() is None:
            self.skipTest("libKalmanApps is not built")
        twin, python = run(CentralBallistic, False), run(CentralBallistic, True)
        np.testing.assert_allclose(twin, python, rtol=1e-6, atol=1e-9)
        self.assertGreater(np.abs(twin - run(BallisticModel, False)).max(), 0.0)   # and it is not the forward one


class TestSamplerSeed(unittest.TestCase):
    def test_a_seeded_sample_repeats(self):
        from modules.apps.ballistics.extended import BallisticModel
        a, b = (BallisticModel("b", fs=20.0).sample(2.0, seed=4) for _ in range(2))
        np.testing.assert_array_equal(np.asarray(a.zk), np.asarray(b.zk))
        np.testing.assert_array_equal(np.asarray(a.xk), np.asarray(b.xk))
        c = BallisticModel("b", fs=20.0).sample(2.0, seed=5)
        self.assertGreater(np.abs(np.asarray(a.zk) - np.asarray(c.zk)).max(), 0.0)


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
class TestSamplerInCxx(unittest.TestCase):
    """The sampling loop runs in C++; the draws stay numpy's, so a seeded run is unchanged."""

    def both(self, make, duration, **kw):
        a = make().sample(duration, seed=7, **kw)
        with native.python_reference():
            b = make().sample(duration, seed=7, **kw)
        return a, b

    def check(self, a, b):
        # The covariance goes through a differenced Jacobian, so it agrees as the
        # filter's does (tests/test_native_twins.py), not to rounding.
        for name, tol in (("xk", 1e-9), ("zk", 1e-9), ("Pk", 1e-6)):
            x, y = np.asarray(getattr(a, name), float), np.asarray(getattr(b, name), float)
            self.assertEqual(x.shape, y.shape, name)
            scale = max(1.0, np.nanmax(np.abs(y)))
            self.assertLess(np.nanmax(np.abs(x - y)) / scale, tol, name)

    def test_a_twin_and_a_python_model(self):
        from modules.apps.ballistics.extended import BallisticModel
        from modules.apps.vehicles.lane import CarLaneModel

        def ballistic():
            m = BallisticModel("b", fs=20.0, c_drag=0.02)
            m.initial_state([0, 30, 0, 0, 20, 0, 100, 40, -9.81])
            return m
        self.assertIsNotNone(ballistic().native_handle())
        self.check(*self.both(ballistic, 5.0))
        self.check(*self.both(lambda: BallisticModel("b", fs=20.0, drag_uncertainty=0.3, c_drag=0.01), 3.0))   # no twin
        self.check(*self.both(lambda: CarLaneModel("lane"), 4.0))

    def test_noise_switches_and_resuming(self):
        from modules.apps.ballistics.extended import BallisticModel
        from modules.framework.sampler import KalmanSampler
        make = lambda: BallisticModel("b", fs=20.0)
        self.check(*self.both(make, 2.0, process_noise=False))
        self.check(*self.both(make, 2.0, measurement_noise=False))

        np.random.seed(3)
        s = KalmanSampler(make())
        first, second = s.measurement(1.0), s.measurement(1.0)
        np.random.seed(3)
        with native.python_reference():
            r = KalmanSampler(make())
            first_r, second_r = r.measurement(1.0), r.measurement(1.0)
        self.check(first, first_r)
        self.check(second, second_r)
        self.assertAlmostEqual(float(np.asarray(second.time)[0]), 1.0, places=9)        # it resumed where it stopped

    def test_a_python_error_in_the_model_comes_back_as_itself(self):
        from modules.framework.models.extended import ExtendedKalmanModel

        class Breaks(ExtendedKalmanModel):
            def __init__(self):
                self.n, self.m, self.l = 1, 1, 0
                super().__init__("x", Q=np.eye(1), R=np.eye(1), L=np.eye(1), fs=10.0, x0=np.ones((1, 1)))
            def f(self, x, u=None, w=None, t=None):
                if t is not None and t > 0.25:
                    raise OverflowError("diverged")
                return np.asarray(x, float)
            def h(self, x, v=None, t=None):
                return np.asarray(x, float)
        with self.assertRaises(OverflowError):
            Breaks().sample(1.0)

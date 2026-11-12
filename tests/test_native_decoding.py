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

"""The Python integrators forward to the C++ ones (Kalman::Integrators)."""
import unittest

import numpy as np

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.integrator import KalmanIntegrator
from modules.framework.integrators import EulerIntegrator, RK4Integrator, DormandPrinceIntegrator


def pendulum(x, u, t):
    th, om = np.asarray(x).reshape(-1)
    torque = 0.0 if u is None else float(np.asarray(u).reshape(-1)[0])
    return np.array([om, -9.81 * np.sin(th) - 0.1 * om + torque + 0.3 * np.cos(t)]).reshape(np.shape(x))


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")
class TestIntegratorsRunInCxx(unittest.TestCase):

    def both(self, integrator, call):
        calls = []
        def counted(x, u, t):
            calls.append(1)
            return pendulum(x, u, t)
        got = call(integrator, counted)
        n_cxx = len(calls)
        calls.clear()
        with native.python_reference():
            want = call(integrator, counted)
        return got, want, n_cxx, len(calls)

    def test_step_and_integrate_match_the_python_schemes(self):
        x = np.array([[0.7], [-0.2]])
        for integrator, atol in ((EulerIntegrator(), 1e-14), (RK4Integrator(), 1e-14),
                                 (DormandPrinceIntegrator(), 1e-12), (DormandPrinceIntegrator(rtol=1e-9, atol=1e-12), 1e-12)):
            for u in (None, np.array([[0.4]])):
                got, want, n_cxx, n_py = self.both(integrator, lambda i, f: i.step(f, x, u=u, t=1.5, dt=0.05))
                self.assertEqual(got.shape, (2, 1))
                np.testing.assert_allclose(got, want, rtol=0, atol=atol, err_msg=integrator.name())
                self.assertEqual(n_cxx, n_py, "the same number of derivative evaluations")
                got, want, _, _ = self.both(integrator, lambda i, f: i.integrate(f, x.ravel(), u=u, t=0.0, dt=0.02, n=25))
                self.assertEqual(got.shape, (2,))
                np.testing.assert_allclose(got, want, rtol=0, atol=100 * atol, err_msg=integrator.name())

    def test_the_cxx_path_is_the_one_taken(self):
        x = np.array([0.7, -0.2])
        self.assertIsNotNone(RK4Integrator()._native(pendulum, x, None, 0.0, 0.01, 1))
        with native.python_reference():
            self.assertIsNone(RK4Integrator()._native(pendulum, x, None, 0.0, 0.01, 1))

    def test_backward_in_time(self):
        x = np.array([0.7, -0.2])
        for integrator in (RK4Integrator(), DormandPrinceIntegrator()):
            there = integrator.step(pendulum, x, t=0.0, dt=0.1)
            back = integrator.step(pendulum, there, t=0.1, dt=-0.1)
            np.testing.assert_allclose(back, x, atol=1e-4)      # Dormand-Prince's default rtol is 1e-6 a substep

    def test_a_python_error_in_the_derivative_comes_back_as_itself(self):
        def broken(x, u, t):
            raise ZeroDivisionError("stalled")
        with self.assertRaises(ZeroDivisionError):
            RK4Integrator().step(broken, np.zeros(2), dt=0.1)
        def short(x, u, t):
            return np.zeros(1)
        with self.assertRaises(ValueError):
            RK4Integrator().step(short, np.zeros(2), dt=0.1)

    def test_what_stays_in_python(self):
        # a complex state (the spectral solvers' fields)
        z = np.array([1.0 + 1.0j, 0.5j])
        out = RK4Integrator().step(lambda x, u, t: 1j * x, z, dt=0.1)
        self.assertTrue(np.iscomplexobj(out))
        np.testing.assert_allclose(out, z * np.exp(0.1j), atol=1e-6)

        # a scheme written in Python
        class Midpoint(KalmanIntegrator):
            def step(self, f, x, u=None, t=None, dt=1.0):
                return x + dt * f(x + 0.5 * dt * f(x, u, t), u, t)
        out = Midpoint().integrate(lambda x, u, t: -x, np.array([1.0]), dt=0.1, n=10)
        self.assertAlmostEqual(float(out[0]), (1 - 0.1 + 0.005) ** 10, places=12)

        # a subclass that changes a built-in scheme is its own scheme
        class Halved(RK4Integrator):
            def _step(self, f, x, u=None, t=None, dt=1.0):
                return super()._step(f, x, u=u, t=t, dt=0.5 * dt)
        a = Halved().step(pendulum, np.array([0.7, -0.2]), dt=0.1)
        b = RK4Integrator().step(pendulum, np.array([0.7, -0.2]), dt=0.05)
        np.testing.assert_allclose(a, b, atol=1e-14)

    def test_a_derivative_that_integrates_inside_itself(self):
        inner = RK4Integrator()
        def nested(x, u, t):
            return inner.step(lambda y, u, t: -y, x, dt=0.01) - x
        out = RK4Integrator().step(nested, np.array([1.0, 2.0]), dt=0.1)
        with native.python_reference():
            want = RK4Integrator().step(nested, np.array([1.0, 2.0]), dt=0.1)
        np.testing.assert_allclose(out, want, atol=1e-15)


if __name__ == "__main__":
    unittest.main()

"""Tests for the framework contracts: Model construction paths, Jacobians,
covariances, engine binding/resolution, the sampler, and the composite model.

Exercises the three construction styles directly (not through an app), so the
``KalmanModel`` matrix path, the ``ExtendedKalmanModel`` override path and the
declarative engine binding are covered independently of any domain code.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")
from modules.framework import (
    LinearKalmanModel, ExtendedKalmanModel, UnscentedKalmanModel,
    CompositeKalmanModel, KalmanEngineKind,
    LinearKalmanEngine, ExtendedKalmanEngine,
)
from modules.framework.abstract.mixins import KalmanModelMixin
from modules.helpers import numerical_jacobian


def _no_nan_inf(arr, label="array"):
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


# Constant-velocity building blocks reused across the module.
DT = 0.1
F_CV = np.array([[1.0, DT], [0.0, 1.0]])
H_POS = np.array([[1.0, 0.0]])
Q_CV = np.eye(2) * 1e-3
R_POS = np.array([[0.1]])


def _linear_model(sensor="s"):
    return LinearKalmanModel(sensor, F=F_CV, H=H_POS, Q=Q_CV, R=R_POS, fs=10)


# ===========================================================================
# Linear (matrix) construction path
# ===========================================================================
class TestLinearMatrixModel(unittest.TestCase):

    def test_dims_and_default_engine(self):
        m = _linear_model()
        self.assertEqual((m.n, m.m, m.l), (2, 1, 0))
        self.assertEqual(m.DEFAULT_ENGINE, KalmanEngineKind.LINEAR)
        self.assertIsInstance(m.default_engine(), LinearKalmanEngine)

    def test_requires_F_and_H(self):
        with self.assertRaises(ValueError):
            LinearKalmanModel("s", H=H_POS, Q=Q_CV, R=R_POS, fs=10)   # no F

    def test_f_h_match_matrices(self):
        m = _linear_model()
        x = np.array([[2.0], [3.0]])
        np.testing.assert_allclose(m.f(x), F_CV @ x)
        np.testing.assert_allclose(m.h(x), H_POS @ x)
        np.testing.assert_allclose(m.F(x), F_CV)
        np.testing.assert_allclose(m.H(x), H_POS)

    def test_covariances_and_qr(self):
        m = _linear_model()
        np.testing.assert_allclose(m.Q(), Q_CV)
        np.testing.assert_allclose(m.R(), R_POS)
        Q, R = m.covariances()
        np.testing.assert_allclose(Q, Q_CV)
        np.testing.assert_allclose(R, R_POS)

    def test_jacobians_tuple_order(self):
        m = _linear_model()
        x = np.array([[1.0], [0.0]])
        F, H, B, L, M = m.jacobians(x)
        np.testing.assert_allclose(F, F_CV)
        np.testing.assert_allclose(H, H_POS)
        self.assertIsNone(B)   # no control map supplied

    def test_time_varying_callable_matrix(self):
        """A callable H(t) must be evaluated at the requested time."""
        H_t = lambda t: np.array([[np.cos(t or 0.0), 0.0]])
        m = LinearKalmanModel("s", F=F_CV, H=H_t, Q=Q_CV, R=R_POS, fs=10)
        np.testing.assert_allclose(m.H(t=0.0), [[1.0, 0.0]])
        np.testing.assert_allclose(m.H(t=np.pi), [[-1.0, 0.0]], atol=1e-12)


# ===========================================================================
# Extended (f/h override) construction path
# ===========================================================================
class _Pendulum(ExtendedKalmanModel):
    """Damped pendulum: x = [theta, omega]; only theta is observed."""

    def __init__(self):
        self.n, self.m, self.l = 2, 1, 0
        super().__init__("pendulum", Q=np.eye(2) * 1e-5, R=np.array([[0.04]]), fs=20)

    def f(self, x, u=None, w=None, t=None):
        th, om = float(x[0, 0]), float(x[1, 0])
        dt = 0.05
        return np.array([[th + om * dt], [om - 9.81 * np.sin(th) * dt]])

    def h(self, x, v=None, t=None):
        # additive measurement noise (the sampler passes v from R; matrix models add it for free)
        return x[:1] if v is None else x[:1] + v


class TestExtendedModel(unittest.TestCase):

    def test_default_engine_and_allowed(self):
        m = _Pendulum()
        self.assertEqual(m.DEFAULT_ENGINE, KalmanEngineKind.EXTENDED)
        self.assertNotIn(KalmanEngineKind.LINEAR, m.ALLOWED_ENGINES)
        self.assertIsInstance(m.default_engine(), ExtendedKalmanEngine)

    def test_numerical_jacobian_matches_default(self):
        """The model's default F()/H() are the numerical Jacobians of f()/h()."""
        m = _Pendulum()
        x = np.array([[0.4], [0.2]])
        np.testing.assert_allclose(m.F(x), numerical_jacobian(lambda xx: m.f(xx), x), atol=1e-5)
        np.testing.assert_allclose(m.H(x), numerical_jacobian(lambda xx: m.h(xx), x), atol=1e-6)

    def test_sample_then_filter_tracks(self):
        np.random.seed(0)
        m = _Pendulum()
        m.initial_state([0.5, 0.0])
        truth = m.sample(3.0)
        rec = m.filter(truth.zk)
        self.assertEqual(rec.xk.shape, (60, 2))
        _no_nan_inf(rec.xk, "pendulum xk")
        # after the initial transient the filtered angle beats the raw measurement
        tail = slice(20, None)
        rmse_filt = np.sqrt(np.mean((rec.xk[tail, 0] - truth.xk[tail, 0]) ** 2))
        rmse_meas = np.sqrt(np.mean((truth.zk[tail, 0] - truth.xk[tail, 0]) ** 2))
        self.assertLess(rmse_filt, rmse_meas)


# ===========================================================================
# Unscented model binding
# ===========================================================================
class TestUnscentedModel(unittest.TestCase):

    def test_engine_binding(self):
        class M(UnscentedKalmanModel):
            def __init__(self):
                self.n, self.m, self.l = 1, 1, 0
                super().__init__("u", Q=np.array([[1e-3]]), R=np.array([[1e-2]]), fs=10)
            def f(self, x, u=None, w=None, t=None): return np.tanh(x)
            def h(self, x, v=None, t=None): return x
        m = M()
        self.assertEqual(m.DEFAULT_ENGINE, KalmanEngineKind.UNSCENTED)
        self.assertNotIn(KalmanEngineKind.LINEAR, m.ALLOWED_ENGINES)


# ===========================================================================
# Composite model (block-stacked submodels)
# ===========================================================================
class TestCompositeModel(unittest.TestCase):

    def test_block_dimensions(self):
        comp = CompositeKalmanModel(_linear_model("a"), _linear_model("b"))
        self.assertEqual((comp.n, comp.m, comp.l), (4, 2, 0))

    def test_requires_at_least_one_submodel(self):
        with self.assertRaises(ValueError):
            CompositeKalmanModel()

    def test_mismatched_fs_rejected(self):
        a = _linear_model("a")
        b = LinearKalmanModel("b", F=F_CV, H=H_POS, Q=Q_CV, R=R_POS, fs=20)
        with self.assertRaises(ValueError):
            CompositeKalmanModel(a, b)

    def test_sample_and_filter(self):
        np.random.seed(1)
        comp = CompositeKalmanModel(_linear_model("a"), _linear_model("b"))
        truth = comp.sample(2.0)
        rec = comp.filter(truth.zk)
        self.assertEqual(rec.xk.shape, (20, 4))
        _no_nan_inf(rec.xk, "composite xk")


# ===========================================================================
# Sampler behaviour
# ===========================================================================
class TestSampler(unittest.TestCase):

    def test_sample_tracks_analytic_pk(self):
        """sample() always routes through engine._predict, so Pk carries the
        analytic prior covariance band; with process_noise=False the mean is
        also deterministic (no random draw injected)."""
        m = _linear_model()
        m.initial_state([0.0, 1.0])
        a = m.sample(1.0, process_noise=False)
        b = m.sample(1.0, process_noise=False)
        np.testing.assert_allclose(a.xk, b.xk)        # deterministic mean
        self.assertGreater(float(np.sum(np.abs(a.Pk))), 0.0)   # analytic Pk present

    def test_process_noise_toggle(self):
        """process_noise=False removes the stochastic drift from the truth path."""
        np.random.seed(2)
        m = LinearKalmanModel("s", F=F_CV, H=H_POS, Q=np.eye(2) * 0.5, R=R_POS, fs=50)
        m.initial_state([0.0, 0.0])
        noisy = m.sample(2.0, process_noise=True)
        clean = m.sample(2.0, process_noise=False)
        self.assertGreater(np.std(np.diff(noisy.xk[:, 1])),
                           np.std(np.diff(clean.xk[:, 1])))


# ===========================================================================
# Engine resolution / validation
# ===========================================================================
class TestEngineBinding(unittest.TestCase):

    def test_disallowed_engine_rejected(self):
        from modules.framework.filters import LinearKalmanFilter
        with self.assertRaises(TypeError):
            LinearKalmanFilter(_Pendulum())     # nonlinear model forbids the linear engine

    def test_plot_helpers_live_on_mixin(self):
        """The internal _plot_static/_plot_interactive renderers resolve via the mixin."""
        self.assertIn("_plot_static", vars(KalmanModelMixin))
        self.assertIn("_plot_interactive", vars(KalmanModelMixin))
        m = _linear_model()
        self.assertTrue(hasattr(m, "_plot_static"))
        self.assertTrue(callable(m.plot))


if __name__ == "__main__":
    unittest.main()

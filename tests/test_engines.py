"""Tests for framework engines — linear, extended, unscented, particle, combinatorial.

Each engine is tested directly against its _predict/_update interface,
then cross-validated against other engines on a linear model where
all should agree.
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.framework import LinearKalmanModel, ExtendedKalmanModel
from modules.framework.engines.linear      import LinearKalmanEngine
from modules.framework.engines.extended    import ExtendedKalmanEngine
from modules.framework.engines.unscented   import UnscentedKalmanEngine
from modules.framework.engines.particle    import ParticleEngine
from modules.framework.engines.combinatorial import CombinatorialKalmanEngine


# ── shared fixtures ───────────────────────────────────────────────────────────

DT = 0.1
F_CV = np.array([[1.0, DT], [0.0, 1.0]])
H_POS = np.array([[1.0, 0.0]])
Q0 = np.eye(2) * 1e-3
R0 = np.array([[0.25]])

X0 = np.array([[0.0], [1.0]])
P0 = np.eye(2) * 0.1


def _cv_model():
    return LinearKalmanModel("cv", F=F_CV, H=H_POS, Q=Q0, R=R0, fs=10)


def _assert_valid_cov(P, n, label="P"):
    P = np.asarray(P)
    assert P.shape == (n, n),      f"{label} shape {P.shape} != ({n},{n})"
    assert not np.any(np.isnan(P)), f"{label} contains NaN"
    assert not np.any(np.isinf(P)), f"{label} contains Inf"
    eigv = np.linalg.eigvalsh(P)
    assert np.all(eigv > -1e-8),    f"{label} not PSD (min eig {eigv.min():.2e})"


# ── LinearKalmanEngine ────────────────────────────────────────────────────────

class TestLinearKalmanEngine(unittest.TestCase):

    def setUp(self):
        self.m = _cv_model()
        self.e = LinearKalmanEngine()

    def test_predict_shape(self):
        xp, Pp, F = self.e._predict(self.m, X0, P0, t=0.0)
        self.assertEqual(xp.shape, (2, 1))
        _assert_valid_cov(Pp, 2, "Pp")
        self.assertEqual(F.shape, (2, 2))

    def test_predict_state_transition(self):
        x = np.array([[1.0], [2.0]])
        xp, _, _ = self.e._predict(self.m, x, P0, t=0.0)
        expected = F_CV @ x
        np.testing.assert_allclose(xp, expected, atol=1e-10)

    def test_predict_with_control(self):
        m = LinearKalmanModel("cv_ctrl", F=F_CV, H=H_POS, Q=Q0, R=R0, fs=10,
                               B=np.array([[0.0], [1.0]]))
        e = LinearKalmanEngine()
        u = np.array([[0.5]])
        xp, _, _ = e._predict(m, X0, P0, u=u, t=0.0)
        expected = F_CV @ X0 + np.array([[0.0], [0.5]])
        np.testing.assert_allclose(xp, expected, atol=1e-10)

    def test_predict_covariance_grows(self):
        _, Pp, _ = self.e._predict(self.m, X0, P0, t=0.0)
        # Prior covariance must be larger than initial (process noise added)
        self.assertGreater(np.trace(Pp), np.trace(P0))

    def test_update_reduces_uncertainty(self):
        xp, Pp, _ = self.e._predict(self.m, X0, P0, t=0.0)
        z = np.array([[0.5]])
        x_post, P_post = self.e._update(self.m, xp, Pp, z, t=0.0)
        self.assertEqual(x_post.shape, (2, 1))
        _assert_valid_cov(P_post, 2, "P_post")
        self.assertLess(P_post[0, 0], Pp[0, 0])   # position variance reduced

    def test_update_moves_towards_measurement(self):
        x_prior = np.array([[3.0], [0.0]])          # predicted position = 3
        P_prior  = np.eye(2) * 10                   # high uncertainty
        z = np.array([[0.0]])                        # observed position = 0
        x_post, _ = self.e._update(self.m, x_prior, P_prior, z, t=0.0)
        # posterior should be pulled toward z=0
        self.assertLess(x_post[0, 0], x_prior[0, 0])

    def test_nan_measurement_skipped(self):
        """Missing measurement (NaN) → predict-only, covariance unchanged."""
        xp, Pp, _ = self.e._predict(self.m, X0, P0, t=0.0)
        z_nan = np.array([[float("nan")]])
        x_post, P_post = self.e._update(self.m, xp, Pp, z_nan, t=0.0)
        np.testing.assert_array_equal(x_post, xp)
        np.testing.assert_array_equal(P_post, Pp)

    def test_measure_returns_H_and_pred(self):
        zk, H = self.e._measure(self.m, X0, t=0.0)
        np.testing.assert_array_equal(H, H_POS)
        np.testing.assert_allclose(zk, H_POS @ X0, atol=1e-12)


# ── ExtendedKalmanEngine ──────────────────────────────────────────────────────

class _NonlinearModel(ExtendedKalmanModel):
    """Angle-only observation (nonlinear h)."""

    def __init__(self):
        self.n, self.m, self.l = 2, 1, 0
        super().__init__("angle", Q=Q0, R=np.array([[0.01]]), fs=10)

    def f(self, x, u=None, w=None, t=None):
        return F_CV @ x

    def h(self, x, v=None, t=None):
        angle = np.arctan2(float(x[1, 0]), float(x[0, 0]))
        if v is not None:
            angle += float(np.asarray(v).reshape(-1)[0])
        return np.array([[angle]])


class TestExtendedKalmanEngine(unittest.TestCase):

    def setUp(self):
        self.m_lin = _cv_model()
        self.m_nl  = _NonlinearModel()
        self.e     = ExtendedKalmanEngine()

    def test_agrees_with_linear_on_linear_model(self):
        """EKF == KF when f and h are linear."""
        e_lin = LinearKalmanEngine()
        xp_kf, Pp_kf, _ = e_lin._predict(self.m_lin, X0, P0, t=0.0)
        xp_ek, Pp_ek, _ = self.e._predict(self.m_lin, X0, P0, t=0.0)
        np.testing.assert_allclose(xp_kf, xp_ek, atol=1e-10)
        np.testing.assert_allclose(Pp_kf, Pp_ek, atol=1e-10)

    def test_predict_nonlinear_model(self):
        x = np.array([[1.0], [0.0]])
        xp, Pp, F = self.e._predict(self.m_nl, x, np.eye(2) * 0.5, t=0.0)
        self.assertEqual(xp.shape, (2, 1))
        _assert_valid_cov(Pp, 2, "EKF Pp")

    def test_update_nonlinear_model(self):
        x = np.array([[1.0], [1.0]])
        P = np.eye(2) * 0.5
        xp, Pp, _ = self.e._predict(self.m_nl, x, P, t=0.0)
        z = np.array([[np.pi / 4 + 0.01]])    # approx arctan(1/1)
        x_post, P_post = self.e._update(self.m_nl, xp, Pp, z, t=0.0)
        self.assertEqual(x_post.shape, (2, 1))
        _assert_valid_cov(P_post, 2, "EKF P_post")


# ── UnscentedKalmanEngine ─────────────────────────────────────────────────────

class TestUnscentedKalmanEngine(unittest.TestCase):

    def setUp(self):
        self.m_lin = _cv_model()
        self.e     = UnscentedKalmanEngine()

    def test_agrees_with_kf_on_linear_model(self):
        e_kf = LinearKalmanEngine()
        xp_kf, Pp_kf, _ = e_kf._predict(self.m_lin, X0, P0, t=0.0)
        xp_uk, Pp_uk, _ = self.e._predict(self.m_lin, X0, P0, t=0.0)
        np.testing.assert_allclose(xp_kf, xp_uk, atol=1e-6)
        np.testing.assert_allclose(Pp_kf, Pp_uk, atol=1e-6)

    def test_predict_returns_valid_shapes(self):
        xp, Pp, F = self.e._predict(self.m_lin, X0, P0, t=0.0)
        self.assertEqual(xp.shape, (2, 1))
        _assert_valid_cov(Pp, 2, "UKF Pp")

    def test_update_on_linear_model(self):
        e_kf = LinearKalmanEngine()
        xp, Pp, _ = e_kf._predict(self.m_lin, X0, P0, t=0.0)
        z = np.array([[0.5]])
        x_kf, P_kf = e_kf._update(self.m_lin, xp, Pp, z, t=0.0)
        x_uk, P_uk = self.e._update(self.m_lin, xp, Pp, z, t=0.0)
        np.testing.assert_allclose(x_kf, x_uk, atol=1e-6)
        np.testing.assert_allclose(P_kf, P_uk, atol=1e-6)

    def test_predict_covariance_is_psd(self):
        _, Pp, _ = self.e._predict(self.m_lin, X0, P0 * 5, t=0.0)
        _assert_valid_cov(Pp, 2, "UKF Pp (large P0)")


# ── ParticleEngine ────────────────────────────────────────────────────────────

class TestParticleEngine(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.m = _cv_model()
        self.m.initial_state(X0.ravel(), P0)
        self.e = ParticleEngine(n_particles=200)

    def test_predict_returns_mean_cov(self):
        xp, Pp, F = self.e._predict(self.m, X0, P0, t=0.0)
        self.assertEqual(xp.shape, (2, 1))
        _assert_valid_cov(Pp, 2, "PF Pp")

    def test_predict_mean_close_to_linear(self):
        e_lin = LinearKalmanEngine()
        xp_kf, _, _ = e_lin._predict(self.m, X0, P0, t=0.0)
        xp_pf, _, _ = self.e._predict(self.m, X0, P0, t=0.0)
        # With 200 particles the mean should be within noise
        np.testing.assert_allclose(xp_kf, xp_pf, atol=0.3)

    def test_update_produces_valid_state(self):
        xp, Pp, _ = self.e._predict(self.m, X0, P0, t=0.0)
        z = np.array([[0.2]])
        x_post, P_post = self.e._update(self.m, xp, Pp, z, t=0.0)
        self.assertEqual(x_post.shape, (2, 1))
        self.assertFalse(np.any(np.isnan(x_post)))

    def test_sequential_predict_update_tracks(self):
        np.random.seed(5)
        x    = X0.copy()
        P    = P0.copy()
        rmse = []
        truth = 0.0
        for _ in range(30):
            truth += 1.0 * DT
            z     = np.array([[truth + np.random.normal(0, 0.5)]])
            x, P, _ = self.e._predict(self.m, x, P, t=0.0)
            x, P    = self.e._update(self.m, x, P, z, t=0.0)
            rmse.append((float(x[0, 0]) - truth) ** 2)
        rmse_val = np.sqrt(np.mean(rmse))
        self.assertLess(rmse_val, 1.0, f"PF RMSE {rmse_val:.3f} too high")


# ── CombinatorialKalmanEngine ─────────────────────────────────────────────────

class TestCombinatorialKalmanEngine(unittest.TestCase):

    def setUp(self):
        self.m = _cv_model()
        self.e = CombinatorialKalmanEngine()

    def test_agrees_with_ekf_on_linear_model(self):
        e_ek = ExtendedKalmanEngine()
        xp_ck, Pp_ck, _ = self.e._predict(self.m, X0, P0, t=0.0)
        xp_ek, Pp_ek, _ = e_ek._predict(self.m, X0, P0, t=0.0)
        np.testing.assert_allclose(xp_ck, xp_ek, atol=1e-10)
        np.testing.assert_allclose(Pp_ck, Pp_ek, atol=1e-10)

    def test_predict_returns_valid_cov(self):
        _, Pp, _ = self.e._predict(self.m, X0, P0, t=0.0)
        _assert_valid_cov(Pp, 2, "CKF Pp")

    def test_update_reduces_uncertainty(self):
        xp, Pp, _ = self.e._predict(self.m, X0, P0, t=0.0)
        z = np.array([[0.5]])
        _, P_post = self.e._update(self.m, xp, Pp, z, t=0.0)
        self.assertLess(P_post[0, 0], Pp[0, 0])


# ── Cross-engine agreement (full step) ───────────────────────────────────────

class TestCrossEngineFullStep(unittest.TestCase):
    """All deterministic engines produce the same result on a linear/Gaussian model."""

    def _run_engine(self, engine):
        m   = _cv_model()
        x   = X0.copy()
        P   = P0.copy()
        zk  = [np.array([[k * DT + 0.01]]) for k in range(20)]
        ests = []
        for z in zk:
            x, P, _ = engine._predict(m, x, P, t=0.0)
            x, P    = engine._update(m, x, P, z, t=0.0)
            ests.append(float(x[0, 0]))
        return np.array(ests)

    def test_linear_ekf_ckf_agree(self):
        xk_lin = self._run_engine(LinearKalmanEngine())
        xk_ek  = self._run_engine(ExtendedKalmanEngine())
        xk_ck  = self._run_engine(CombinatorialKalmanEngine())
        np.testing.assert_allclose(xk_lin, xk_ek,  atol=1e-8)
        np.testing.assert_allclose(xk_lin, xk_ck,  atol=1e-8)

    def test_ukf_close_to_kf(self):
        xk_lin = self._run_engine(LinearKalmanEngine())
        xk_uk  = self._run_engine(UnscentedKalmanEngine())
        np.testing.assert_allclose(xk_lin, xk_uk, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

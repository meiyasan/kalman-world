"""Tests for the public filter terminologies (KF / EKF / UKF / PF) in ``modules.framework.filters``.

Each filter is a thin ``KalmanFilter`` leaf binding a model to an engine. These tests
run every engine on a shared model so the sigma-point (UKF) and particle (PF) engines —
otherwise only reached through apps — are exercised directly, and check that on a linear
problem the four agree.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")
from modules.framework import LinearKalmanModel, ExtendedKalmanModel, KalmanEngineKind
from modules.framework.filters import (
    LinearKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter,
)


def _no_nan_inf(arr, label="array"):
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


DT = 0.1
F_CV = np.array([[1.0, DT], [0.0, 1.0]])
H_POS = np.array([[1.0, 0.0]])


def _cv_model():
    return LinearKalmanModel("cv", F=F_CV, H=H_POS,
                             Q=np.eye(2) * 1e-3, R=np.array([[0.25]]), fs=10)


def _make_track(model, seed=0):
    np.random.seed(seed)
    model.initial_state([0.0, 1.0])
    return model.sample(6.0)


# ===========================================================================
# Linear Kalman filter
# ===========================================================================
class TestLinearKalmanFilter(unittest.TestCase):

    def test_forward_backward_shapes(self):
        m = _cv_model()
        truth = _make_track(m)
        kf = LinearKalmanFilter(m)
        rec = kf.forward(truth.zk, record=True)
        self.assertEqual(rec.xk.shape, (60, 2))
        _no_nan_inf(rec.xk, "kf xk")
        rec_s = kf.backward()
        self.assertEqual(rec_s.xk.shape, (60, 2))
        _no_nan_inf(rec_s.xk, "kf smoothed")

    def test_reset_clears_run_state(self):
        m = _cv_model()
        truth = _make_track(m)
        kf = LinearKalmanFilter(m)
        kf.forward(truth.zk)
        self.assertGreater(len(kf.posterior), 0)
        kf.reset()
        self.assertEqual(len(kf.posterior), 0)

    def test_smoother_not_worse_than_forward(self):
        m = _cv_model()
        truth = _make_track(m, seed=7)
        kf = LinearKalmanFilter(m)
        fwd = kf.forward(truth.zk, record=True)
        sm = kf.backward()
        rmse_f = np.sqrt(np.mean((fwd.xk[:, 0] - truth.xk[:, 0]) ** 2))
        rmse_s = np.sqrt(np.mean((sm.xk[:, 0] - truth.xk[:, 0]) ** 2))
        self.assertLessEqual(rmse_s, rmse_f * 1.05)   # smoothing should help (allow tiny slack)


# ===========================================================================
# Cross-engine agreement on a linear problem
# ===========================================================================
class TestCrossEngineAgreement(unittest.TestCase):

    def test_kf_ekf_ukf_agree_on_linear_model(self):
        m = _cv_model()
        truth = _make_track(m, seed=3)
        x_kf = LinearKalmanFilter(m).forward(truth.zk, record=True).xk
        x_ekf = ExtendedKalmanFilter(m).forward(truth.zk, record=True).xk
        x_ukf = UnscentedKalmanFilter(m).forward(truth.zk, record=True).xk
        # On an exactly-linear/Gaussian model all three are the same estimator.
        np.testing.assert_allclose(x_kf, x_ekf, atol=1e-6)
        np.testing.assert_allclose(x_kf, x_ukf, atol=1e-4)

    def test_particle_filter_tracks_linear_model(self):
        m = _cv_model()
        truth = _make_track(m, seed=4)
        pf = ParticleFilter(m, n_particles=400)
        rec = pf.forward(truth.zk)
        self.assertEqual(rec.xk.shape, (60, 2))
        _no_nan_inf(rec.xk, "pf xk")
        # the cloud mean should follow the position truth to within the noise scale
        rmse = np.sqrt(np.mean((rec.xk[20:, 0] - truth.xk[20:, 0]) ** 2))
        self.assertLess(rmse, 1.0)

    def test_particle_filter_has_no_smoother(self):
        m = _cv_model()
        pf = ParticleFilter(m, n_particles=100)
        with self.assertRaises(NotImplementedError):
            pf.backward()


# ===========================================================================
# Nonlinear models: EKF vs UKF
# ===========================================================================
class _Projectile(ExtendedKalmanModel):
    """2-D projectile with quadratic drag; position observed."""

    def __init__(self):
        self.n, self.m, self.l = 4, 2, 0
        super().__init__("proj", Q=np.eye(4) * 1e-4,
                         R=np.eye(2) * 0.25, fs=20)

    def f(self, x, u=None, w=None, t=None):
        px, py, vx, vy = x.ravel()
        dt, k, g = 0.05, 0.02, 9.81
        speed = np.hypot(vx, vy)
        return np.array([[px + vx * dt],
                         [py + vy * dt],
                         [vx - k * speed * vx * dt],
                         [vy - (g + k * speed * vy) * dt]])

    def h(self, x, v=None, t=None):
        # additive measurement noise (sampler passes v drawn from R)
        return x[:2] if v is None else x[:2] + v


class TestNonlinearFilters(unittest.TestCase):

    def _truth(self, seed=0):
        np.random.seed(seed)
        m = _Projectile()
        m.initial_state([0.0, 0.0, 12.0, 18.0])
        return m, m.sample(3.0)

    def test_ekf_and_ukf_track_projectile(self):
        m, truth = self._truth(seed=11)
        tail = slice(15, None)   # skip the initial-covariance transient
        rmse_meas = np.sqrt(np.mean((truth.zk[tail] - truth.xk[tail, :2]) ** 2))
        for Filt in (ExtendedKalmanFilter, UnscentedKalmanFilter):
            rec = Filt(m).forward(truth.zk, record=True)
            self.assertEqual(rec.xk.shape, (60, 4))
            _no_nan_inf(rec.xk, f"{Filt.__name__} xk")
            rmse = np.sqrt(np.mean((rec.xk[tail, :2] - truth.xk[tail, :2]) ** 2))
            self.assertLess(rmse, rmse_meas, f"{Filt.__name__} should beat raw position")

    def test_filter_KIND_matches_engine(self):
        self.assertEqual(LinearKalmanFilter.KIND, KalmanEngineKind.LINEAR)
        self.assertEqual(ExtendedKalmanFilter.KIND, KalmanEngineKind.EXTENDED)
        self.assertEqual(UnscentedKalmanFilter.KIND, KalmanEngineKind.UNSCENTED)
        self.assertEqual(ParticleFilter.KIND, KalmanEngineKind.PARTICLE)


if __name__ == "__main__":
    unittest.main()

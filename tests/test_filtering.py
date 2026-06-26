"""
Tests for the comparison/teaching helpers and apps:
  - signals   (runtime.signals: Wiener/low-pass/high-pass baselines, pull, error_ratio)
  - fusion    (SensorFusionModel, multi-sensor variance reduction)
  - detection (DetectionModel, linear object tracker with occlusion)

Follows the conventions of ``test_module.py`` / ``test_tracking.py``.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")
from modules.apps import (
    SensorFusionModel, SensorSpec,
    DetectionModel,
    GPSModel,
)
from modules.runtime.signals import (
    steady_state_gain, WienerFilter, LowPassFilter, HighPassFilter, pull, error_ratio,
)


def _no_nan_inf(arr, label="array"):
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


def _assert_record(tc, rec, name, n, m, N):
    tc.assertEqual(rec.xk.shape[0], N, f"{name} xk time dim")
    tc.assertEqual(rec.xk.shape[1], n, f"{name} xk state dim")
    tc.assertEqual(rec.zk.shape[0], N, f"{name} zk time dim")
    tc.assertEqual(rec.zk.shape[1], m, f"{name} zk obs dim")
    _no_nan_inf(rec.xk, f"{name} xk")
    _no_nan_inf(rec.zk, f"{name} zk")
    _no_nan_inf(rec.Pk, f"{name} Pk")


# ===========================================================================
# App 1 — scalar signal helpers / Kalman ≡ low-pass ≡ Wiener
#   (runtime.signals — no Kalman model class needed)
# ===========================================================================
def _random_walk(n, q, r, seed=0):
    """Synthetic local-level random walk: truth x_{k+1}=x_k+w (w~N(0,q²)) and z=x+v (v~N(0,r²))."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0.0, q, n))
    z = x + rng.normal(0.0, r, n)
    return x, z


def _scalar_kf(z, q, r):
    """Plain scalar constant-model Kalman filter; returns (estimate, posterior_var) per step."""
    q2, r2 = q * q, r * r
    x, P = float(z[0]), r2
    xs, Ps = np.empty_like(z), np.empty_like(z)
    for k, zk in enumerate(z):
        P = P + q2                       # predict
        K = P / (P + r2)                 # gain
        x = x + K * (zk - x)             # update
        P = (1.0 - K) * P
        xs[k], Ps[k] = x, P
    return xs, Ps


class TestScalarSignal(unittest.TestCase):

    def test_steady_state_gain_matches_converged_kalman(self):
        """The closed-form K_ss must match the gain the recursion converges to."""
        q, r = 0.05, 1.0
        ss = steady_state_gain(q, r)
        # Iterate the scalar Riccati recursion to convergence.
        q2, r2 = q ** 2, r ** 2
        P = r2
        for _ in range(5000):
            P = P + q2                       # predict
            K = P / (P + r2)                 # gain
            P = (1.0 - K) * P                # update
        self.assertAlmostEqual(K, ss.gain, places=6)
        self.assertTrue(0.0 < ss.gain < 1.0)

    def test_steady_state_gain_edge_cases(self):
        self.assertEqual(steady_state_gain(0.0, 1.0).gain, 0.0)   # static signal → no trust in data
        self.assertEqual(steady_state_gain(1.0, 0.0).gain, 1.0)   # perfect sensor → follow it fully

    def test_lowpass_from_qr_reproduces_steady_state_kalman(self):
        """A low-pass seeded with K_ss matches the steady-state scalar Kalman estimate."""
        q, r = 0.1, 1.0
        _, z = _random_walk(1600, q, r, seed=0)
        xk, _ = _scalar_kf(z, q, r)                             # converged Kalman posterior
        lp = LowPassFilter.from_qr(q, r).apply(z)
        tail = slice(len(lp) // 2, None)                       # after the gain has converged
        diff = np.std(xk[tail] - lp[tail])
        self.assertLess(diff, 1e-9, f"low-pass vs steady Kalman diff too large: {diff:.2e}")

    def test_complementary_highpass(self):
        q, r = 0.1, 1.0
        _, z = _random_walk(400, q, r, seed=1)
        lp = LowPassFilter.from_qr(q, r).apply(z)
        hp = HighPassFilter.from_qr(q, r).apply(z)
        np.testing.assert_allclose(lp + hp, z, atol=1e-9)       # low + high == identity

    def test_pull_is_unit_gaussian(self):
        """Scalar-KF pulls should have ~unit std (well-calibrated covariance)."""
        q, r = 0.05, 1.0
        stds = []
        for seed in range(12):
            x, z = _random_walk(1200, q, r, seed=seed)
            xk, Pk = _scalar_kf(z, q, r)
            burn = 200
            stds.append(np.std(pull(xk[burn:], x[burn:], np.sqrt(Pk[burn:]))))
        self.assertAlmostEqual(float(np.mean(stds)), 1.0, delta=0.15)

    def test_filter_beats_raw(self):
        q, r = 0.05, 1.0
        x, z = _random_walk(1200, q, r, seed=3)
        xk, _ = _scalar_kf(z, q, r)
        burn = 200
        ratio = error_ratio(z[burn:] - x[burn:], xk[burn:] - x[burn:])
        self.assertGreater(ratio, 1.5, "Kalman should noticeably beat the raw measurement")

    def test_wiener_response_dc_passes(self):
        wf = WienerFilter.from_qr(0.05, 1.0)
        self.assertAlmostEqual(float(wf.response(np.array([0.0]))[0]), 1.0, places=6)  # DC → 1
        self.assertLess(float(wf.response(np.array([np.pi]))[0]), 0.5)                 # Nyquist attenuated


# ===========================================================================
# App 2 — sensor fusion
# ===========================================================================
class TestSensorFusion(unittest.TestCase):

    def _model(self):
        sensors = [SensorSpec("GPS", "position", 5.0, 10),
                   SensorSpec("Radar", "position", 1.0, 50),
                   SensorSpec("IMU", "velocity", 0.5, 50)]
        return SensorFusionModel("fuse", sensors, fs=50, process_uncertainty=2.0)

    def test_dims(self):
        m = self._model()
        self.assertEqual((m.n, m.m, m.l), (2, 3, 0))           # [pos,vel] state, 3 sensors

    def test_multirate_measurements_filter(self):
        np.random.seed(0)
        m = self._model()
        m.initial_state([0.0, 8.0])
        truth = m.sample(6.0)
        zk = m.make_measurements(truth, seed=1)
        self.assertEqual(zk.shape, (300, 3))
        self.assertGreater(np.isnan(zk[:, 0]).mean(), 0.5)     # 10 Hz sensor mostly NaN at 50 Hz base
        rec = m.filter(zk)
        _assert_record(self, rec, "SensorFusion", 2, 3, 300)

    def test_fusion_beats_every_single_sensor(self):
        np.random.seed(0)
        m = self._model()
        m.initial_state([0.0, 8.0])
        truth = m.sample(6.0)
        zk = m.make_measurements(truth, seed=1)
        rec = m.filter(zk)
        fused_var = rec.Pk[-1, 0, 0]
        for i, s in enumerate(m.sensors):
            single = m.single_sensor_record(zk, i)
            self.assertLess(fused_var, single.Pk[-1, 0, 0],
                            f"fused position variance should beat {s.name}-only")


# ===========================================================================
# App 3 — object detection / tracking
# ===========================================================================
class TestDetection(unittest.TestCase):

    def test_dims(self):
        m = DetectionModel("cam", fs=30)
        self.assertEqual((m.n, m.m, m.l), (4, 2, 0))           # [x,y,vx,vy], observe (x,y)
        self.assertEqual(len(list(m.states)), 4)               # distinct enum values → 4 members

    def test_track_beats_raw_on_detected_frames(self):
        """On a constant-velocity object the CV tracker denoises the detections."""
        d = DetectionModel("cam", fs=30, measurement_uncertainty=10.0, process_uncertainty=60.0)
        truth = d.synthetic_track(6.0, motion="linear", seed=2)
        det = d.detect(truth, occlusion=(2.5, 3.2), seed=2)
        rec = d.filter(det)
        _assert_record(self, rec, "Detection", 4, 2, len(truth))
        m = ~np.isnan(det[:, 0])
        rmse_track = np.sqrt(np.mean((rec.xk[m, :2] - truth.xk[m, :2]) ** 2))
        rmse_raw = np.sqrt(np.mean((det[m] - truth.xk[m, :2]) ** 2))
        self.assertLess(rmse_track, rmse_raw, "tracker should beat raw detections")

    def test_covariance_inflates_through_occlusion(self):
        """During the occlusion (predict-only) the position variance must grow."""
        d = DetectionModel("cam", fs=30, measurement_uncertainty=8.0, process_uncertainty=40.0)
        truth = d.synthetic_track(6.0, motion="linear", seed=2)
        det = d.detect(truth, occlusion=(2.5, 3.2), seed=2)
        rec = d.filter(det)
        gap = np.isnan(det[:, 0])
        k0 = np.argmax(gap)                                    # first occluded frame
        k1 = len(gap) - 1 - np.argmax(gap[::-1])               # last occluded frame
        self.assertGreater(rec.Pk[k1, 0, 0], rec.Pk[k0, 0, 0],
                           "covariance should inflate across the occlusion")
        # ... and contract again once detections resume.
        if k1 + 5 < len(rec):
            self.assertLess(rec.Pk[k1 + 5, 0, 0], rec.Pk[k1, 0, 0])


# ===========================================================================
# App 4 — GPS positioning with relativity
# ===========================================================================
class TestGPS(unittest.TestCase):

    def _model(self, **kw):
        return GPSModel("rx", n_sat=8, fs=1.0, pseudorange_uncertainty=1.0, **kw)

    def test_dims(self):
        m = self._model()
        self.assertEqual((m.n, m.m, m.l), (5, 8, 0))          # [x,y,z,b,b_dot], 8 pseudoranges

    def test_analytic_H_matches_numerical(self):
        from modules.helpers import numerical_jacobian
        m = self._model()
        x = m.x0.copy()
        Hn = numerical_jacobian(lambda xx: m.h(xx, t=10.0), x)
        Ha = m.H(x, t=10.0)
        # Position columns are well-conditioned; compare them to the numerical jacobian.
        np.testing.assert_allclose(Ha[:, :3], Hn[:, :3], atol=1e-4)
        # Each line-of-sight row is a unit vector; the clock-bias column is exactly 1
        # (the numerical step underflows against the ~2e7 m range, so assert by construction).
        np.testing.assert_allclose(np.linalg.norm(Ha[:, :3], axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(Ha[:, 3], 1.0, atol=1e-12)
        np.testing.assert_allclose(Ha[:, 4], 0.0, atol=1e-12)

    def test_relativistic_correction_improves_position(self):
        """Applying the relativistic correction must shrink the position error."""
        m = self._model()
        z = m.simulate(900.0, seed=0)
        self.assertEqual(z.shape, (900, 8))
        guess = np.concatenate([m.receiver_ecef + [40, -30, 20], [0.0, 0.0]])

        m.apply_relativistic = True
        m.initial_state(guess)
        rec_corr = m.filter(z)
        m.apply_relativistic = False
        m.initial_state(guess)
        rec_unc = m.filter(z)

        rmse_corr = np.sqrt(np.mean(m.position_error(rec_corr)[300:] ** 2))
        rmse_unc = np.sqrt(np.mean(m.position_error(rec_unc)[300:] ** 2))
        self.assertLess(rmse_corr, 3.0, "GR-corrected position should be accurate")
        self.assertGreater(rmse_unc, 3.0 * rmse_corr, "uncorrected should be clearly worse")

    def test_clock_drift_recovers_secular_rate(self):
        """Uncorrected, the clock-drift state absorbs the ~38.6 us/day secular rate."""
        m = self._model()
        z = m.simulate(1800.0, seed=1)
        m.apply_relativistic = False
        m.initial_state(np.concatenate([m.receiver_ecef, [0.0, 0.0]]))
        rec = m.filter(z)
        b_dot = rec.xk[-1, 4]                                  # m/s
        secular = 299792458.0 * 38.6e-6 / 86400.0             # ~0.134 m/s
        self.assertGreater(b_dot, 0.5 * secular)
        self.assertLess(b_dot, 2.0 * secular)


if __name__ == "__main__":
    unittest.main()

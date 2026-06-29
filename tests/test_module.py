import sys
import unittest
import numpy as np

# Allow running from project root or tests/ directory
sys.path.insert(0, ".")
from modules.apps import (
    OscillatorModel,
    BallisticLinearModel,
    SuspensionModel,
)
from modules.filters import LinearKalmanFilter


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _no_nan_inf(arr, label="array"):
    """Assert that a numpy array contains no NaN or Inf."""
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


# ===========================================================================
# Dimension checks
# ===========================================================================
class TestModelDimensions(unittest.TestCase):
    """Oscillator/PowerGrid dimension checks live in test_oscillators.py /
    test_powergrids.py — this file covers the remaining apps (ballistic, suspension)
    plus engine-level behavior shared across all models."""

    def test_ballistic_linear_dims(self):
        m = BallisticLinearModel("gps", fs=10)
        self.assertEqual(m.n, 3)   # [pos, vel, acc] for 1D (order-2 kinematic)
        self.assertEqual(m.m, 1)   # single position observation
        self.assertEqual(m.l, 0)

    def test_suspension_dims(self):
        m = SuspensionModel("road", fs=256)
        self.assertEqual(m.n, 4)   # [z_s, dz_s, z_u, dz_u]
        self.assertEqual(m.m, 1)   # sprung displacement only
        self.assertEqual(m.l, 1)   # road input


# ===========================================================================
# Sample tests (no crash, valid shapes)
# ===========================================================================
class TestModelSample(unittest.TestCase):

    def test_ballistic_linear_sample(self):
        m = BallisticLinearModel("gps", fs=10)
        rec = m.sample(1.0)
        self.assertEqual(rec.xk.shape[0], 10)
        _no_nan_inf(rec.xk)

    def test_suspension_sample_no_crash(self):
        m = SuspensionModel("road", fs=256)
        rec = m.sample(0.5)
        _no_nan_inf(rec.xk, "SuspensionModel xk")

    def test_sample_injects_process_noise_and_tracks_covariance(self):
        """sample() routes the truth through engine._predict (so Pk carries the
        analytic prior covariance band) and injects an explicit noise draw into
        the mean (so xk is a genuine drifting stochastic realisation)."""
        np.random.seed(0)
        m = BallisticLinearModel("gps", fs=100, process_uncertainty=1.0)
        rec = m.sample(2.0)
        self.assertGreater(np.std(np.diff(rec.xk[:, 0])), 0.0)     # truth actually moves
        self.assertGreater(float(np.sum(np.abs(rec.Pk))), 0.0)     # analytic Pk present


# ===========================================================================
# Filter tests (no crash, correct shape, no NaN/Inf)
# ===========================================================================
class TestModelFilter(unittest.TestCase):

    def _assert_record(self, rec, name, n, m, N):
        self.assertEqual(rec.xk.shape[0], N, f"{name} xk time dim")
        self.assertEqual(rec.xk.shape[1], n, f"{name} xk state dim")
        self.assertEqual(rec.zk.shape[0], N, f"{name} zk time dim")
        self.assertEqual(rec.zk.shape[1], m, f"{name} zk obs dim")
        _no_nan_inf(rec.xk, f"{name} xk")
        _no_nan_inf(rec.zk, f"{name} zk")
        _no_nan_inf(rec.Pk, f"{name} Pk")

    def test_ballistic_linear_filter(self):
        m = BallisticLinearModel("gps", fs=10)
        rec_s = m.sample(1.0)
        rec_f = m.filter(rec_s.zk)
        self._assert_record(rec_f, "BallisticLinear", m.n, m.m, 10)

    def test_suspension_filter_with_road_input(self):
        N, fs = 256, 256
        s = SuspensionModel("road", fs=fs)
        zk = 0.001 * np.random.randn(N, 1)
        uk = 0.005 * np.sin(2 * np.pi * 2 * np.arange(N) / fs).reshape(N, 1)
        rec_f = s.filter(zk, uk)
        self._assert_record(rec_f, "SuspensionModel", 4, 1, N)


# ===========================================================================
# Filter convergence / tracking test
# ===========================================================================
class TestFilterConvergence(unittest.TestCase):

    def test_suspension_covariance_reaches_steady_state(self):
        """Filter covariance should stabilise (vary < 50%) in the second half."""
        s = SuspensionModel("road", fs=256, P0=np.eye(4) * 1.0)  # large initial P
        N = 1024
        zk = 0.001 * np.random.randn(N, 1)
        uk = np.zeros((N, 1))
        rec_f = s.filter(zk, uk)

        # Trace should be much smaller after convergence than at the start
        trace_start = np.trace(rec_f.Pk[0])
        trace_mid   = np.trace(rec_f.Pk[N // 2])
        self.assertLess(trace_mid, trace_start,
                        "Suspension covariance should drop from large initial value")


# ===========================================================================
# NaN-masked multi-rate filter test
# ===========================================================================
class TestMultiRate(unittest.TestCase):

    def test_multirate_via_nan_mask(self):
        """A plain filter handles multi-rate: NaN rows -> predict-only (engine masks update)."""
        from modules.helpers import make_multirate_input

        m = BallisticLinearModel("gps", fs=100)
        N_base = 100  # 1 second at 100 Hz
        N_slow = 10   # 10 Hz sensor
        z_slow = 0.1 * np.random.randn(N_slow, 1)

        # make_multirate_input builds the NaN-padded base-rate measurement matrix.
        zk = make_multirate_input([(z_slow, 10)], base_fs=100)
        self.assertEqual(zk.shape, (N_base, 1))
        # ~90% of rows should be NaN (prediction-only steps)
        self.assertGreater(np.sum(np.isnan(zk[:, 0])), 80)

        rec_f = LinearKalmanFilter(m).forward(zk, record=True)
        self.assertEqual(rec_f.xk.shape[0], N_base)
        _no_nan_inf(rec_f.xk, "multirate xk")


# ===========================================================================
# Engine-level test
# ===========================================================================
class TestKalmanFilter(unittest.TestCase):
    """Generic filter-engine behavior (reset, RTS smoother) — uses OscillatorModel
    only as a convenient linear model; the oscillator-specific contract is tested
    in test_oscillators.py."""

    def test_reset_clears_logs(self):
        fs, N = 256, 256
        strain = np.cos(2 * np.pi * 10 * np.arange(N) / fs)
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs)
        kf = LinearKalmanFilter(m)
        kf.forward(strain.reshape(-1, 1))
        self.assertGreater(len(kf.posterior), 0)
        kf.reset()
        self.assertEqual(len(kf.posterior), 0)
        self.assertEqual(len(kf.transition), 0)

    def test_rts_smoother_no_crash(self):
        fs, N = 256, 256
        strain = np.cos(2 * np.pi * 10 * np.arange(N) / fs) + 0.1 * np.random.randn(N)
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs)
        rec_smooth = m.filter(strain, smooth=True)
        self.assertLess(rec_smooth.xk.shape[0], N)   # decimated rate
        _no_nan_inf(rec_smooth.xk, "RTS smoother xk")


if __name__ == "__main__":
    unittest.main()

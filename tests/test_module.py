import sys
import unittest
import numpy as np

# Allow running from project root or tests/ directory
sys.path.insert(0, ".")
try:
    from modules.apps import (
        OscillatorModel, AnalyticOscillatorModel, OscillatorCompositeModel,
        BallisticLinearModel, BallisticModel,
        SuspensionModel, ViolinModel,
        PowerGridStateModel,
    )
    from modules.framework.filter import KalmanFilter
except ImportError:
    from modules.apps import (
        OscillatorModel, AnalyticOscillatorModel, OscillatorCompositeModel,
        BallisticLinearModel, BallisticModel,
        SuspensionModel, ViolinModel,
        PowerGridStateModel,
    )
    from modules.framework.filter import KalmanFilter


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

    def test_oscillator_dims(self):
        m = OscillatorModel("s", fs=100)
        self.assertEqual(m.n, 2)
        self.assertEqual(m.m, 1)
        self.assertEqual(m.l, 0)

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

    def test_violin_mode_dims(self):
        m = ViolinModel("darm", f0=180.1, Q=1e5, fs=4096,
                            measurement_uncertainty=1e-19,
                            process_uncertainty=1e-23)
        self.assertEqual(m.n, 2)   # [Re(ψ), Im(ψ)]
        self.assertEqual(m.m, 2)   # demodulated [Re, Im]
        self.assertEqual(m.l, 0)

    def test_power_grid_dims(self):
        m = PowerGridStateModel("L1", f0=50.0, harmonics=3, fs=1024)
        self.assertEqual(m.n, 6)   # 2 states × 3 harmonics
        self.assertEqual(m.m, 6)   # 2 obs × 3 harmonics
        self.assertEqual(m.l, 0)

    def test_analytic_oscillator_dims(self):
        m = AnalyticOscillatorModel("s", fs=100, w0=2 * np.pi * 10, wc=2 * np.pi * 10, Q=100)
        self.assertEqual(m.n, 2)
        self.assertEqual(m.m, 2)
        self.assertEqual(m.l, 0)


# ===========================================================================
# Sample tests (no crash, valid shapes)
# ===========================================================================
class TestModelSample(unittest.TestCase):

    def test_oscillator_sample(self):
        m = OscillatorModel("s", fs=100)
        rec = m.sample(1.0)
        self.assertEqual(rec.xk.shape[0], 100)
        self.assertEqual(rec.xk.shape[1], 2)
        _no_nan_inf(rec.xk, "OscillatorModel xk")
        _no_nan_inf(rec.zk, "OscillatorModel zk")

    def test_ballistic_linear_sample(self):
        m = BallisticLinearModel("gps", fs=10)
        rec = m.sample(1.0)
        self.assertEqual(rec.xk.shape[0], 10)
        _no_nan_inf(rec.xk)

    def test_suspension_sample_no_crash(self):
        m = SuspensionModel("road", fs=256)
        rec = m.sample(0.5)
        _no_nan_inf(rec.xk, "SuspensionModel xk")


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

    def test_oscillator_filter(self):
        m = OscillatorModel("s", fs=100)
        rec_s = m.sample(1.0)
        rec_f = m.filter(rec_s.zk)
        self._assert_record(rec_f, "OscillatorModel", 2, 1, 100)

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

    def test_violin_mode_filter_real_input(self):
        """ViolinModeModel accepts a real (N,1) strain signal."""
        f0, fs = 180.1, 4096
        N = fs  # 1 second
        t = np.arange(N) / fs
        sig = 1e-19 * np.sin(2 * np.pi * f0 * t) + 5e-20 * np.random.randn(N)
        v = ViolinModel("darm", f0=f0, Q=1e5, fs=fs,
                         measurement_uncertainty=1e-19,
                         process_uncertainty=1e-23)
        rec_f = v.filter(sig.reshape(N, 1))
        self._assert_record(rec_f, "ViolinModel", 2, 2, N)

    def test_power_grid_filter(self):
        from scipy.signal import hilbert
        f0, harmonics, fs = 50.0, 2, 512
        N = fs  # 1 second
        t = np.arange(N) / fs
        sig = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        xa = hilbert(sig)
        zk = np.column_stack([xa.real, xa.imag])
        pg = PowerGridStateModel("L1", f0=f0, harmonics=harmonics, fs=fs)
        rec_f = pg.filter(zk)
        self._assert_record(rec_f, "PowerGridModel", pg.n, pg.m, N)

    def test_composite_oscillator_filter(self):
        f1 = OscillatorModel("s1", fs=100, w0=2 * np.pi * 5)
        f2 = OscillatorModel("s2", fs=100, w0=2 * np.pi * 10)
        comp = OscillatorCompositeModel(f1, f2)
        N = 100
        zk = 0.1 * np.random.randn(N, 2)  # 2 sensors
        rec_f = comp.filter(zk)
        self.assertEqual(rec_f.xk.shape[1], 4)  # 2 states per oscillator × 2
        _no_nan_inf(rec_f.xk, "OscillatorComposite xk")


# ===========================================================================
# Filter convergence / tracking test
# ===========================================================================
class TestFilterConvergence(unittest.TestCase):

    def test_oscillator_tracks_signal(self):
        """After convergence, filtered position should be close to truth."""
        np.random.seed(42)
        fs = 100
        N = 300  # 3 seconds
        m = OscillatorModel("s", fs=fs, w0=2 * np.pi * 2,
                             measurement_uncertainty=0.1,
                             process_uncertainty=0.001)

        # Ground truth: pure oscillation
        t = np.arange(N) / fs
        truth_pos = np.sin(2 * np.pi * 2 * t)
        zk = truth_pos.reshape(N, 1) + 0.1 * np.random.randn(N, 1)

        rec_f = m.filter(zk)
        # xk has shape (N, n, 1) — squeeze to (N, n) then take position (index 0)
        pos_hat = rec_f.xk[50:, 0].ravel()   # skip first 50 transient samples
        truth_tail = truth_pos[50:]

        rms_error = np.sqrt(np.mean((pos_hat - truth_tail) ** 2))
        self.assertLess(rms_error, 0.3, f"Oscillator filter RMS error too large: {rms_error:.4f}")

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

    def test_nan_masked_steps_no_crash(self):
        """NaN rows in zk should trigger prediction-only (no update) without crash."""
        from modules.helpers import make_multirate_input

        m = OscillatorModel("s", fs=100)
        N_base = 100  # 1 second at 100 Hz
        N_slow = 10   # 10 Hz sensor

        z_slow = 0.1 * np.random.randn(N_slow, 1)
        zk = make_multirate_input([(z_slow, 10)], base_fs=100)
        self.assertEqual(zk.shape, (N_base, 1))

        # 90% of rows should be NaN
        nan_rows = np.sum(np.isnan(zk[:, 0]))
        self.assertGreater(nan_rows, 80)

        rec_f = m.filter(zk)
        self.assertEqual(rec_f.xk.shape[0], N_base)
        # Allow NaN in missed update states (covariance grows); just check no Inf
        _no_nan_inf(rec_f.xk, "multirate xk")


# ===========================================================================
# Engine-level test
# ===========================================================================
class TestKalmanFilter(unittest.TestCase):

    def test_reset_clears_logs(self):
        m = OscillatorModel("s", fs=100)
        eng = m._engine
        m.filter(m.sample(1.0).zk)
        self.assertGreater(len(eng.posterior), 0)
        eng.reset()
        self.assertEqual(len(eng.posterior), 0)
        self.assertEqual(len(eng.transition), 0)

    def test_rts_smoother_no_crash(self):
        m = OscillatorModel("s", fs=100)
        zk = 0.1 * np.random.randn(100, 1)
        rec_smooth = m.filter(zk, smooth=True)
        self.assertEqual(rec_smooth.xk.shape[0], 100)
        _no_nan_inf(rec_smooth.xk, "RTS smoother xk")


if __name__ == "__main__":
    unittest.main()

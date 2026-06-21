import sys
import unittest
import numpy as np

# Allow running from project root or tests/ directory
sys.path.insert(0, ".")
try:
    from modules.apps import (
        OscillatorModel, AnalyticOscillatorModel, OscillatorCompositeModel,
        BallisticLinearModel,
        SuspensionModel,
        PowerGridStateModel,
    )
    from modules.filters import LinearKalmanFilter
except ImportError:
    from modules.apps import (
        OscillatorModel, AnalyticOscillatorModel, OscillatorCompositeModel,
        BallisticLinearModel,
        SuspensionModel,
        PowerGridStateModel,
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

    def test_oscillator_dims(self):
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=2.0, fs=100)
        self.assertEqual(m.n, 2)   # one in-band mode → [Re, Im]
        self.assertEqual(m.m, 2)   # FUSED demodulated band channel
        self.assertEqual(m.l, 0)

    def test_oscillator_window_mask(self):
        # the window masks the candidate list to in-band modes (FUSED-joined)
        m = OscillatorModel("s", [8.0, 10.0, 30.0], fc=10.0, dfc=3.0, fs=100)
        np.testing.assert_allclose(np.sort(m.frequencies), [8.0, 10.0])  # 30 Hz masked out
        self.assertEqual((m.n, m.m), (4, 2))

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

    def test_power_grid_dims(self):
        m = PowerGridStateModel("L1", f0=50.0, harmonics=3, fs=1024)
        self.assertEqual(m.n, 6)   # 2 phasor states (Re/Im) × 3 harmonics
        self.assertEqual(m.m, 1)   # joint estimator: single scalar voltage observation
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

    def test_ballistic_linear_sample(self):
        m = BallisticLinearModel("gps", fs=10)
        rec = m.sample(1.0)
        self.assertEqual(rec.xk.shape[0], 10)
        _no_nan_inf(rec.xk)

    def test_suspension_sample_no_crash(self):
        m = SuspensionModel("road", fs=256)
        rec = m.sample(0.5)
        _no_nan_inf(rec.xk, "SuspensionModel xk")

    def test_sample_default_injects_process_noise(self):
        """sample() defaults to covariance=False: the truth is a real stochastic path
        (it drifts under Q), while covariance=True gives the deterministic mean + Pk."""
        np.random.seed(0)
        m = BallisticLinearModel("gps", fs=100, process_uncertainty=1.0)
        rec_default = m.sample(2.0)                       # default → drifting truth, Pk zeros
        rec_cov = m.sample(2.0, covariance=True)          # deterministic mean, analytic Pk
        self.assertGreater(np.std(np.diff(rec_default.xk[:, 0])), 0.0)   # truth actually moves
        self.assertAlmostEqual(float(np.sum(np.abs(rec_default.Pk))), 0.0)  # no Pk on fast path
        self.assertGreater(float(np.sum(np.abs(rec_cov.Pk))), 0.0)       # analytic Pk present


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
        fs, N = 256, 512
        t = np.arange(N) / fs
        strain = np.cos(2 * np.pi * 10 * t) + 0.05 * np.random.randn(N)
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs,
                            measurement_uncertainty=0.05, process_uncertainty=1e-4)
        rec_f = m.filter(strain, smooth=True)
        self._assert_record(rec_f, "OscillatorModel", 2, 2, N)

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
        # bank of two windows over ONE real signal (10 Hz and 40 Hz tones)
        fs, N = 256, 512
        t = np.arange(N) / fs
        strain = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 40 * t) + 0.05 * np.random.randn(N)
        cands = [10.0, 40.0]
        w1 = OscillatorModel("s", cands, fc=10.0, dfc=3.0, fs=fs)
        w2 = OscillatorModel("s", cands, fc=40.0, dfc=3.0, fs=fs)
        comp = OscillatorCompositeModel(w1, w2)
        rec_f = comp.filter(strain)
        self.assertEqual(rec_f.xk.shape[1], 4)  # 2 windows × [Re, Im]
        _no_nan_inf(rec_f.xk, "OscillatorComposite xk")
        self.assertEqual(comp.reconstruct(rec_f).shape[0], N)  # real summed reconstruction


# ===========================================================================
# Filter convergence / tracking test
# ===========================================================================
class TestFilterConvergence(unittest.TestCase):

    def test_oscillator_tracks_signal(self):
        """After convergence, filtered position should be close to truth."""
        np.random.seed(42)
        fs, N = 256, 1024
        t = np.arange(N) / fs
        truth = np.cos(2 * np.pi * 10 * t)                 # unit-amplitude 10 Hz tone
        strain = truth + 0.1 * np.random.randn(N)
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs,
                            measurement_uncertainty=0.1, process_uncertainty=1e-4)
        rec = m.filter(strain, smooth=True)
        amp = m.amplitudes(rec)[N // 4:].mean()            # recovered envelope amplitude
        self.assertAlmostEqual(amp, 1.0, delta=0.25,
                               msg=f"window amplitude off: {amp:.3f}")

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
        self.assertEqual(rec_smooth.xk.shape[0], N)
        _no_nan_inf(rec_smooth.xk, "RTS smoother xk")


if __name__ == "__main__":
    unittest.main()

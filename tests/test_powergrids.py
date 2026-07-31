import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.apps.resonators.powergrids import PowerGridStateModel
from modules.apps.resonators.oscillators import PhasorOscillatorModel


def _no_nan_inf(arr, label="array"):
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


# ===========================================================================
# PhasorOscillatorModel — generic lab-frame tracker (base class)
# ===========================================================================
class TestPhasorOscillatorModel(unittest.TestCase):

    def test_dims(self):
        m = PhasorOscillatorModel("s", [60.0, 173.2], fs=1024)
        self.assertEqual((m.n, m.m, m.l), (4, 1, 0))   # 2 phasors -> 4 states, 1 scalar obs

    def test_requires_at_least_one_frequency(self):
        with self.assertRaises(ValueError):
            PhasorOscillatorModel("s", [], fs=1024)

    def test_recovers_two_unrelated_tones(self):
        """No demodulation, no shared structure: two unrelated real tones recovered
        jointly via the time-varying H(t), F=I lab-frame formulation."""
        np.random.seed(0)
        fs, T = 1024.0, 4.0
        t = np.arange(int(T * fs)) / fs
        f_a, f_b = 60.0, 173.2
        A_a, A_b = 1.0, 0.5
        sig = (A_a * np.cos(2 * np.pi * f_a * t + 0.4)
               + A_b * np.cos(2 * np.pi * f_b * t - 1.1)
               + 0.05 * np.random.randn(len(t)))
        m = PhasorOscillatorModel("demo", [f_a, f_b], fs=fs,
                                  measurement_uncertainty=0.05,
                                  process_uncertainty=1e-4, p0_std=10.0)
        rec = m.filter(sig, smooth=True)
        amp = m.amplitudes(rec)[len(t) // 2:].mean(axis=0)
        self.assertAlmostEqual(amp[0], A_a, delta=0.15)
        self.assertAlmostEqual(amp[1], A_b, delta=0.15)

    def test_reconstruct_shape(self):
        fs, N = 1024.0, 512
        t = np.arange(N) / fs
        sig = np.cos(2 * np.pi * 60 * t)
        m = PhasorOscillatorModel("demo", [60.0], fs=fs)
        rec = m.filter(sig, smooth=True)
        recon = m.reconstruct(rec)
        self.assertEqual(recon.shape, (N,))


# ===========================================================================
# PowerGridStateModel — mains harmonic specialization
# ===========================================================================
class TestPowerGridStateModel(unittest.TestCase):

    def test_dims(self):
        m = PowerGridStateModel("L1", f0=50.0, harmonics=3, fs=1024)
        self.assertEqual(m.n, 6)   # 2 phasor states (Re/Im) x 3 harmonics
        self.assertEqual(m.m, 1)   # joint estimator: single scalar voltage observation
        self.assertEqual(m.l, 0)

    def test_filter_runs_with_analytic_input(self):
        from scipy.signal import hilbert
        f0, harmonics, fs = 50.0, 2, 512
        N = fs
        t = np.arange(N) / fs
        sig = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
        xa = hilbert(sig)
        zk = np.column_stack([xa.real, xa.imag])
        m = PowerGridStateModel("L1", f0=f0, harmonics=harmonics, fs=fs)
        rec = m.filter(zk)
        self.assertEqual(rec.xk.shape, (N, m.n))
        self.assertEqual(rec.zk.shape, (N, m.m))
        _no_nan_inf(rec.xk, "PowerGridModel xk")

    def test_odd_harmonic_dominance_even_settle_near_zero(self):
        """For a half-wave-symmetric (ideal) grid, joint estimation should leave
        the even harmonics (H2, H4) near zero while H1/H3/H5 converge to truth."""
        np.random.seed(0)
        f0, fs, T = 50.0, 1024.0, 4.0
        t = np.arange(int(T * fs)) / fs
        A1, A3, A5 = 230.0, 15.0, 5.0
        v_clean = (A1 * np.cos(2 * np.pi * f0 * t)
                  + A3 * np.cos(2 * np.pi * 3 * f0 * t + np.pi / 4)
                  + A5 * np.cos(2 * np.pi * 5 * f0 * t + np.pi / 3))
        noise_std = 10.0
        v_noisy = v_clean + np.random.normal(0, noise_std, len(t))

        grid = PowerGridStateModel("L1", f0=f0, harmonics=5, fs=fs,
                                   measurement_uncertainty=noise_std,
                                   process_uncertainty=0.01)
        grid.initial_state([[A1, 0], [0, 0], [A3, 0], [0, 0], [A5, 0]])
        rec = grid.filter(v_noisy, smooth=True, reset=True)
        amp, _ = grid.harmonic_envelope(rec)
        steady = amp[-len(t) // 10:].mean(axis=0)

        self.assertAlmostEqual(steady[0], A1, delta=15.0)   # H1
        self.assertAlmostEqual(steady[2], A3, delta=5.0)    # H3
        self.assertAlmostEqual(steady[4], A5, delta=3.0)    # H5
        self.assertLess(steady[1], 10.0)                     # H2 ~ 0
        self.assertLess(steady[3], 10.0)                     # H4 ~ 0

    def test_h2_rises_under_half_wave_rectifier_fault(self):
        """Injecting a 2nd-harmonic component (simulated rectifier fault) must be
        picked up by the H2 channel, which otherwise stays near zero."""
        np.random.seed(7)
        f0, fs, T = 50.0, 1024.0, 3.0
        t = np.arange(int(T * fs)) / fs
        A1 = 230.0
        V2_fault = 20.0
        v = A1 * np.cos(2 * np.pi * f0 * t) + np.random.normal(0, 10.0, len(t))
        v += V2_fault * np.cos(2 * np.pi * 2 * f0 * t)   # fault present for the whole signal

        grid = PowerGridStateModel("L1", f0=f0, harmonics=3, fs=fs,
                                   measurement_uncertainty=10.0,
                                   process_uncertainty=0.1)
        grid.initial_state([[A1, 0], [0, 0], [0, 0]])
        rec = grid.filter(v, smooth=True, reset=True)
        amp, _ = grid.harmonic_envelope(rec)
        steady_h2 = amp[-len(t) // 10:, 1].mean()
        self.assertAlmostEqual(steady_h2, V2_fault, delta=8.0)

    def test_thd_close_to_true_value(self):
        np.random.seed(1)
        f0, fs, T = 50.0, 1024.0, 4.0
        t = np.arange(int(T * fs)) / fs
        A1, A3, A5 = 230.0, 15.0, 5.0
        v = (A1 * np.cos(2 * np.pi * f0 * t)
            + A3 * np.cos(2 * np.pi * 3 * f0 * t)
            + A5 * np.cos(2 * np.pi * 5 * f0 * t)
            + np.random.normal(0, 10.0, len(t)))

        grid = PowerGridStateModel("L1", f0=f0, harmonics=5, fs=fs,
                                   measurement_uncertainty=10.0,
                                   process_uncertainty=0.01)
        grid.initial_state([[A1, 0], [0, 0], [A3, 0], [0, 0], [A5, 0]])
        rec = grid.filter(v, smooth=True, reset=True)
        amp, _ = grid.harmonic_envelope(rec)
        steady = amp[-len(t) // 10:].mean(axis=0)
        thd_est = 100 * np.hypot(steady[2], steady[4]) / steady[0]
        thd_true = 100 * np.hypot(A3, A5) / A1
        self.assertAlmostEqual(thd_est, thd_true, delta=2.0)


if __name__ == "__main__":
    unittest.main()

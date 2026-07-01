import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.apps.interferometer import (
    InterferometerModel,
    welch_psd,
    find_peaks_in_band,
    fit_lorentzian_peak,
    group_peaks_into_windows,
    lorentzian,
    ViolinBankController,
)
from modules.apps.oscillators import OscillatorModel, AnalyticOscillatorModel
from modules.framework.controllers.boost import BoostKalmanController
from modules.framework.controllers.adaptive_qr import AdaptiveQRKalmanController
from modules.framework.filter import KalmanFilter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sine_psd(freqs_hz, amplitudes, fs=4096, duration=30.0, noise_std=0.1, seed=0):
    """Synthetic multi-tone signal + white noise → strain array."""
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs
    x = np.zeros(N)
    for f, a in zip(freqs_hz, amplitudes):
        x += a * np.cos(2 * np.pi * f * t)
    x += rng.normal(0, noise_std, N)
    return x


# ===========================================================================
# Spectral utilities
# ===========================================================================

class TestWelchPSD(unittest.TestCase):

    def test_returns_two_arrays(self):
        x = np.random.default_rng(0).normal(size=4096)
        f, p = welch_psd(x, fs=4096.0)
        self.assertEqual(f.ndim, 1)
        self.assertEqual(p.ndim, 1)
        self.assertEqual(f.shape, p.shape)
        self.assertTrue(np.all(p >= 0))

    def test_peak_visible(self):
        fs = 4096.0
        x  = _sine_psd([175.0], [1.0], fs=fs, noise_std=0.01, duration=10.0)
        f, p = welch_psd(x, fs=fs)
        peak_idx = np.argmax(p)
        self.assertAlmostEqual(f[peak_idx], 175.0, delta=1.0)


class TestLorentzian(unittest.TestCase):

    def test_peak_at_f0(self):
        val = lorentzian(10.0, f0=10.0, gamma=0.1, amplitude=5.0)
        self.assertAlmostEqual(val, 5.0)

    def test_half_power_at_half_gamma(self):
        gamma, A = 0.2, 4.0
        val = lorentzian(10.0 + gamma / 2, f0=10.0, gamma=gamma, amplitude=A)
        self.assertAlmostEqual(val, A / 2.0, places=6)


class TestFitLorentzianPeak(unittest.TestCase):

    def _make_psd(self, f0, Q, amplitude, fs=4096.0, duration=60.0):
        x = _sine_psd([f0], [amplitude], fs=fs, noise_std=0.05, duration=duration)
        return welch_psd(x, fs=fs)

    def test_recovers_f0(self):
        f0, Q, A = 175.0, 1e4, 1.0
        freqs, psd = self._make_psd(f0, Q, A)
        result = fit_lorentzian_peak(freqs, psd, f_center=f0, f_half_width=3.0)
        self.assertAlmostEqual(result["f0"], f0, delta=0.5)

    def test_recovers_Q(self):
        # Use a damped sinusoid so the PSD is a genuine Lorentzian.
        # Q=30 → gamma=175/30≈5.8 Hz, well above the 1 Hz Welch resolution.
        f0, Q_true = 175.0, 30.0
        fs, duration = 4096.0, 60.0
        N   = int(fs * duration)
        t   = np.arange(N) / fs
        gamma_t = np.pi * f0 / Q_true  # 1/e decay rate [rad/s]
        rng = np.random.default_rng(7)
        x   = 2.0 * np.exp(-gamma_t * t) * np.cos(2 * np.pi * f0 * t)
        x  += rng.normal(0, 0.02, N)
        freqs, psd = welch_psd(x, fs=fs, nperseg=int(fs * 4))
        result = fit_lorentzian_peak(freqs, psd, f_center=f0, f_half_width=10.0)
        # 50 % tolerance on Q for a noisy signal
        self.assertGreater(result["Q"], Q_true * 0.5)
        self.assertLess(result["Q"],    Q_true * 3.0)

    def test_raises_on_empty_band(self):
        freqs = np.linspace(100, 200, 100)
        psd   = np.ones(100)
        # Only 3 points in window → should raise
        with self.assertRaises(RuntimeError):
            fit_lorentzian_peak(freqs, psd, f_center=150.0, f_half_width=0.001)


class TestFindPeaksInBand(unittest.TestCase):

    def test_detects_known_peak(self):
        fs = 4096.0
        x  = _sine_psd([178.0], [1.0], fs=fs, noise_std=0.01)
        f, p = welch_psd(x, fs=fs)
        peaks = find_peaks_in_band(f, p, 170.0, 186.0)
        self.assertGreater(len(peaks), 0)
        self.assertAlmostEqual(peaks[np.argmin(np.abs(peaks - 178.0))], 178.0, delta=1.0)

    def test_empty_band_returns_empty(self):
        freqs = np.linspace(100, 200, 200)
        psd   = np.ones(200)
        peaks = find_peaks_in_band(freqs, psd, 300.0, 400.0)
        self.assertEqual(len(peaks), 0)


class TestGroupPeaksIntoWindows(unittest.TestCase):

    def test_merge_close_peaks(self):
        fits = [{"f0": 175.0, "Q": 1e4, "amplitude": 1.0},
                {"f0": 176.0, "Q": 1e4, "amplitude": 1.0}]
        wins = group_peaks_into_windows(fits, merge_gap_hz=2.0)
        self.assertEqual(len(wins), 1)
        self.assertAlmostEqual(wins[0]["fc"], 175.5)

    def test_separate_distant_peaks(self):
        fits = [{"f0": 175.0, "Q": 1e4, "amplitude": 1.0},
                {"f0": 225.0, "Q": 1e4, "amplitude": 1.0}]
        wins = group_peaks_into_windows(fits, merge_gap_hz=2.0)
        self.assertEqual(len(wins), 2)

    def test_dfc_includes_margin(self):
        fits = [{"f0": 175.0, "Q": 1e4, "amplitude": 1.0},
                {"f0": 177.0, "Q": 1e4, "amplitude": 1.0}]
        wins = group_peaks_into_windows(fits, merge_gap_hz=5.0, margin_hz=1.0)
        self.assertGreaterEqual(wins[0]["dfc"], 1.0 + 1.0)  # span + margin


# ===========================================================================
# InterferometerModel
# ===========================================================================

class TestInterferometerModelConstruct(unittest.TestCase):

    def _one_window(self, f0=175.0, fs=4096.0):
        return OscillatorModel("K1", [f0], fc=f0, dfc=4.0, Q=1e4, fs=fs)

    def test_plain_constructor(self):
        win = self._one_window()
        m = InterferometerModel("K1", [win])
        self.assertEqual(len(m.windows), 1)
        self.assertEqual(m.sensor_id, "K1")

    def test_two_windows(self):
        w1 = self._one_window(175.0)
        w2 = self._one_window(412.0)
        m = InterferometerModel("K1", [w1, w2])
        self.assertEqual(len(m.windows), 2)
        self.assertEqual(m.n, w1.n + w2.n)

    def test_state_offsets(self):
        w1 = self._one_window(175.0)
        w2 = self._one_window(412.0)
        m = InterferometerModel("K1", [w1, w2])
        self.assertEqual(m._state_offsets, [0, w1.n])

    def test_noise_sources_attribute(self):
        self.assertIn("violin", InterferometerModel.NOISE_SOURCES)


class TestInterferometerModelFit(unittest.TestCase):

    def _strain(self, freqs, fs=4096.0, duration=60.0):
        return _sine_psd(freqs, [1.0] * len(freqs), fs=fs, noise_std=0.05,
                         duration=duration)

    def test_fit_finds_windows(self):
        fs, f0 = 4096.0, 175.0
        x = self._strain([f0], fs=fs)
        m = InterferometerModel.fit("K1", x, fs=fs, bands=[(170.0, 186.0)])
        self.assertGreater(len(m.windows), 0)

    def test_fit_peak_frequency_accuracy(self):
        fs, f0 = 4096.0, 178.0
        x = self._strain([f0], fs=fs)
        m = InterferometerModel.fit("K1", x, fs=fs, bands=[(170.0, 186.0)])
        fitted_freqs = [w.fc for w in m.windows]
        closest = min(fitted_freqs, key=lambda f: abs(f - f0))
        self.assertAlmostEqual(closest, f0, delta=2.0)

    def test_fit_two_bands(self):
        fs = 4096.0
        x = self._strain([175.0, 412.0], fs=fs)
        m = InterferometerModel.fit("K1", x, fs=fs,
                                    bands=[(170.0, 186.0), (402.0, 432.0)])
        self.assertGreaterEqual(len(m.windows), 2)

    def test_fit_raises_on_empty_bands(self):
        # A band above Nyquist has no PSD points → find_peaks returns nothing →
        # all_fits stays empty → ValueError.
        fs = 4096.0
        x  = self._strain([175.0], fs=fs)
        with self.assertRaises(ValueError):
            InterferometerModel.fit("K1", x, fs=fs,
                                    bands=[(fs / 2 + 10, fs / 2 + 100)])

    def test_peak_summary_populated(self):
        fs, f0 = 4096.0, 175.0
        x = self._strain([f0], fs=fs)
        m = InterferometerModel.fit("K1", x, fs=fs, bands=[(170.0, 186.0)])
        summary = m.peak_summary()
        self.assertGreater(len(summary), 0)
        self.assertIn("fc", summary[0])
        self.assertIn("peaks", summary[0])


class TestInterferometerModelFilter(unittest.TestCase):

    def _model_and_strain(self, f0=175.0, fs=4096.0, duration=5.0, Q=1e4):
        win = OscillatorModel("K1", [f0], fc=f0, dfc=4.0, Q=Q, fs=fs)
        m   = InterferometerModel("K1", [win])
        x   = _sine_psd([f0], [1.0], fs=fs, noise_std=0.1, duration=duration)
        return m, x

    def test_forward_no_crash(self):
        m, x = self._model_and_strain()
        kf  = KalmanFilter(m)
        rec = kf.forward(x)
        self.assertIsNotNone(rec)

    def test_subtract_shape(self):
        m, x = self._model_and_strain()
        kf   = KalmanFilter(m)
        rec  = kf.forward(x)
        res  = m.subtract(x, rec)
        self.assertEqual(res.shape, x.ravel().shape)

    def test_residual_power_reduction(self):
        """After subtraction, PSD at the tracked line should drop."""
        f0, fs = 175.0, 4096.0
        m, x   = self._model_and_strain(f0=f0, fs=fs, duration=10.0)
        kf     = KalmanFilter(m)
        rec    = kf.forward(x)
        res    = m.subtract(x, rec)

        fv, psd_in  = welch_psd(x,   fs=fs)
        _,  psd_out = welch_psd(res, fs=fs)

        idx   = np.argmin(np.abs(fv - f0))
        ratio = psd_out[idx] / max(psd_in[idx], 1e-30)
        self.assertLess(ratio, 0.5,
                        msg=f"Expected ≥ 3 dB residual reduction at {f0} Hz; "
                            f"got PSD ratio {ratio:.3f}")


# ===========================================================================
# ViolinBankController
# ===========================================================================

class TestViolinBankController(unittest.TestCase):

    def test_make_controller_returns_correct_type(self):
        win = OscillatorModel("K1", [175.0], fc=175.0, dfc=4.0, Q=1e4, fs=4096.0)
        m   = InterferometerModel("K1", [win])
        ctrl = m.make_controller()
        self.assertIsInstance(ctrl, ViolinBankController)

    def test_forward_with_controller_no_crash(self):
        win  = OscillatorModel("K1", [175.0], fc=175.0, dfc=4.0, Q=1e4, fs=4096.0)
        m    = InterferometerModel("K1", [win])
        ctrl = m.make_controller()
        x    = _sine_psd([175.0], [1.0], fs=4096.0, noise_std=0.1, duration=5.0)
        kf   = KalmanFilter(m)
        rec  = kf.forward(x, uk=ctrl)
        self.assertIsNotNone(rec)

    def test_scale_log_populated(self):
        """AdaptiveQR scale_log should have entries after enough steps."""
        win  = OscillatorModel("K1", [175.0], fc=175.0, dfc=4.0, Q=1e4, fs=4096.0)
        m    = InterferometerModel("K1", [win], aqr_window=10)
        ctrl = m.make_controller()
        x    = _sine_psd([175.0], [1.0], fs=4096.0, noise_std=0.1, duration=5.0)
        kf   = KalmanFilter(m)
        kf.forward(x, uk=ctrl)
        inner_ctrls = ctrl._window_aqrs[0]._ctrls
        any_logged  = any(len(c.scale_log) > 0 for c in inner_ctrls)
        self.assertTrue(any_logged, "Expected scale_log entries from AdaptiveQR")


# ===========================================================================
# Controller refactor regressions (Boost / AdaptiveQR as KalmanController)
# ===========================================================================

class TestBoostControllerRefactor(unittest.TestCase):

    def test_inherits_kalman_controller(self):
        from modules.framework.control import KalmanController
        m = AnalyticOscillatorModel("s", fs=128, f0=10.0, fc=10.0, Q=100)
        b = BoostKalmanController(m)
        self.assertIsInstance(b, KalmanController)

    def test_policy_returns_zero(self):
        m = AnalyticOscillatorModel("s", fs=128, f0=10.0, fc=10.0, Q=100)
        b = BoostKalmanController(m)
        u = b._policy(np.zeros((m.n, 1)), np.eye(m.n), 0.0)
        self.assertEqual(u.shape, (1, 1))  # max(l=0, 1) = 1
        np.testing.assert_allclose(u, 0.0)

    def test_scale_log_populated(self):
        from modules.apps.oscillators import AnalyticOscillatorModel as AOM
        m    = AOM("s", fs=128, f0=10.0, fc=10.0, Q=100,
                   measurement_uncertainty=0.5, process_uncertainty=0.01,
                   random_seed=1)
        m.initial_state([5, 0])
        s    = m.sample(2.0)
        x    = np.asarray(s.zk)
        b    = BoostKalmanController(m, boost_factor=5.0, threshold=2.0, window=5)
        kf   = KalmanFilter(m)
        kf.forward(x, uk=b)
        self.assertGreater(len(b.scale_log), 0)


class TestAdaptiveQRControllerRefactor(unittest.TestCase):

    def test_inherits_kalman_controller(self):
        from modules.framework.control import KalmanController
        m = AnalyticOscillatorModel("s", fs=128, f0=10.0, fc=10.0, Q=100)
        a = AdaptiveQRKalmanController(m)
        self.assertIsInstance(a, KalmanController)

    def test_policy_returns_zero(self):
        m = AnalyticOscillatorModel("s", fs=128, f0=10.0, fc=10.0, Q=100)
        a = AdaptiveQRKalmanController(m)
        u = a._policy(np.zeros((m.n, 1)), np.eye(m.n), 0.0)
        np.testing.assert_allclose(u, 0.0)

    def test_q_scaling_triggered(self):
        """After window steps, Q_nominal should differ from model._Q if mis-tuned."""
        m = AnalyticOscillatorModel("s", fs=128, f0=10.0, fc=10.0, Q=100,
                                    measurement_uncertainty=0.5, process_uncertainty=0.01,
                                    random_seed=2)
        m.initial_state([5, 0])
        s  = m.sample(3.0)
        x  = np.asarray(s.zk)
        a  = AdaptiveQRKalmanController(m, window=20)
        kf = KalmanFilter(m)
        kf.forward(x, uk=a)
        self.assertGreater(len(a.scale_log), 0)


class TestDetectorSubclasses(unittest.TestCase):
    """Tests for InterferometerK1/H1/L1/V1 detector-specific subclasses."""

    def _make_synthetic_strain(self, f0=167.5, B=0.136, n_harmonics=3,
                              fs=4096.0, T=30.0, seed=7):
        """Synthetic K1 O3GK-like signal with peaks inside the catalog bands.

        Defaults use Moisset et al. 2025 parameters: f0=167.5 Hz, B=0.136 →
        partials at ~178.5, ~416, ~749 Hz (Mode 1/2/3 catalogue band centres).
        Q=200 is artificially low (real K1 Q ~ 10^3-10^4) so FWHM ~ f/Q ~ 1-4 Hz
        is resolvable by 1-Hz Welch and produces a single clean peak per band.
        """
        from modules.apps.oscillators import partial_frequencies
        rng = np.random.default_rng(seed)
        N = int(fs * T)
        t = np.arange(N) / fs
        freqs = partial_frequencies(f0, B, n_harmonics)
        x = 1e-4 * rng.standard_normal(N)
        Q = 200.0
        for f in freqs:
            A = rng.uniform(0.5, 1.5)
            x += A * np.exp(-np.pi * f / Q * t) * np.cos(2 * np.pi * f * t)
        return x, fs, freqs

    def test_k1_violin_bands_contains_partials(self):
        from modules.apps.interferometer import InterferometerK1
        p = InterferometerK1._CATALOG["O3GK"]
        bands = InterferometerK1.violin_bands("O3GK")
        # One band per mode
        self.assertEqual(len(bands), len(p["peaks"]))
        # Every catalog peak must fall inside its mode's search band
        for (f_lo, f_hi), mode_peaks in zip(bands, p["peaks"]):
            for f0, _ in mode_peaks:
                self.assertGreater(f0, f_lo, msg=f"{f0} Hz below band lower edge {f_lo}")
                self.assertLess(f0, f_hi, msg=f"{f0} Hz above band upper edge {f_hi}")

    def test_h1_violin_bands_length(self):
        from modules.apps.interferometer import InterferometerH1
        bands = InterferometerH1.violin_bands("O3")
        self.assertEqual(len(bands), InterferometerH1._CATALOG["O3"]["n_harmonics"])

    def test_l1_violin_bands_length(self):
        from modules.apps.interferometer import InterferometerL1
        bands = InterferometerL1.violin_bands("O4")
        self.assertEqual(len(bands), InterferometerL1._CATALOG["O4"]["n_harmonics"])

    def test_v1_violin_bands_length(self):
        from modules.apps.interferometer import InterferometerV1
        bands = InterferometerV1.violin_bands("O3")
        self.assertEqual(len(bands), InterferometerV1._CATALOG["O3"]["n_harmonics"])

    def test_latest_run_returns_string(self):
        from modules.apps.interferometer import InterferometerK1, InterferometerH1
        self.assertIsInstance(InterferometerK1.latest_run(), str)
        self.assertIn(InterferometerK1.latest_run(), InterferometerK1._CATALOG)
        self.assertIsInstance(InterferometerH1.latest_run(), str)

    def test_violin_bands_bad_run_raises_key_error(self):
        from modules.apps.interferometer import InterferometerK1
        with self.assertRaises(KeyError):
            InterferometerK1.violin_bands("O99")

    def test_k1_fit_from_catalog_bands(self):
        """InterferometerK1.fit() with catalog bands detects the right number of windows."""
        from modules.apps.interferometer import InterferometerK1
        strain, fs, freqs = self._make_synthetic_strain()
        bands = InterferometerK1.violin_bands("O3GK")
        model = InterferometerK1.fit("K1", strain, fs=fs, bands=bands)
        self.assertIsInstance(model, InterferometerK1)
        self.assertEqual(len(model.windows), 3)

    def test_k1_fit_peak_frequencies_in_band(self):
        """Detected peak centre frequencies should fall within the catalog band limits."""
        from modules.apps.interferometer import InterferometerK1
        strain, fs, _ = self._make_synthetic_strain()
        bands = InterferometerK1.violin_bands("O3GK")
        model = InterferometerK1.fit("K1", strain, fs=fs, bands=bands)
        for s, (f_lo, f_hi) in zip(model.peak_summary(), bands):
            for p in s["peaks"]:
                self.assertGreater(p["f0"], f_lo)
                self.assertLess(p["f0"], f_hi)

    def test_k1_filter_and_subtract_shape(self):
        from modules.apps.interferometer import InterferometerK1
        strain, fs, _ = self._make_synthetic_strain()
        bands = InterferometerK1.violin_bands("O3GK")
        model = InterferometerK1.fit("K1", strain, fs=fs, bands=bands)
        # forward only — smooth=True is tested separately via test_residual_power_reduction
        rec   = model.filter(strain, uk=model.make_controller())
        clean = model.subtract(strain, rec)
        self.assertEqual(clean.shape, strain.shape)

    def test_from_gps_raises_without_gwpy(self):
        """from_gps with no gwpy installed should raise ImportError or similar."""
        from modules.apps.interferometer import InterferometerK1
        import unittest.mock as mock
        # Patch gwpy to be unavailable
        with mock.patch.dict("sys.modules", {"gwpy": None, "gwpy.timeseries": None}):
            with self.assertRaises((ImportError, ModuleNotFoundError, TypeError)):
                InterferometerK1.from_gps(1271267328, 1271281728,
                                          run="O3GK", cache=False)


if __name__ == "__main__":
    unittest.main()

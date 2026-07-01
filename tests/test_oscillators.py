import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.apps.oscillators import (
    OscillatorModel, AnalyticOscillatorModel, OscillatorBankModel, OscillatorCompositeModel,
    HarmonicOscillatorModel, InharmonicOscillatorModel,
    partial_frequencies, wire_fundamental, wire_inharmonicity,
)


def _no_nan_inf(arr, label="array"):
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


# ===========================================================================
# AnalyticOscillatorModel — the rotating-frame atom
# ===========================================================================
class TestAnalyticOscillatorModel(unittest.TestCase):

    def test_dims(self):
        m = AnalyticOscillatorModel("s", fs=100, f0=10.0, fc=10.0, Q=100)
        self.assertEqual((m.n, m.m, m.l), (2, 2, 0))

    def test_f0_fc_hz_params(self):
        """Preferred Hz-based parameters set the right internal angular frequencies."""
        m = AnalyticOscillatorModel("s", fs=100, f0=10.0, fc=8.0, Q=100)
        self.assertAlmostEqual(m["f0"], 10.0)
        self.assertAlmostEqual(m["fc"], 8.0)
        self.assertAlmostEqual(m["w0"], 2 * np.pi * 10.0)
        self.assertAlmostEqual(m["wc"], 2 * np.pi * 8.0)

    def test_w0_wc_radps_backward_compat(self):
        """Legacy w0/wc (rad/s) kwargs still work and map to the same f0/fc."""
        m = AnalyticOscillatorModel("s", fs=100, w0=2 * np.pi * 10, wc=2 * np.pi * 10, Q=100)
        self.assertEqual((m.n, m.m, m.l), (2, 2, 0))
        self.assertAlmostEqual(m["f0"], 10.0)
        self.assertAlmostEqual(m["fc"], 10.0)

    def test_fc_defaults_to_f0_lock_in(self):
        """Omitting fc defaults the carrier to resonance (lock-in mode)."""
        m = AnalyticOscillatorModel("s", fs=100, f0=12.5, Q=100)
        self.assertAlmostEqual(m["fc"], 12.5)

    def test_sample_postprocess_roundtrip(self):
        """preprocess/postprocess are inverse rotations: sample().zk is analytic
        lab-frame data that can be fed straight back into filter()."""
        m = AnalyticOscillatorModel("s", fs=128, f0=10.0, fc=10.0, Q=1000,
                                    measurement_uncertainty=0.5, process_uncertainty=0.01,
                                    random_seed=1)
        m.initial_state([5, 0])
        sample = m.sample(2.0)
        zk = np.asarray(sample.zk)
        self.assertEqual(zk.shape[1], 2)   # analytic [real, imag]
        rec = m.filter(zk, smooth=True)
        _no_nan_inf(rec.xk, "AnalyticOscillator roundtrip xk")


# ===========================================================================
# OscillatorModel — one analysis window [fc ± dfc]
# ===========================================================================
class TestOscillatorModel(unittest.TestCase):

    def test_single_peak_dims(self):
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=2.0, fs=100)
        self.assertEqual(m.n, 2)   # one in-band mode -> [Re, Im]
        self.assertEqual(m.m, 2)   # FUSED demodulated band channel (filter space)
        self.assertEqual(m.l, 0)

    def test_window_mask_drops_out_of_band(self):
        """fc ± dfc is the active filter: candidates outside the window are
        generated but silently dropped from the state, even if more modes exist."""
        m = OscillatorModel("s", [8.0, 10.0, 30.0], fc=10.0, dfc=3.0, fs=100)
        np.testing.assert_allclose(np.sort(m.frequencies), [8.0, 10.0])  # 30 Hz masked out
        self.assertEqual((m.n, m.m), (4, 2))

    def test_window_mask_raises_if_nothing_in_band(self):
        with self.assertRaises(ValueError):
            OscillatorModel("s", [100.0], fc=10.0, dfc=2.0, fs=100)

    def test_filter_zk_at_decimated_rate(self):
        """After heterodyne + resample_poly downsampling, record.xk and record.zk
        live at the decimated rate (4·dfc); reconstruct() upsamples back to N."""
        fs, N = 256, 512
        t = np.arange(N) / fs
        strain = np.cos(2 * np.pi * 10 * t) + 0.05 * np.random.randn(N)
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs,
                            measurement_uncertainty=0.05, process_uncertainty=1e-4)
        rec = m.filter(strain, smooth=True)
        N_dec = rec.xk.shape[0]
        self.assertLess(N_dec, N)                    # decimated
        self.assertEqual(rec.xk.shape[1], 2)
        self.assertEqual(rec.zk.shape, (N_dec, 2))   # [Re, Im] at decimated rate
        _no_nan_inf(rec.zk, "OscillatorModel zk")

    def test_sample_produces_decimated_record(self):
        """sample() produces a record at the decimated rate; its xk and zk are
        consistent at that rate."""
        fs = 256
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs,
                            measurement_uncertainty=0.1, process_uncertainty=1e-4)
        m.initial_state([1.0, 0.0])
        sample = m.sample(2.0)
        zk = np.asarray(sample.zk)
        self.assertEqual(zk.shape[0], sample.xk.shape[0])
        _no_nan_inf(zk, "OscillatorModel sample zk")

    def test_reconstruct_shape_and_amplitudes_from_state(self):
        fs, N = 256, 512
        t = np.arange(N) / fs
        strain = np.cos(2 * np.pi * 10 * t) + 0.05 * np.random.randn(N)
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs,
                            measurement_uncertainty=0.05, process_uncertainty=1e-4)
        rec = m.filter(strain, smooth=True)
        recon = m.reconstruct(rec)
        self.assertEqual(recon.shape, (N,))            # upsampled back to original length
        N_dec = rec.xk.shape[0]
        amp = m.amplitudes(rec)                        # derived from xk at decimated rate
        self.assertEqual(amp.shape, (N_dec, 1))

    def test_tracks_amplitude(self):
        """After convergence, the recovered envelope amplitude matches truth."""
        np.random.seed(42)
        fs, N = 256, 1024
        t = np.arange(N) / fs
        truth = np.cos(2 * np.pi * 10 * t)
        strain = truth + 0.1 * np.random.randn(N)
        m = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs,
                            measurement_uncertainty=0.1, process_uncertainty=1e-4)
        rec = m.filter(strain, smooth=True)
        amp_arr = m.amplitudes(rec)
        N_dec = amp_arr.shape[0]
        amp = amp_arr[N_dec // 4:].mean()
        self.assertAlmostEqual(amp, 1.0, delta=0.25, msg=f"window amplitude off: {amp:.3f}")

    def test_fit_inharmonicity_recovers_f0_and_b(self):
        f0_true, B_true = 180.0, 0.1
        peaks = partial_frequencies(f0_true, B_true, 5)
        f0_fit, B_fit = OscillatorModel.fit_inharmonicity(peaks)
        self.assertAlmostEqual(f0_fit, f0_true, delta=0.1)
        self.assertAlmostEqual(B_fit, B_true, delta=1e-3)


# ===========================================================================
# Harmonic / Inharmonic ladder generation
# ===========================================================================
class TestPartialFrequencies(unittest.TestCase):

    def test_b_zero_is_harmonic_series(self):
        freqs = partial_frequencies(100.0, 0.0, 4)
        np.testing.assert_allclose(freqs, [100.0, 200.0, 300.0, 400.0])

    def test_b_positive_shifts_partials_upward(self):
        freqs = partial_frequencies(100.0, 0.1, 4)
        harmonic = np.array([100.0, 200.0, 300.0, 400.0])
        self.assertTrue(np.all(freqs > harmonic))


class TestHarmonicOscillatorModel(unittest.TestCase):

    def test_ladder_is_integer_multiples(self):
        m = HarmonicOscillatorModel("s", f0=100.0, fc=300.0, dfc=5.0, n_harmonics=4, fs=2000)
        np.testing.assert_allclose(m.partial_freqs, [100.0, 200.0, 300.0, 400.0])
        np.testing.assert_allclose(m.frequencies, [300.0])   # window keeps only H3

    def test_rejects_nonzero_b(self):
        with self.assertRaises(TypeError):
            HarmonicOscillatorModel("s", f0=100.0, fc=100.0, dfc=5.0, B=0.1, fs=2000)


class TestInharmonicOscillatorModel(unittest.TestCase):

    def test_ladder_matches_partial_frequencies(self):
        expected = partial_frequencies(180.0, 0.105, 3)
        m = InharmonicOscillatorModel("s", f0=180.0, fc=expected[1], dfc=20.0,
                                       n_harmonics=3, B=0.105, fs=4096)
        np.testing.assert_allclose(m.partial_freqs, expected)

    def test_window_keeps_only_in_band_partial(self):
        freqs_full = partial_frequencies(180.0, 0.105, 5)
        fc_2nd = freqs_full[1]
        m = InharmonicOscillatorModel("s", f0=180.0, fc=fc_2nd, dfc=5.0,
                                       n_harmonics=5, B=0.105, fs=4096)
        self.assertEqual(len(m.frequencies), 1)
        self.assertAlmostEqual(m.frequencies[0], fc_2nd)


# ===========================================================================
# Physical-parameter utilities (Young's modulus / wire geometry -> f0, B)
# ===========================================================================
class TestWirePhysics(unittest.TestCase):

    def test_wire_fundamental_known_steel_wire(self):
        # Steel suspension wire ~ KAGRA-like parameters -> f0 of order tens-to-hundreds Hz.
        f0 = wire_fundamental(d=0.31e-3, L=0.600, T=7.0, rho=7800.0)
        self.assertGreater(f0, 50.0)
        self.assertLess(f0, 250.0)

    def test_wire_inharmonicity_known_steel_wire(self):
        B = wire_inharmonicity(E=2e11, d=0.31e-3, L=0.600, T=7.0)
        self.assertGreater(B, 0.0)
        self.assertLess(B, 1.0)

    def test_wire_inharmonicity_scales_with_diameter(self):
        """B ~ d^4 — a thicker wire is much stiffer (more inharmonic)."""
        B_thin  = wire_inharmonicity(E=2e11, d=0.2e-3, L=0.6, T=7.0)
        B_thick = wire_inharmonicity(E=2e11, d=0.4e-3, L=0.6, T=7.0)
        self.assertAlmostEqual(B_thick / B_thin, 2.0 ** 4, delta=1e-6)

    def test_physical_params_feed_partial_frequencies(self):
        """End-to-end: physical parameters -> (f0, B) -> predicted peak positions."""
        f0 = wire_fundamental(d=0.31e-3, L=0.600, T=7.0, rho=7800.0)
        B = wire_inharmonicity(E=2e11, d=0.31e-3, L=0.600, T=7.0)
        freqs = partial_frequencies(f0, B, 3)
        self.assertEqual(len(freqs), 3)
        self.assertTrue(np.all(np.diff(freqs) > 0))   # strictly increasing


# ===========================================================================
# OscillatorBankModel — INDEPENDENT bank of windows
# ===========================================================================
class TestOscillatorBankModel(unittest.TestCase):

    def test_legacy_alias(self):
        """OscillatorCompositeModel remains a backward-compatible alias."""
        self.assertIs(OscillatorCompositeModel, OscillatorBankModel)

    def test_requires_at_least_one_window(self):
        with self.assertRaises(ValueError):
            OscillatorBankModel()

    def test_composite_filter_dims_and_reconstruct(self):
        fs, N = 256, 512
        t = np.arange(N) / fs
        strain = (np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 40 * t)
                  + 0.05 * np.random.randn(N))
        cands = [10.0, 40.0]
        w1 = OscillatorModel("s", cands, fc=10.0, dfc=3.0, fs=fs)
        w2 = OscillatorModel("s", cands, fc=40.0, dfc=3.0, fs=fs)
        comp = OscillatorBankModel(w1, w2)
        rec = comp.filter(strain)
        self.assertEqual(rec.xk.shape[1], 4)        # 2 windows × [Re, Im]
        self.assertEqual(rec.zk.shape[1], 4)        # identity postprocess: [Re1,Im1,Re2,Im2]
        _no_nan_inf(rec.xk, "OscillatorComposite xk")
        recon = comp.reconstruct(rec)
        self.assertEqual(recon.shape, (N,))         # upsampled back to original

    def test_composite_sample_produces_decimated_record(self):
        """sample() produces a record at the decimated rate."""
        fs = 256
        w1 = OscillatorModel("s", [10.0], fc=10.0, dfc=3.0, fs=fs)
        w2 = OscillatorModel("s", [40.0], fc=40.0, dfc=3.0, fs=fs)
        comp = OscillatorBankModel(w1, w2)
        sample = comp.sample(2.0)
        zk = np.asarray(sample.zk)
        self.assertEqual(zk.shape[0], sample.xk.shape[0])
        _no_nan_inf(zk, "OscillatorComposite sample zk")

    def test_violin_mode_suppression(self):
        """End-to-end KAGRA-style check: window-bank subtraction suppresses each
        injected mode by at least 10 dB at its own frequency bin."""
        np.random.seed(0)
        f0v, Bv, nv = 180.1, 0.105, 3
        fsv, Tv = 4096.0, 4.0
        Nv = int(Tv * fsv)
        tv = np.arange(Nv) / fsv
        sigma_noise = 5e-19
        Qs = [1.3e4, 1.8e4, 2.1e4]
        A = [1.0e-18, 3.0e-19, 1.5e-19]
        freqs = partial_frequencies(f0v, Bv, nv)
        strain = sigma_noise * np.random.randn(Nv)
        for a, f, Q in zip(A, freqs, Qs):
            strain += a * np.exp(-tv / (Q / (np.pi * f))) * np.cos(2 * np.pi * f * tv)

        windows = [OscillatorModel("KAGRA", freqs, fc=f, dfc=5.0, Q=Q, fs=fsv,
                                    measurement_uncertainty=40 * sigma_noise,
                                    process_uncertainty=1e-21)
                  for f, Q in zip(freqs, Qs)]
        bank = OscillatorBankModel(*windows)
        for w, a in zip(bank.windows, A):
            w.initial_state([[a, 0.0]])
        bank.initial_state(np.vstack([w.x0 for w in bank.windows]))
        rec = bank.filter(strain, smooth=True)
        cleaned = strain - bank.reconstruct(rec)

        from numpy.fft import rfft, rfftfreq
        fr = rfftfreq(Nv, 1 / fsv)
        Sr = np.abs(rfft(strain))
        Sc = np.abs(rfft(cleaned))
        for fk in freqs:
            mask = (fr > fk - 3) & (fr < fk + 3)
            suppression_db = 20 * np.log10(Sr[mask].max() / Sc[mask].max())
            self.assertGreater(suppression_db, 10.0,
                               f"mode @ {fk:.1f} Hz suppressed only {suppression_db:.1f} dB")


if __name__ == "__main__":
    unittest.main()

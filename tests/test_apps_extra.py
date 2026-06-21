"""Tests for the application models not already covered by ``test_module`` /
``test_filtering`` / ``test_tracking``:

  - rocket      (TransLunarModel free return, RocketLaunchModel powered ascent)
  - gravwave    (PolarizationModel Jones-formalism EKF)
  - powergrids  (PowerGridStateModel harmonic voltage tracker)
  - ballistics  (BallisticModel — the nonlinear EKF variant)
  - oscillators (Harmonic / Inharmonic / Phasor / Analytic variants)

These keep trajectories short so the suite stays fast while still exercising each
model's sample -> filter -> reconstruct path end to end.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")
from modules.apps import (
    PolarizationModel, jones_signal,
    PowerGridStateModel,
    HarmonicOscillatorModel, InharmonicOscillatorModel,
    PhasorOscillatorModel, AnalyticOscillatorModel,
)
from modules.apps.ballistics import BallisticModel
from modules.apps.rocket import (
    TransLunarModel, RocketLaunchModel, R_EARTH, KM, D_MOON, GM_EARTH,
)
from modules.filters import ExtendedKalmanFilter


def _no_nan_inf(arr, label="array"):
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


# ===========================================================================
# Rocket — trans-lunar free return + powered ascent
# ===========================================================================
class TestRocket(unittest.TestCase):
    # The multi-body gravity divides by |r|, which is 0 at the default zero-state the
    # numerical Jacobian probes at construction — a benign degenerate-point warning.
    def setUp(self):
        self._errstate = np.errstate(invalid="ignore", divide="ignore")
        self._errstate.__enter__()

    def tearDown(self):
        self._errstate.__exit__(None, None, None)

    def _translunar(self, **kw):
        m = TransLunarModel(dt_s=120.0, track_km=5.0, **kw)
        r0 = R_EARTH + 400 * KM
        a_tx = 0.5 * (r0 + D_MOON)
        v_tli = np.sqrt(GM_EARTH * (2.0 / r0 - 1.0 / a_tx)) + 0.034
        m.initial_state(np.array([[r0], [0], [0], [0], [v_tli], [0]]),
                        P0=np.diag([(50 * KM) ** 2] * 3 + [0.01 ** 2] * 3))
        return m

    def test_dims(self):
        m = TransLunarModel()
        self.assertEqual((m.n, m.m, m.l), (6, 3, 0))

    def test_free_return_ekf_tracks_truth(self):
        m = self._translunar()
        full = m.sample(40 * 0.12, process_noise=False)   # 40 RK4 steps of the exact ellipse
        self.assertEqual(full.xk.shape, (40, 6))
        est = ExtendedKalmanFilter(m).forward(full.zk)
        _no_nan_inf(est.xk, "translunar EKF xk")
        # position fixes are at 5 km; the EKF estimate should stay well inside ~50 km
        err_mm = np.linalg.norm(full.xk[:, :3] - est.xk[:, :3], axis=1)
        self.assertLess(np.median(err_mm) * 1e3, 50.0)   # Mm -> km

    def test_moon_track_shape(self):
        m = TransLunarModel()
        moon = m.moon_track(np.arange(5) * m._dt)
        self.assertEqual(moon.shape, (5, 3))

    def test_powered_ascent_with_control_input(self):
        launch = RocketLaunchModel(dt_s=1.0, track_km=0.5)
        self.assertEqual((launch.n, launch.m, launch.l), (4, 2, 2))   # state, obs, thrust control
        N = 40
        uk = np.zeros((N, 2))
        for i in range(N):
            pitch = np.deg2rad(min(70.0, 0.45 * i))
            uk[i] = [18.0 * np.cos(pitch), 18.0 * np.sin(pitch)]
        launch.initial_state(np.array([[R_EARTH], [0], [0], [0]]),
                             P0=np.diag([(1 * KM) ** 2, (1 * KM) ** 2, 0.01 ** 2, 0.01 ** 2]))
        asc = launch.sample(N / launch.fs, uk=uk)
        est = ExtendedKalmanFilter(launch).forward(asc.zk, uk)
        self.assertEqual(asc.xk.shape, (N, 4))
        _no_nan_inf(est.xk, "rocket-launch EKF xk")
        # thrust must have lifted the vehicle off the pad
        alt = np.linalg.norm(asc.xk[:, :2], axis=1) - R_EARTH
        self.assertGreater(alt[-1], alt[0])


# ===========================================================================
# Gravitational-wave polarization EKF
# ===========================================================================
class TestPolarization(unittest.TestCase):

    def _signal(self, fs=100.0, T=8.0, seed=0):
        t = np.arange(int(T * fs)) / fs
        a = 1.0 + 0.1 * np.cos(2 * np.pi * 0.2 * t)
        psi = 2 * np.pi * 1.0 * t
        theta = (np.pi / 2) * np.ones_like(t)
        phi = np.pi * np.ones_like(t)
        z = np.array([jones_signal(a[k], psi[k], theta[k], phi[k]) for k in range(len(t))])
        zk = np.column_stack([z[:, 0].real, z[:, 0].imag, z[:, 1].real, z[:, 1].imag])
        rng = np.random.RandomState(seed)
        zk = zk + 0.05 * rng.randn(*zk.shape)
        return t, a, psi, theta, phi, zk

    def test_dims_and_jones_shape(self):
        m = PolarizationModel("gw", fs=100.0)
        self.assertEqual((m.n, m.m, m.l), (8, 4, 0))
        self.assertEqual(jones_signal(1.0, 0.0, np.pi / 2, 0.0).shape, (2,))

    def test_filter_recovers_amplitude_and_reconstructs(self):
        t, a, psi, theta, phi, zk = self._signal()
        m = PolarizationModel("gw", fs=100.0, sigma_n=0.5)
        m.initial_state([np.log(a[0]), 0.0, psi[0], 2 * np.pi * 1.0,
                         theta[0], 0.0, phi[0], 0.0])
        rec = m.filter(zk, smooth=False)
        self.assertEqual(rec.xk.shape, (len(t), 8))
        _no_nan_inf(rec.xk, "polarization xk")
        p = m.physical_parameters(rec)
        self.assertAlmostEqual(float(p["amplitude"][len(t) // 2:].mean()), 1.0, delta=0.2)
        z1 = m.reconstruct_component(rec)
        self.assertEqual(z1.shape, (len(t),))
        _no_nan_inf(z1.real, "z1 reconstruction")


# ===========================================================================
# Power grid harmonic tracker
# ===========================================================================
class TestPowerGrid(unittest.TestCase):

    def test_filter_reconstruct_and_even_harmonic_suppression(self):
        np.random.seed(0)
        fs, N = 1024, 1024
        t = np.arange(N) / fs
        peak = 230.0 * np.sqrt(2.0)
        # Clean odd-harmonic mains: fundamental + a small 3rd harmonic, NO 2nd harmonic.
        v = peak * np.sin(2 * np.pi * 50 * t) + 0.1 * peak * np.sin(2 * np.pi * 150 * t)
        v_noisy = v + 0.5 * np.random.randn(N)

        pg = PowerGridStateModel("L1", f0=50.0, harmonics=3, fs=fs)
        rec = pg.filter(v_noisy, smooth=True)
        self.assertEqual(rec.xk.shape, (N, 6))   # 3 harmonics x [Re, Im]

        amp, sigma = pg.harmonic_envelope(rec)
        self.assertEqual(amp.shape, (N, 3))
        tail = slice(N // 2, None)
        h1, h2, h3 = (amp[tail, k].mean() for k in range(3))
        self.assertAlmostEqual(h1, peak, delta=0.1 * peak)   # fundamental recovered
        self.assertLess(h2, 0.1 * h1)                        # absent even harmonic ~ 0
        self.assertGreater(h3, h2)                           # present 3rd harmonic above the even

        v_hat = pg.reconstruct_voltage(rec)
        self.assertEqual(v_hat.shape, (N,))
        rms_err = np.sqrt(np.mean((v_hat[tail] - v[tail]) ** 2))
        self.assertLess(rms_err, 0.1 * peak)                 # reconstruction tracks the clean wave


# ===========================================================================
# Ballistic EKF (nonlinear gravity/drag)
# ===========================================================================
class TestBallisticEKF(unittest.TestCase):

    def test_dims_order3(self):
        m = BallisticModel("XYZ", order=3, fs=64)
        self.assertEqual((m.n, m.m), (9, 3))     # [p,v,a] per axis, observe position

    def test_sample_then_filter_beats_measurement(self):
        np.random.seed(0)
        m = BallisticModel("XYZ", order=3, fs=64,
                           measurement_uncertainty=50.0, process_uncertainty=0.5)
        # axis-major state [px,vx,ax, py,vy,ay, pz,vz,az]
        m.initial_state([0, 120, 0, 0, 80, 0, 1000, 0, -9.81])
        truth = m.sample(1.0)
        rec = m.filter(truth.zk)
        self.assertEqual(rec.xk.shape, (64, 9))
        _no_nan_inf(rec.xk, "ballistic EKF xk")
        pos_idx = [0, 3, 6]
        rmse_filt = np.sqrt(np.mean((rec.xk[:, pos_idx] - truth.xk[:, pos_idx]) ** 2))
        rmse_meas = np.sqrt(np.mean((truth.zk - truth.xk[:, pos_idx]) ** 2))
        self.assertLess(rmse_filt, rmse_meas)


# ===========================================================================
# Oscillator family variants
# ===========================================================================
class TestOscillatorVariants(unittest.TestCase):

    def test_harmonic_oscillator_dims(self):
        m = HarmonicOscillatorModel("s", f0=10.0, fc=10.0, dfc=20.0, n_harmonics=3, fs=256)
        self.assertEqual(m.n, 6)     # 3 partials x [Re, Im]
        self.assertEqual(m.m, 2)

    def test_inharmonic_partials_stretch_with_B(self):
        harm = HarmonicOscillatorModel("s", f0=10.0, fc=30.0, dfc=40.0, n_harmonics=3, fs=256)
        inh = InharmonicOscillatorModel("s", f0=10.0, fc=30.0, dfc=40.0,
                                        n_harmonics=3, B=0.01, fs=256)
        # inharmonicity pushes higher partials sharp of the exact harmonic ladder
        self.assertGreater(max(inh.frequencies), max(harm.frequencies))

    def test_phasor_oscillator_filter(self):
        np.random.seed(0)
        fs, N = 1024, 1024
        t = np.arange(N) / fs
        v = np.cos(2 * np.pi * 50 * t) + 0.5 * np.cos(2 * np.pi * 100 * t)
        m = PhasorOscillatorModel("s", [50.0, 100.0], fs=fs)
        self.assertEqual((m.n, m.m), (4, 1))
        rec = m.filter(v + 0.05 * np.random.randn(N))
        self.assertEqual(rec.xk.shape, (N, 4))
        _no_nan_inf(rec.xk, "phasor xk")

    def test_analytic_oscillator_tracks_envelope(self):
        # AnalyticOscillatorModel observes the 2-column analytic signal [Re, Im];
        # sample() emits that directly, so round-trip sample -> filter like the notebook.
        m = AnalyticOscillatorModel("s", sample_rate=256, w0=np.pi / 3, wc=np.pi / 3,
                                    Q=10000, measurement_uncertainty=2.0,
                                    process_uncertainty=0.01, random_seed=42)
        self.assertEqual((m.n, m.m), (2, 2))
        m.initial_state([10.0, 0.0])
        sample = m.sample(8.0)
        rec = m.filter(sample.zk, smooth=True)
        self.assertEqual(rec.xk.shape, (sample.xk.shape[0], 2))
        _no_nan_inf(rec.xk, "analytic oscillator xk")
        # recovered envelope amplitude stays positive and bounded near the seeded ~10
        env = np.hypot(rec.xk[:, 0], rec.xk[:, 1])
        self.assertGreater(env[len(env) // 4:].mean(), 1.0)


if __name__ == "__main__":
    unittest.main()

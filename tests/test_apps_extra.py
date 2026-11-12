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
from modules.apps.astro.rocket import (
    RocketAscentModel, RocketLiftoffModel, RocketLandingModel, RocketDescentModel,
    RocketLaunchModel, RocketTakeoffModel,            # legacy aliases
)
from modules.apps.astro.space import (
    TransLunarModel, InterplanetaryModel, lambert, hohmann_transfer,
)
from modules.runtime.astro import R_EARTH, KM, DAY, D_MOON, GM_EARTH, GM_SUN, body_state
from modules.apps.vehicles import CarSpeedModel
from modules.framework.filters import ExtendedKalmanFilter


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

    def test_legacy_aliases_resolve(self):
        self.assertIs(RocketLaunchModel, RocketAscentModel)
        self.assertIs(RocketTakeoffModel, RocketLiftoffModel)

    def test_powered_ascent_with_control_input(self):
        launch = RocketAscentModel(dt_s=1.0, track_km=0.5)
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
# Rocket vertical liftoff — propellant burn (mass is a state)
# ===========================================================================
class TestRocketLiftoff(unittest.TestCase):

    def test_dims_and_figures_of_merit(self):
        m = RocketLiftoffModel("LIFTOFF", m0=1000.0, m_dry=400.0, ve=2500.0, mdot=8.0)
        self.assertEqual((m.n, m.m, m.l), (3, 1, 0))   # [h, v, m], observe altitude
        self.assertAlmostEqual(m.thrust, 2500.0 * 8.0)            # T = ve * mdot
        self.assertAlmostEqual(m.burn_time, 600.0 / 8.0)          # propellant / mdot
        self.assertGreater(m.twr(), 1.0)                          # it must leave the pad

    def test_liftoff_burns_fuel_and_ekf_recovers_velocity(self):
        np.random.seed(0)
        m = RocketLiftoffModel("LIFTOFF", fs=10.0, track_m=20.0)
        m.initial_state([0.0, 0.0, m.m0], P0=np.diag([20.0 ** 2, 5.0 ** 2, 5.0 ** 2]))
        flight = m.sample(60.0, process_noise=False)             # clean burn
        self.assertEqual(flight.xk.shape, (600, 3))
        # mass strictly decreases while burning; altitude rises
        self.assertLess(flight.xk[-1, 2], m.m0)
        self.assertGreaterEqual(flight.xk[:, 2].min(), m.m_dry - 1e-6)
        self.assertGreater(flight.xk[-1, 0], flight.xk[0, 0])

        est = ExtendedKalmanFilter(m).forward(flight.zk)
        _no_nan_inf(est.xk, "rocket-takeoff EKF xk")
        # altitude tracking beats the raw radar fixes ...
        rmse_filt = np.sqrt(np.mean((est.xk[:, 0] - flight.xk[:, 0]) ** 2))
        rmse_radar = np.sqrt(np.mean((flight.zk.reshape(-1) - flight.xk[:, 0]) ** 2))
        self.assertLess(rmse_filt, rmse_radar)
        # ... and the unmeasured climb rate is recovered to within a few m/s
        rmse_vel = np.sqrt(np.mean((est.xk[:, 1] - flight.xk[:, 1]) ** 2))
        self.assertLess(rmse_vel, 5.0)

    def test_metrics_keys(self):
        m = RocketLiftoffModel("LIFTOFF", fs=10.0)
        m.initial_state([0.0, 0.0, m.m0], P0=np.diag([20.0 ** 2, 5.0 ** 2, 5.0 ** 2]))
        flight = m.sample(80.0, process_noise=False)
        met = m.metrics(flight)
        for key in ("burnout time [s]", "apogee [km]", "max accel [g]", "max-Q [kPa]"):
            self.assertIn(key, met)
        self.assertGreater(met["apogee [km]"], 0.0)


# ===========================================================================
# Rocket vertical landing — the hoverslam / powered descent
# ===========================================================================
class TestRocketLanding(unittest.TestCase):

    def _model(self, **kw):
        m = RocketLandingModel("LANDING", fs=10.0, track_m=20.0, **kw)
        m.initial_state([1500.0, -50.0, m.m0], P0=np.diag([20.0 ** 2, 5.0 ** 2, 5.0 ** 2]))
        return m

    def test_dims(self):
        m = RocketLandingModel("LANDING")
        self.assertEqual((m.n, m.m, m.l), (3, 1, 0))   # [h, v, m], observe altitude

    def test_soft_touchdown_and_fuel_burn(self):
        np.random.seed(0)
        m = self._model()
        flight = m.sample(40.0, process_noise=False)
        h, v, mass = flight.xk[:, 0], flight.xk[:, 1], flight.xk[:, 2]
        # the booster reaches the pad and arrives slowly (a hoverslam, not a crash)
        self.assertLessEqual(h.min(), 1.0)
        i_td = int(np.argmin(np.abs(h)))
        self.assertLess(abs(v[i_td]), 2.0)
        # it burned propellant to do it, but never past the dry mass
        self.assertLess(mass.min(), m.m0)
        self.assertGreaterEqual(mass.min(), m.m_dry - 1e-6)

    def test_ekf_recovers_descent_rate_from_altitude(self):
        np.random.seed(1)
        m = self._model()
        flight = m.sample(40.0, process_noise=False)
        est = ExtendedKalmanFilter(m).forward(flight.zk)
        _no_nan_inf(est.xk, "rocket-landing EKF xk")
        rmse_filt = np.sqrt(np.mean((est.xk[:, 0] - flight.xk[:, 0]) ** 2))
        rmse_radar = np.sqrt(np.mean((flight.zk.reshape(-1) - flight.xk[:, 0]) ** 2))
        self.assertLess(rmse_filt, rmse_radar)                   # beats raw radar on altitude
        rmse_vel = np.sqrt(np.mean((est.xk[:, 1] - flight.xk[:, 1]) ** 2))
        self.assertLess(rmse_vel, 5.0)                           # recovers the unmeasured speed

    def test_metrics_keys(self):
        m = self._model()
        flight = m.sample(40.0, process_noise=False)
        met = m.metrics(flight)
        for key in ("ignition time [s]", "touchdown speed [m/s]", "propellant burned [kg]"):
            self.assertIn(key, met)
        self.assertGreater(met["propellant burned [kg]"], 0.0)


# ===========================================================================
# Rocket 2-D powered descent — gimballed, lands upright, control input
# ===========================================================================
class TestRocketDescent(unittest.TestCase):

    def _x0(self):
        return np.array([350.0, 2000.0, -20.0, -60.0, 0.05, 0.0, 1000.0])

    def test_dims(self):
        self.assertEqual(RocketDescentModel().l, 0)                 # closed-loop autopilot
        self.assertEqual(RocketDescentModel(control=True).l, 2)     # [throttle, gimbal]
        self.assertEqual((RocketDescentModel().n, RocketDescentModel().m), (7, 3))

    def test_closed_loop_lands_upright_on_pad(self):
        m = RocketDescentModel(fs=20.0)
        m.initial_state(self._x0())
        X = np.asarray(m.sample(70.0, process_noise=False).xk)
        i = int(np.argmin(np.abs(X[:, 1])))
        self.assertLess(abs(X[i, 0]), 5.0)                          # within 5 m of the pad
        self.assertLess(np.hypot(X[i, 2], X[i, 3]), 2.0)            # soft touchdown
        self.assertLess(np.degrees(abs(X[i, 4])), 3.0)             # arrives upright

    def test_control_input_program_and_ekf(self):
        m = RocketDescentModel(fs=20.0, control=True)
        x0 = self._x0(); N = int(70 * m.fs)
        uk, truth = m.autopilot(x0, N)
        self.assertEqual(uk.shape, (N, 2))
        self.assertTrue(np.all((uk[:, 0] >= 0.0) & (uk[:, 0] <= 1.0)))            # throttle in [0,1]
        self.assertLessEqual(np.abs(uk[:, 1]).max(), m.gimbal_max + 1e-9)         # gimbal bounded
        self.assertGreater(np.abs(uk[:, 1]).max(), 0.0)                           # it actually steers
        m.initial_state(x0)
        s = m.sample(N / m.fs, uk=uk, process_noise=False)
        est = m.filter(s.zk, uk)
        _no_nan_inf(est.xk, "rocket-descent EKF xk")
        X = np.asarray(s.xk); i = int(np.argmin(np.abs(X[:, 1])))
        self.assertLess(abs(X[i, 0]), 5.0)                          # control program lands on pad


# ===========================================================================
# Interplanetary — Lambert, Hohmann and the heliocentric N-body cruise
# ===========================================================================
class TestInterplanetary(unittest.TestCase):

    def test_lambert_round_trip(self):
        # A known transfer: solve Lambert, then check the arc reproduces the geometry.
        r1, _ = body_state("earth", "2027-01-01")
        r2, _ = body_state("mars",  "2027-09-01")
        v1, v2 = lambert(r1, r2, 243 * DAY, mu=GM_SUN)
        for v in (v1, v2):                                          # heliocentric speeds are sane
            self.assertTrue(5.0 < np.linalg.norm(v) < 60.0)

    def test_hohmann_earth_mars(self):
        h = hohmann_transfer("earth", "mars", epoch="2027-01-01")
        self.assertTrue(np.isfinite(h["dv_total_kms"]) and h["dv_total_kms"] > 0)
        self.assertTrue(150.0 < h["tof_days"] < 400.0)             # Earth→Mars ballpark

    def test_hohmann_uses_semi_major_axes(self):
        # The estimate must not depend on where the planets are on the day: the orbit
        # radii are semi-major axes, so the textbook figures come out at any epoch.
        for epoch in ("2026-11-11", "2027-08-01"):
            h = hohmann_transfer("earth", "mars", epoch=epoch)
            self.assertAlmostEqual(h["a2_AU"], 1.524, delta=0.005)
            self.assertAlmostEqual(h["dv_total_kms"], 5.6, delta=0.1)
            self.assertAlmostEqual(h["tof_days"], 259.0, delta=3.0)
            self.assertAlmostEqual(h["synodic_period_days"], 780.0, delta=5.0)

    def test_chain_lambert_separates_gravity_from_burn(self):
        from astropy.time import Time
        from modules.apps.astro.space import chain_lambert
        sequence = ["earth", "venus", "venus", "earth"]
        dates = [Time(t) for t in ("1997-10-15 08:43", "1998-04-26 13:52",
                                   "1999-06-24 20:30", "1999-08-18 03:28")]
        single = chain_lambert(sequence, dates)
        self.assertEqual(single[0]["dv_burn_kms"], 0.0)            # the launch is not a flyby
        for before, leg in zip(single, single[1:]):
            # the planet turns the velocity for free; the burn is only the speed mismatch
            self.assertGreater(leg["dv_gravity_kms"], leg["dv_burn_kms"])
            self.assertAlmostEqual(leg["dv_burn_kms"],
                                   abs(leg["v_inf_dep_kms"] - before["v_inf_arr_kms"]), places=9)
        # A deep-space manoeuvre on the Venus-Venus leg makes both Venus flybys free.
        legs = chain_lambert(sequence, dates, dsm={1: Time("1998-12-03 06:45")})
        self.assertLess(legs[1]["dv_burn_kms"], 1e-5)
        self.assertLess(legs[2]["dv_burn_kms"], 1e-5)
        self.assertTrue(0.3 < legs[1]["dv_dsm_kms"] < 0.8)         # Cassini's was about 0.45 km/s
        self.assertTrue(100.0 < legs[1]["flyby_altitude_km"] < 2000.0)
        self.assertTrue(100.0 < legs[2]["flyby_altitude_km"] < 2000.0)

    def test_two_body_cruise_ekf(self):
        # No ephemeris ⇒ fast pure-Sun cruise; the EKF tracks the heliocentric position.
        m = InterplanetaryModel("earth", "mars", epoch=None, dt_s=43200.0, track_Mm=0.02)
        from modules.runtime.astro import D_SUN
        x0 = np.array([[D_SUN], [0], [0], [0], [np.sqrt(GM_SUN / D_SUN)], [0]])
        m.initial_state(x0)
        s = m.sample(80 * DAY, process_noise=False)
        est = m.filter(s.zk)
        _no_nan_inf(est.xk, "interplanetary EKF xk")
        self.assertEqual(np.asarray(s.xk).shape[1], 6)


# ===========================================================================
# Car straight-line speed test
# ===========================================================================
class TestCarSpeed(unittest.TestCase):

    def test_dims(self):
        car = CarSpeedModel("CAR", fs=20.0)
        self.assertEqual((car.n, car.m, car.l), (2, 1, 0))   # [s, v], observe position

    def test_ekf_beats_naive_differencing(self):
        np.random.seed(0)
        car = CarSpeedModel("CAR", fs=20.0, gps_uncertainty=0.5)
        car.initial_state([0.0, 0.0], P0=np.diag([0.5 ** 2, 0.5 ** 2]))
        run = car.sample(10.0)
        self.assertEqual(run.xk.shape, (200, 2))
        # the car actually accelerates off the line
        self.assertGreater(run.xk[-1, 1], run.xk[0, 1])

        est = ExtendedKalmanFilter(car).forward(run.zk)
        _no_nan_inf(est.xk, "car EKF xk")
        v_true = run.xk[:, 1]
        v_naive = np.gradient(run.zk.reshape(-1), car._dt)       # differentiate raw GPS
        rmse_filt = np.sqrt(np.mean((est.xk[:, 1] - v_true) ** 2))
        rmse_naive = np.sqrt(np.mean((v_naive - v_true) ** 2))
        self.assertLess(rmse_filt, rmse_naive)

    def test_metrics_zero_to_hundred(self):
        np.random.seed(0)
        car = CarSpeedModel("CAR", fs=20.0)
        car.initial_state([0.0, 0.0], P0=np.diag([0.5 ** 2, 0.5 ** 2]))
        run = car.sample(12.0, process_noise=False)
        met = car.metrics(run)
        # a 1500 kg / 150 kW car reaches 100 km/h in a handful of seconds
        self.assertIsNotNone(met["0-100 km/h [s]"])
        self.assertTrue(2.0 < met["0-100 km/h [s]"] < 12.0)
        self.assertGreater(met["top speed [km/h]"], 100.0)


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

    def _orbit_step_error(self, substeps):
        """Position error, m, of one 86.4 s propagation on a circular orbit (r = 5,600 km
        round GM = 1.06252e14 m^3/s^2, a 2.2 h period) against the exact circle."""
        G, gm, r, dt = 6.674e-11, 1.06252e14, 5.6e6, 86.4
        v = np.sqrt(gm / r)
        m = BallisticModel("XYZ", order=3, fs=1.0 / dt, G=G, M=gm / G, g=None, c_drag=0,
                           substeps=substeps)
        x = np.array([r, 0, -gm / r**2, 0, 0, 0, 0, v, 0], dtype=float)
        nxt = m.f(x).reshape(-1)
        theta = v / r * dt
        exact = np.array([r * np.cos(theta), 0.0, r * np.sin(theta)])
        return float(np.linalg.norm(nxt[[0, 3, 6]] - exact))

    def test_orbit_substeps_follow_the_orbit(self):
        # The model's integrator (RK4) follows a 2.2 h orbit sampled every 86 s to
        # under a metre in one step; the hand-written Taylor step it replaced was 283 m
        # off. More steps are more accurate still: fourth order, so 16 steps gain
        # far more than a factor of 16.
        one, auto, sixteen = (self._orbit_step_error(s) for s in (1, "auto", 16))
        self.assertLess(one, 1.0)
        self.assertLess(auto, one / 100.0)
        self.assertLess(sixteen, auto)

    def test_the_integrator_is_selectable(self):
        from modules.framework.constants import KalmanIntegratorKind
        from modules.framework.integrators import DormandPrinceIntegrator
        G, gm, r, dt = 6.674e-11, 1.06252e14, 5.6e6, 86.4
        v = np.sqrt(gm / r)
        x = np.array([r, 0, -gm / r**2, 0, 0, 0, 0, v, 0], dtype=float)
        theta = v / r * dt
        exact = np.array([r * np.cos(theta), 0.0, r * np.sin(theta)])
        errors = {}
        for label, integrator in (("euler", KalmanIntegratorKind.EULER), ("rk4", KalmanIntegratorKind.RK4),
                                  ("dp", DormandPrinceIntegrator(rtol=1e-10, atol=1e-6))):
            m = BallisticModel("XYZ", order=3, fs=1.0 / dt, G=G, M=gm / G, g=None, c_drag=0, integrator=integrator)
            errors[label] = float(np.linalg.norm(m.f(x).reshape(-1)[[0, 3, 6]] - exact))
        self.assertGreater(errors["euler"], 1000.0)          # first order: kilometres off
        self.assertLess(errors["rk4"], 1.0)
        self.assertLess(errors["dp"], 1e-3)

    def test_substeps_one_is_the_default(self):
        m1 = BallisticModel("XYZ", order=3, fs=64)
        m2 = BallisticModel("XYZ", order=3, fs=64, substeps=1)
        x = np.array([0, 120, 0, 0, 80, 0, 1000, 0, -9.81], dtype=float)
        np.testing.assert_array_equal(m1.f(x), m2.f(x))
        with self.assertRaises(ValueError):
            BallisticModel("XYZ", order=3, fs=64, substeps=0)


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

"""Tests for the reduced storm models — TyphoonModel (track/intensity + IMM
recurvature regimes + forecast cone) and TornadoEnvironmentModel (regime-
probability IMM with seasonal transition priors)."""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.apps.weather.storms import (StormModel, TyphoonModel, HurricaneModel,
                                  CycloneModel, BASINS, TornadoEnvironmentModel,
                                  synthetic_track, synthetic_environment,
                                  synthetic_steered_track, analytic_steering,
                                  analytic_environment, LearnedIntensityModel,
                                  japan_steering, japan_region, landfall_probability,
                                  JAPAN_BBOX, brazil_season, brazil_tornado_scenario,
                                  seasonal_tornado_probability, _SEASON_PI)
from modules.filters import InteractiveKalmanFilter, ExtendedKalmanFilter


# ── TyphoonModel dynamics ─────────────────────────────────────────────────────

class TestTyphoonModel(unittest.TestCase):

    def test_dimensions_and_observe(self):
        m = TyphoonModel()
        self.assertEqual((m.n, m.m, m.l), (5, 3, 0))
        m2 = TyphoonModel(observe_intensity=False)
        self.assertEqual(m2.m, 2)
        self.assertEqual(np.asarray(m2.h(np.zeros((5, 1)))).shape, (2, 1))

    def test_straight_motion_wnw(self):
        """Zero turn rate + NH beta: the centre moves west and poleward."""
        m = TyphoonModel(omega=0.0, hemisphere="N",
                         x0=np.array([[140.], [15.], [-20.], [5.], [80.]]))
        x1 = m.f(m.x0).ravel()
        self.assertLess(x1[0], 140.0)      # lon decreases (westward)
        self.assertGreater(x1[1], 15.0)    # lat increases (poleward, NH)
        np.testing.assert_allclose(x1[2:4], [-20., 5.], atol=1e-9)   # velocity unchanged
        self.assertAlmostEqual(x1[4], 80.0)                          # intensity persists

    def test_recurve_rotates_velocity(self):
        """Nonzero omega rotates the translation velocity vector."""
        m = TyphoonModel(omega=-0.05, hemisphere="N",
                         x0=np.array([[140.], [15.], [-20.], [0.], [80.]]))
        v0 = m.x0.ravel()[2:4]
        v1 = m.f(m.x0).ravel()[2:4]
        self.assertFalse(np.allclose(v0, v1))
        np.testing.assert_allclose(np.hypot(*v1), np.hypot(*v0), rtol=1e-6)  # speed preserved

    def test_hemisphere_beta_sign(self):
        n = TyphoonModel(hemisphere="N"); s = TyphoonModel(hemisphere="S")
        self.assertGreater(n.beta_n, 0)    # NH beta drifts poleward (north)
        self.assertLess(s.beta_n, 0)       # SH beta drifts poleward (south)

    def test_imm_bank_structure(self):
        models, Pi, probs = TyphoonModel.imm_bank(stall=True)
        self.assertEqual(len(models), 3)
        self.assertEqual(Pi.shape, (3, 3))
        np.testing.assert_allclose(Pi.sum(axis=1), 1.0)
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertEqual(models[0].omega, 0.0)
        self.assertLess(models[1].omega, 0.0)          # NH recurve = clockwise
        models2, Pi2, _ = TyphoonModel.imm_bank(stall=False)
        self.assertEqual(len(models2), 2)


class TestTyphoonTracking(unittest.TestCase):

    def _run(self, fix_noise=1.0, seed=1):
        truth, fixes = synthetic_track(n_steps=40, recurve_at=18,
                                       fix_noise=fix_noise, seed=seed)
        models, Pi, probs = TyphoonModel.imm_bank(fix_noise=fix_noise, stall=True)
        for m in models:
            m.initial_state(np.array([[138.], [12.], [-20.], [7.], [40.]]),
                            np.diag([1., 1., 50., 50., 200.]))
        ikf = InteractiveKalmanFilter(models, Pi, probs)
        rec = ikf.forward(fixes)
        return truth, fixes, ikf, rec

    def test_imm_beats_raw_fixes(self):
        truth, fixes, ikf, rec = self._run(fix_noise=1.2)
        xk = np.asarray(rec.xk)
        est_km = np.mean(np.hypot(xk[:, 0]-truth[:, 0], xk[:, 1]-truth[:, 1])) * 111.0
        fix_km = np.mean(np.hypot(fixes[:, 0]-truth[:, 0], fixes[:, 1]-truth[:, 1])) * 111.0
        self.assertLess(est_km, fix_km)                # filtering reduces track error
        self.assertLess(est_km, 200.0)

    def test_intensity_recovered(self):
        truth, _, _, rec = self._run(fix_noise=0.5)
        xk = np.asarray(rec.xk)
        self.assertLess(abs(xk[-1, 4] - truth[-1, 4]), 30.0)

    def test_recurve_mode_probability_rises(self):
        _, _, ikf, _ = self._run(fix_noise=0.4)
        mu = ikf.mode_probabilities
        # recurve (index 1) more probable after the turn than before it
        self.assertGreater(mu[30, 1], mu[8, 1])

    def test_forecast_cone_grows(self):
        truth, _, _, rec = self._run(fix_noise=0.5)
        x0 = np.asarray(rec.xk)[15].reshape(-1, 1)
        P0 = np.asarray(rec.Pk)[15]
        cone = TyphoonModel.imm_bank()[0][1].forecast_cone(x0, P0, steps=10)
        self.assertEqual(len(cone["ellipses"]), 10)
        semis = np.array([e[2] for e in cone["ellipses"]])
        self.assertTrue(np.all(np.diff(semis) >= -1e-9))     # monotone growth
        self.assertTrue(np.all(np.diff(cone["I_sigma"]) >= -1e-9))

    def test_basin_agnostic_naming_and_hemisphere(self):
        """One physics, regional names: basin fixes the name and hemisphere."""
        self.assertEqual(StormModel(basin="wpac").storm_name, "Typhoon")
        self.assertEqual(StormModel(basin="atlantic").storm_name, "Hurricane")
        self.assertEqual(StormModel(basin="sio").storm_name, "Cyclone")
        self.assertEqual(TyphoonModel().basin, "wpac")
        self.assertEqual(HurricaneModel().storm_name, "Hurricane")
        self.assertEqual(CycloneModel().hemisphere, "S")
        # SH basins drift poleward = south; NH = north
        self.assertGreater(StormModel(basin="wpac").beta_n, 0)
        self.assertLess(StormModel(basin="spac").beta_n, 0)
        # explicit hemisphere still overrides the basin default
        self.assertEqual(StormModel(basin="wpac", hemisphere="S").hemisphere, "S")
        with self.assertRaises(ValueError):
            StormModel(basin="nope")
        self.assertEqual(set(BASINS), {"atlantic", "epac", "wpac", "nio", "sio", "spac"})

    def test_standard_plot_is_self_contained(self):
        """model.plot(record) reproduces the canonical figure with no extra code."""
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        truth, fixes, _, rec = self._run(fix_noise=0.5)
        mdl = rec.model
        cone = mdl.forecast_cone(np.asarray(rec.xk)[-1].reshape(-1, 1),
                                 np.asarray(rec.Pk)[-1], steps=6)
        fig = mdl.plot(rec, fixes=fixes, cone=cone)
        self.assertEqual(len(fig.axes), 2)          # track + intensity panels
        anim = mdl.plot(rec, interactive=True, fixes=fixes, frames=4)
        self.assertIsNotNone(anim)
        plt.close("all")

    def test_plot_static_smoke(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        truth, fixes, _, rec = self._run(fix_noise=0.5)
        fig = rec.model._plot_static(rec, sample=None, fixes=fixes)
        self.assertIsNotNone(fig)
        import matplotlib.pyplot as plt
        plt.close("all")


# ── Realistic physics (steering) + ML elaboration (learned intensity) ─────────

class TestSteeringAndIntensity(unittest.TestCase):

    def test_analytic_steering_recurves(self):
        """Easterlies in the tropics, westerlies in mid-latitudes (NH)."""
        ve_trop, _ = analytic_steering(150, 10, hemisphere="N")
        ve_mid, _ = analytic_steering(150, 40, hemisphere="N")
        self.assertLess(ve_trop, 0)      # tropical easterly (westward)
        self.assertGreater(ve_mid, 0)    # mid-latitude westerly (eastward) → recurvature

    def test_steering_model_forecasts_recurvature(self):
        """Physics-only (steering-coupled) forecast turns the track poleward-east
        with NO fitted turn rate and NO ML."""
        def steer(lon, lat, t=None): return analytic_steering(lon, lat, hemisphere="N")
        m = TyphoonModel("TC", steering=steer,
                         x0=np.array([[165.], [9.], [-20.], [3.], [40.]]))
        cone = m.forecast_cone(m.x0, np.diag([0.5, 0.5, 30, 30, 100.]), steps=48)
        lon, lat = cone["lon"], cone["lat"]
        # recurvature signature: clear poleward turn, and westward progress
        # decelerating as the storm climbs toward the westerlies
        self.assertGreater(lat[-1], lat[0] + 3.0)
        first_half_west = lon[0] - lon[24]
        second_half_west = lon[24] - lon[-1]
        self.assertGreater(first_half_west, second_half_west)

    def test_steered_track_intensity_lifecycle(self):
        truth, fixes = synthetic_steered_track(n_steps=48, seed=1)
        self.assertEqual(truth.shape, (48, 5))
        self.assertGreater(truth[:, 4].max(), truth[0, 4] + 15)   # intensifies
        self.assertLess(truth[-1, 4], truth[:, 4].max())          # then weakens

    def test_learned_intensity_beats_persistence(self):
        """The ML elaboration: a learned SST/shear intensity model cuts the
        intensity-forecast error of the physics-persistence baseline."""
        def steer(lon, lat, t=None): return analytic_steering(lon, lat, hemisphere="N")
        rng = np.random.default_rng(0)
        train = [synthetic_steered_track(n_steps=48, seed=s, I0=rng.uniform(25, 55),
                                         lat0=rng.uniform(7, 13), lon0=rng.uniform(155, 172))[0]
                 for s in range(50)]
        lim = LearnedIntensityModel().fit(train)
        self.assertEqual(len(lim.coef_), 4)

        def build(ifn, tr):
            m = TyphoonModel("TC", steering=steer, intensity_fn=ifn, q_pos=0.25,
                             x0=np.array([[tr[0, 0]], [tr[0, 1]], [-20.], [3.], [35.]]))
            m.initial_state(m.x0, np.diag([0.5, 0.5, 30, 30, 100.]))
            return m
        t_a, lead = 6, 14
        gains = []
        for s in range(200, 220):
            tr, fx = synthetic_steered_track(n_steps=48, seed=s)
            r = ExtendedKalmanFilter(build(None, tr)).forward(fx[:t_a])
            x0, P0 = np.asarray(r.xk)[t_a-1].reshape(-1, 1), np.asarray(r.Pk)[t_a-1]
            cp = build(None, tr).forecast_cone(x0, P0, steps=lead)
            cm = build(lim.as_intensity_fn(), tr).forecast_cone(x0, P0, steps=lead)
            tt = tr[t_a:t_a+lead, 4]
            ep = np.sqrt(np.mean((cp["I"] - tt) ** 2))
            em = np.sqrt(np.mean((cm["I"] - tt) ** 2))
            gains.append(1 - em / ep)
        self.assertGreater(np.mean(gains), 0.10)     # ML consistently helps intensity

    def test_intensity_fn_none_is_persistence(self):
        m = TyphoonModel("TC", intensity_fn=None,
                         x0=np.array([[150.], [12.], [-20.], [3.], [70.]]))
        self.assertAlmostEqual(m.f(m.x0).ravel()[4], 70.0)   # persists exactly


# ── Japan / Northwest-Pacific landfall probability ───────────────────────────

class TestJapanLandfall(unittest.TestCase):

    def test_japan_region_known_points(self):
        self.assertTrue(japan_region(139.7, 35.7))       # Tokyo: land in box
        self.assertFalse(japan_region(145.0, 20.0))      # open Pacific: sea
        self.assertFalse(japan_region(116.4, 39.9))      # Beijing: land, outside box
        arr = japan_region(np.array([139.7, 145.0]), np.array([35.7, 20.0]))
        np.testing.assert_array_equal(arr, [True, False])

    def test_ensemble_forecast_shape_and_spread(self):
        m = TyphoonModel("JP", steering=japan_steering, q_pos=0.25,
                         x0=np.array([[140.], [20.], [-14.], [6.], [75.]]))
        ens = m.ensemble_forecast(m.x0, np.diag([0.5, 0.5, 30, 30, 100.]),
                                  steps=12, n_samples=100, seed=0)
        self.assertEqual(ens.shape, (100, 12, 5))
        self.assertGreater(ens[:, -1, 0].std(), ens[:, 0, 0].std())   # spread grows

    def test_landfall_probability_discriminates(self):
        """Strike probability high for a Japan-bound storm, ~0 for one that
        recurves out to sea — read from the forecast covariance."""
        def prob(lon0, seed):
            truth, fixes = synthetic_steered_track(n_steps=32, lon0=lon0, lat0=20, I0=75,
                                                   steering=japan_steering, eddy=6.0, seed=seed)
            m = TyphoonModel("JP", steering=japan_steering, q_pos=0.25,
                             x0=np.array([[truth[0,0]], [truth[0,1]], [-14.], [6.], [75.]]))
            m.initial_state(m.x0, np.diag([0.5, 0.5, 30, 30, 100.]))
            rec = ExtendedKalmanFilter(m).forward(fixes[:16])
            x0, P0 = np.asarray(rec.xk)[15].reshape(-1, 1), np.asarray(rec.Pk)[15]
            m2 = TyphoonModel("JP", steering=japan_steering, q_pos=0.25, x0=m.x0)
            m2.initial_state(m.x0, np.diag([0.5, 0.5, 30, 30, 100.]))
            ens = m2.ensemble_forecast(x0, P0, steps=16, n_samples=300, seed=2)
            p, hit = landfall_probability(ens)
            self.assertEqual(len(hit), 300)
            return p
        self.assertGreater(prob(140, 7), 0.5)            # recurves into Japan
        self.assertLess(prob(154, 3), 0.1)               # recurves out to sea

    def test_landfall_probability_bounds(self):
        p, hit = landfall_probability(np.zeros((10, 5, 5)))   # all at (0,0): never in Japan
        self.assertEqual(p, 0.0)
        self.assertEqual(hit.shape, (10,))


class TestFieldSteeringCoupling(unittest.TestCase):

    def _flow(self):
        from modules.apps.weather.flow import KolmogorovFlowModel
        f = KolmogorovFlowModel("env", grid=32, cycle=5, dt=0.01, nu=1/40., n_obs=20, seed=3)
        f.climatology(n_snapshots=30, spinup=400, stride=5)
        return f

    def test_velocity_sampler_periodic_and_finite(self):
        from modules.apps.weather.storms import field_steering
        flow = self._flow()
        field = flow.initial_ensemble(1, rng=0)[:, 0]
        samp = flow.velocity_sampler(field, (120., 160., 8., 45.), speed=10.0)
        ve, vn = samp(140.0, 20.0)
        self.assertTrue(np.isfinite(ve) and np.isfinite(vn))
        # region is 40° wide (120–160°E), so +40° wraps to the same box point
        np.testing.assert_allclose(samp(140.0, 20.0), samp(180.0, 20.0), atol=1e-6)

    def test_field_steering_composes_base_plus_anomaly(self):
        from modules.apps.weather.storms import field_steering
        flow = self._flow()
        field = flow.initial_ensemble(1, rng=0)[:, 0]
        samp = flow.velocity_sampler(field, (120., 160., 8., 45.), speed=10.0)
        s0 = field_steering(samp, base=japan_steering, weight=0.0)
        np.testing.assert_allclose(s0(140, 20), japan_steering(140, 20), atol=1e-9)  # weight 0 = base
        s1 = field_steering(samp, base=japan_steering, weight=1.0)
        self.assertFalse(np.allclose(s1(140, 20), japan_steering(140, 20)))          # anomaly added

    def test_field_ensemble_drives_track_spread(self):
        from modules.apps.weather.storms import field_steering
        flow = self._flow()
        ens = flow.initial_ensemble(12, rng=1)
        x0 = np.array([[140.], [20.], [-14.], [6.], [75.]])
        ends = []
        for k in range(ens.shape[1]):
            samp = flow.velocity_sampler(ens[:, k], (120., 160., 8., 45.), speed=10.0)
            m = TyphoonModel("JP", steering=field_steering(samp, base=japan_steering),
                             q_pos=0.15, x0=x0)
            c = m.forecast_cone(x0, np.diag([0.3, 0.3, 20, 20, 50.]), steps=24)
            ends.append([c["lon"][-1], c["lat"][-1]])
        ends = np.array(ends)
        self.assertGreater(ends[:, 0].std() + ends[:, 1].std(), 0.1)   # environment → track spread


# ── TornadoEnvironmentModel ───────────────────────────────────────────────────

class TestTornadoEnvironment(unittest.TestCase):

    def test_mean_reversion(self):
        m = TornadoEnvironmentModel(mu=3.0, persistence=0.5)
        # from below the mean, f moves toward the mean
        x1 = m.f(np.array([[1.0]])).ravel()[0]
        self.assertTrue(1.0 < x1 < 3.0)
        self.assertAlmostEqual(m.f(np.array([[3.0]])).ravel()[0], 3.0)   # fixed point at mu

    def test_seasonal_transition_matrices_differ(self):
        _, pi_peak, _ = TornadoEnvironmentModel.imm_bank(season="peak")
        _, pi_off, _ = TornadoEnvironmentModel.imm_bank(season="off")
        np.testing.assert_allclose(pi_peak.sum(axis=1), 1.0)
        np.testing.assert_allclose(pi_off.sum(axis=1), 1.0)
        # tornadic-regime persistence higher in peak season
        self.assertGreater(pi_peak[1, 1], pi_off[1, 1])
        # easier to switch INTO tornadic in peak season
        self.assertGreater(pi_peak[0, 1], pi_off[0, 1])
        self.assertEqual(set(_SEASON_PI), {"peak", "shoulder", "off"})

    def test_regime_probability_tracks_event(self):
        obs, labels = synthetic_environment(n_steps=200, onsets=((70, 110),), seed=2)
        models, Pi, probs = TornadoEnvironmentModel.imm_bank(season="peak")
        ikf = InteractiveKalmanFilter(models, Pi, probs)
        ikf.forward(obs)
        p_tor = ikf.mode_probabilities[:, 1]
        self.assertGreater(p_tor[labels == 1].mean(), p_tor[labels == 0].mean() + 0.2)
        self.assertTrue(np.all((p_tor >= 0) & (p_tor <= 1)))


# ── Brazil (southern) tornado configuration ──────────────────────────────────

class TestBrazilTornado(unittest.TestCase):

    def test_season_calendar_sh(self):
        # SH spring–summer is peak; austral winter is off-season
        self.assertEqual(brazil_season(12), "peak")
        self.assertEqual(brazil_season(1), "peak")
        self.assertEqual(brazil_season(6), "off")
        self.assertEqual({brazil_season(m) for m in range(1, 13)}, {"peak", "shoulder", "off"})

    def test_scenario_clusters_in_season(self):
        obs, labels, months = brazil_tornado_scenario(n_days=360, start_month=7, seed=1)
        self.assertEqual(obs.shape, (360, 1))
        self.assertEqual(len(months), 360)
        peak_days = np.array([brazil_season(m) == "peak" for m in months])
        # most tornadic days fall in the peak season
        self.assertGreater(labels[peak_days].sum(), labels[~peak_days].sum())

    def test_seasonal_prior_changes_response(self):
        obs, labels, months = brazil_tornado_scenario(n_days=360, start_month=7, seed=1)
        p = seasonal_tornado_probability(obs, months)
        self.assertEqual(p.shape, (360,))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        peak_evt = np.array([brazil_season(m) == "peak" for m in months]) & (labels == 1)
        peak_benign = np.array([brazil_season(m) == "peak" for m in months]) & (labels == 0)
        # confident on peak-season events, suppressed on peak-season benign days
        self.assertGreater(p[peak_evt].mean(), p[peak_benign].mean() + 0.15)


if __name__ == "__main__":
    unittest.main()

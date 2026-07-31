"""Tests for the ensemble Kalman machinery — EnsembleKalmanEngine /
EnsembleKalmanFilter, the KolmogorovFlowModel app, and the TwinExperiment
OSSE harness (including the RNG-discipline invariance that motivated its
per-method random streams).
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.framework import LinearKalmanModel
from modules.framework.engines.linear import LinearKalmanEngine
from modules.framework.engines.ensemble import EnsembleKalmanEngine, gaspari_cohn
from modules.filters import EnsembleKalmanFilter, ParticleFilter
from modules.apps.weather.flow import KolmogorovFlowModel, TwinExperiment


# ── shared fixtures ───────────────────────────────────────────────────────────

DT = 0.1
F_CV = np.array([[1.0, DT], [0.0, 1.0]])
H_POS = np.array([[1.0, 0.0]])
Q0 = np.eye(2) * 1e-3
R0 = np.array([[0.25]])
X0 = np.array([[0.0], [1.0]])
P0 = np.eye(2) * 0.1


def _cv_model():
    return LinearKalmanModel("cv", F=F_CV, H=H_POS, Q=Q0, R=R0, fs=10, x0=X0, P0=P0)


def _tiny_flow(seed=7, **kw):
    """Small, fast Kolmogorov flow (256-dim state) with a primed short climatology."""
    kw.setdefault("grid", 16)
    kw.setdefault("cycle", 10)
    kw.setdefault("dt", 0.01)
    kw.setdefault("nu", 1 / 40.0)
    kw.setdefault("n_obs", 40)
    kw.setdefault("loc_radius", 4.0)
    model = KolmogorovFlowModel("flow", seed=seed, **kw)
    model.climatology(n_snapshots=40, spinup=400, stride=5)
    return model


# ── gaspari_cohn ──────────────────────────────────────────────────────────────

class TestGaspariCohn(unittest.TestCase):

    def test_support_and_range(self):
        d = np.linspace(0, 30, 200)
        C = gaspari_cohn(d, 5.0)
        self.assertAlmostEqual(C[0], 1.0, places=12)
        self.assertTrue(np.all(C >= 0) and np.all(C <= 1))
        self.assertTrue(np.all(C[d > 10.0] == 0.0))          # zero beyond 2L

    def test_monotone_decay(self):
        d = np.linspace(0, 10, 100)
        C = gaspari_cohn(d, 5.0)
        self.assertTrue(np.all(np.diff(C) <= 1e-12))


# ── EnsembleKalmanEngine vs exact linear KF ──────────────────────────────────

class TestEnsembleEngine(unittest.TestCase):

    def test_matches_linear_kf_statistically(self):
        """Large-N EnKF posterior mean tracks the exact KF on a linear model."""
        model = _cv_model()
        kf, enkf = LinearKalmanEngine(), EnsembleKalmanEngine(n_members=2000, seed=1)
        enkf.reset(model)
        rng = np.random.default_rng(0)
        x, P = X0.copy(), P0.copy()
        truth = np.array([[0.0], [1.0]])
        for k in range(10):
            truth = F_CV @ truth
            z = (H_POS @ truth + rng.normal(0, 0.5, (1, 1))).ravel()
            x, P, _ = kf._predict(model, x, P, t=k * DT)
            x, P = kf._update(model, x, P, z, t=k * DT)
            xe, Pe, _ = enkf._predict(model, None, None, t=k * DT)
            xe, Pe = enkf._update(model, None, None, z, t=k * DT)
        self.assertLess(np.abs(x - xe).max(), 0.05)
        self.assertLess(np.abs(P - Pe).max(), 0.05)
        eig = np.linalg.eigvalsh(Pe)
        self.assertTrue(np.all(eig > -1e-10))

    def test_nan_measurement_skipped(self):
        model = _cv_model()
        e = EnsembleKalmanEngine(n_members=50, seed=2)
        e.reset(model)
        e._predict(model, None, None, t=0.0)
        before = e.members.copy()
        x, P = e._update(model, None, None, np.array([np.nan]), t=0.0)
        np.testing.assert_array_equal(e.members, before)     # all-NaN → no update
        self.assertTrue(np.all(np.isfinite(x)) and np.all(np.isfinite(P)))

    def test_inflation_widens_prior(self):
        model = _cv_model()
        e1 = EnsembleKalmanEngine(n_members=200, inflation=1.0, seed=3)
        e2 = EnsembleKalmanEngine(n_members=200, inflation=1.5, seed=3)
        for e in (e1, e2):
            e.reset(model)
            e._predict(model, None, None, t=0.0)
            e._update(model, None, None, np.array([0.5]), t=0.0)
        self.assertGreater(e2.spread(), e1.spread())

    def test_engine_owned_rng_stream(self):
        """Two same-seeded engines give identical results; the global stream is untouched."""
        model = _cv_model()
        state = np.random.get_state()[1].copy()
        runs = []
        for _ in range(2):
            e = EnsembleKalmanEngine(n_members=64, seed=42)
            e.reset(model)
            e._predict(model, None, None, t=0.0)
            x, _ = e._update(model, None, None, np.array([0.3]), t=0.0)
            runs.append(x)
        np.testing.assert_array_equal(runs[0], runs[1])
        np.testing.assert_array_equal(state, np.random.get_state()[1])

    def test_smooth_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            EnsembleKalmanEngine()._smooth()


# ── EnsembleKalmanFilter workflow ────────────────────────────────────────────

class TestEnsembleFilter(unittest.TestCase):

    def test_forward_record(self):
        model = _cv_model()
        rec_truth = model.sample(2.0)
        enkf = EnsembleKalmanFilter(_cv_model(), n_members=300, seed=5)
        rec = enkf.forward(rec_truth.zk)
        xk = np.asarray(rec.xk)
        self.assertEqual(xk.shape[0], len(np.asarray(rec_truth.zk)))
        self.assertEqual(xk.shape[1], 2)
        err = xk[:, 0] - np.asarray(rec_truth.xk)[:, 0]
        self.assertLess(np.sqrt(np.mean(err[5:] ** 2)), 0.5)  # tracks position

    def test_backward_raises(self):
        with self.assertRaises(NotImplementedError):
            EnsembleKalmanFilter(_cv_model()).backward()


# ── KolmogorovFlowModel ──────────────────────────────────────────────────────

class TestKolmogorovFlowModel(unittest.TestCase):

    def setUp(self):
        self.model = _tiny_flow()

    def test_dimensions(self):
        m = self.model
        self.assertEqual(m.n, 16 * 16)
        self.assertEqual(m.m, 40)
        self.assertEqual(np.asarray(m.R()).shape, (40, 40))

    def test_f_h_shapes(self):
        m = self.model
        w = m.solver.random_ic().ravel()
        w2 = m.f(w)
        self.assertEqual(w2.shape, w.shape)
        self.assertFalse(np.allclose(w, w2))                 # dynamics actually move
        z = m.h(w2)
        self.assertEqual(z.shape, (40, 1))
        np.testing.assert_allclose(z.ravel(), w2[m.obs_idx])

    def test_forecast_batch_matches_f(self):
        m = self.model
        X = m.initial_ensemble(3, rng=0)
        Xf = m.forecast(X)
        np.testing.assert_allclose(Xf[:, 1], m.f(X[:, 1].copy()), atol=1e-10)

    def test_surrogate_plug_in(self):
        m = self.model
        m.surrogate = lambda X: X * 0.5                      # any columns→columns map
        X = np.ones((m.n, 2))
        np.testing.assert_allclose(m.forecast(X), 0.5 * np.ones((m.n, 2)))
        m.surrogate = None

    def test_localization_shapes(self):
        rho_xy, rho_yy = self.model.localization()
        self.assertEqual(rho_xy.shape, (256, 40))
        self.assertEqual(rho_yy.shape, (40, 40))
        np.testing.assert_allclose(np.diag(rho_yy), 1.0)

    def test_jacobian_is_identity_placeholder(self):
        np.testing.assert_array_equal(self.model.F(), np.eye(self.model.n))
        self.assertNotIn("extended", [str(k) for k in self.model.ALLOWED_ENGINES])

    def test_H_selection_matrix(self):
        H = self.model.H()
        w = self.model.solver.random_ic().ravel()
        np.testing.assert_allclose(H @ w, w[self.model.obs_idx])


# ── Spatial gradients: operator, gradient observations, diagnostic ───────────

class TestFieldGradients(unittest.TestCase):

    def test_spectral_gradient_matches_analytic(self):
        m = _tiny_flow()
        g = m.grid
        x = 2 * np.pi * np.arange(g) / g
        X, Y = np.meshgrid(x, x)
        field = np.sin(3 * X) * np.cos(2 * Y)
        gx, gy = m.vorticity_gradient(field.ravel())
        np.testing.assert_allclose(gx, 3 * np.cos(3 * X) * np.cos(2 * Y), atol=1e-10)
        np.testing.assert_allclose(gy, -2 * np.sin(3 * X) * np.sin(2 * Y), atol=1e-10)

    def test_gradient_is_linear_and_batched(self):
        m = _tiny_flow()
        a = m.solver.random_ic().ravel(); b = m.solver.random_ic().ravel()
        gxa, _ = m.vorticity_gradient(a); gxb, _ = m.vorticity_gradient(b)
        gxab, _ = m.vorticity_gradient((a + 2 * b))
        np.testing.assert_allclose(gxab, gxa + 2 * gxb, atol=1e-9)     # linearity
        bx, by = m.vorticity_gradient(np.stack([a, b], axis=1))        # (n, 2)
        np.testing.assert_allclose(bx[:, 0], gxa.ravel(), atol=1e-9)

    def test_velocity_is_divergence_free(self):
        m = _tiny_flow()
        u, v = m.velocity(m.solver.random_ic())
        # spectral divergence ∂u/∂x + ∂v/∂y ≈ 0 (incompressible)
        div_x, _ = m.vorticity_gradient(u.ravel())
        _, div_y = m.vorticity_gradient(v.ravel())
        self.assertLess(np.abs(div_x + div_y).max(), 1e-8)

    def test_gradient_observation_model(self):
        m = KolmogorovFlowModel("grad", grid=16, cycle=8, dt=0.01, nu=1/40.,
                                n_obs=30, obs_type="gradient", loc_radius=4.0, seed=2)
        self.assertEqual(m.m, 60)                                       # 2 × 30 probes
        self.assertEqual(m.H().shape, (60, 256))
        rho_xy, rho_yy = m.localization()
        self.assertEqual(rho_xy.shape, (256, 60))
        self.assertEqual(rho_yy.shape, (60, 60))
        w = m.solver.random_ic().ravel()
        gx, gy = m.vorticity_gradient(w)
        expect = np.concatenate([gx.ravel()[m.obs_idx], gy.ravel()[m.obs_idx]])
        np.testing.assert_allclose(m.h(w).ravel(), expect, atol=1e-10)
        with self.assertRaises(ValueError):
            KolmogorovFlowModel("bad", grid=16, obs_type="curl")

    def test_gradient_obs_osse_beats_free(self):
        m = KolmogorovFlowModel("grad", grid=16, cycle=8, dt=0.01, nu=1/40.,
                                n_obs=40, obs_type="gradient", sig_o=0.5,
                                loc_radius=4.0, seed=7)
        m.climatology(n_snapshots=40, spinup=400, stride=5)
        osse = TwinExperiment(m, n_cycles=10, n_members=12)
        osse.add_method("enkf-grad", inflation=1.08)
        osse.add_free_run()
        hist = osse.run(verbose=False)
        self.assertLess(np.mean(hist["enkf-grad"]["rmse_a"][-3:]),
                        np.mean(hist["free"]["rmse_a"][-3:]))

    def test_gradient_diagnostic_uncertainty(self):
        m = _tiny_flow()
        ens = m.initial_ensemble(12, rng=0)
        d = m.gradient_diagnostic(ens)
        for key in ("gx_mean", "gy_mean", "mag_mean", "mag_sigma"):
            self.assertEqual(d[key].shape, (m.grid, m.grid))
        self.assertTrue(np.all(d["mag_sigma"] >= 0))
        self.assertGreater(d["mag_sigma"].mean(), 0)     # ensemble spread in ∇ω


# ── TwinExperiment (OSSE) ────────────────────────────────────────────────────

class TestTwinExperiment(unittest.TestCase):

    def test_enkf_beats_free_run(self):
        model = _tiny_flow()
        osse = TwinExperiment(model, n_cycles=10, n_members=10)
        osse.add_method("enkf", inflation=1.05)
        osse.add_free_run()
        hist = osse.run(verbose=False)
        enkf_tail = np.mean(hist["enkf"]["rmse_a"][-3:])
        free_tail = np.mean(hist["free"]["rmse_a"][-3:])
        self.assertLess(enkf_tail, free_tail)
        self.assertLess(enkf_tail, hist["enkf"]["rmse_a"][0])   # error shrinks

    def test_skill_ratio(self):
        model = _tiny_flow()
        osse = TwinExperiment(model, n_cycles=6, n_members=8)
        osse.add_method("enkf", inflation=1.05)
        osse.add_free_run()
        osse.run(verbose=False)
        sk = osse.skill(reference="free")
        self.assertEqual(set(sk), {"enkf"})
        self.assertEqual(len(sk["enkf"]), 6)
        self.assertLess(sk["enkf"][-1], 1.0)                 # beats the free run
        # cross-run reference: pass another history dict directly
        sk2 = osse.skill(reference=osse.hist["enkf"])
        np.testing.assert_allclose(sk2["enkf"], 1.0)

    def test_async_observation_schedule(self):
        model = _tiny_flow()
        osse = TwinExperiment(model, n_cycles=9, n_members=8, obs_periods=(1, 3))
        self.assertEqual(osse.obs_mask.shape, (9, model.m))
        self.assertEqual(len(osse.obs_periods), model.m)
        # every station reports once per its own period (phase shifts by ±1)
        for j, per in enumerate(osse.obs_periods):
            self.assertTrue(np.floor(9 / per) <= osse.obs_mask[:, j].sum() <= np.ceil(9 / per))
        # deterministic given seed_setup
        osse2 = TwinExperiment(model, n_cycles=9, n_members=8, obs_periods=(1, 3))
        np.testing.assert_array_equal(osse.obs_mask, osse2.obs_mask)
        # filter still beats the free run on the sparse multirate network
        osse.add_method("enkf", inflation=1.05)
        osse.add_free_run()
        hist = osse.run(verbose=False)
        self.assertLess(np.mean(hist["enkf"]["rmse_a"][-3:]),
                        np.mean(hist["free"]["rmse_a"][-3:]))

    def test_per_method_q_std(self):
        model = _tiny_flow()
        osse = TwinExperiment(model, n_cycles=2, n_members=6)
        osse.add_method("plain", inflation=1.05)
        eng_q = osse.add_method("with_q", inflation=1.05, q_std=0.5)
        Q = osse.methods["with_q"]["view"].Q()
        np.testing.assert_allclose(np.diag(Q), 0.25)
        self.assertFalse(osse.methods["plain"]["view"] is osse.methods["with_q"]["view"])
        hist = osse.run(verbose=False)
        # extra process noise must widen the ensemble spread
        self.assertGreater(hist["with_q"]["spread"][-1], hist["plain"]["spread"][-1])

    def test_record_fields_and_animate(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        model = _tiny_flow()
        osse = TwinExperiment(model, n_cycles=3, n_members=6)
        osse.add_method("enkf", inflation=1.05)
        osse.add_free_run()
        with self.assertRaises(RuntimeError):
            osse.run(verbose=False) and osse.animate()       # no fields recorded
        osse.run(verbose=False, record_fields=True)
        self.assertEqual(len(osse.fields["truth"]), 3)
        self.assertEqual(len(osse.fields["enkf"]["analysis"]), 3)
        self.assertEqual(osse.fields["enkf"]["forecast"][0].shape, (model.n,))
        anim = osse.animate(interval=50)                     # FuncAnimation outside Jupyter
        self.assertIsNotNone(anim)

    def test_rng_discipline_invariance(self):
        """Adding a second method must not change the first method's results."""
        def run(extra):
            model = _tiny_flow()
            osse = TwinExperiment(model, n_cycles=5, n_members=8)
            osse.add_method("A", inflation=1.05)
            if extra:
                osse.add_method("B", inflation=1.3)
                osse.add_free_run()
            return osse.run(verbose=False)["A"]["rmse_a"]
        np.testing.assert_array_equal(run(False), run(True))


# ── Truth replay (imperfect-model / reanalysis mode) ─────────────────────────

class TestTruthReplay(unittest.TestCase):

    def test_replay_beats_free_run_under_model_error(self):
        """Truth from different physics (nature run), forecasts from the model."""
        model = _tiny_flow()
        nature = KolmogorovFlowModel("nature", grid=16, cycle=10, dt=0.01,
                                     nu=1 / 80., n_obs=40, obs_idx=model.obs_idx,
                                     sig_o=0.30, loc_radius=4.0, seed=7)
        w, traj = model.equilibrium_state(decorrelate=50), []
        for _ in range(8):
            w = nature.f(w)
            traj.append(w.copy())
        osse = TwinExperiment(model, n_cycles=8, n_members=8,
                              truth_trajectory=np.asarray(traj))
        osse.add_method("enkf", inflation=1.10)
        osse.add_free_run()
        hist = osse.run(verbose=False)
        self.assertLess(np.mean(hist["enkf"]["rmse_a"][-3:]),
                        np.mean(hist["free"]["rmse_a"][-3:]))
        np.testing.assert_allclose(osse.truth, traj[-1])     # replayed, not re-evolved

    def test_replay_clamps_cycles_and_validates_shape(self):
        model = _tiny_flow()
        traj = model.initial_ensemble(4, rng=0).T            # (4, n): 4 cycles worth
        osse = TwinExperiment(model, n_cycles=99, n_members=6, truth_trajectory=traj)
        self.assertEqual(osse.n_cycles, 4)
        grid_shaped = traj.reshape(4, model.grid, model.grid)  # (T, g, g) accepted too
        self.assertEqual(TwinExperiment(model, n_members=6,
                                        truth_trajectory=grid_shaped).n_cycles, 4)
        with self.assertRaises(ValueError):
            TwinExperiment(model, n_members=6, truth_trajectory=traj[:, :100])


# ── Reanalysis chain (numpy core; file I/O needs optional xarray) ────────────

class TestReanalysis(unittest.TestCase):

    def test_vorticity_solid_body_rotation(self):
        """u = Ω a cosφ, v = 0  →  ζ = 2 Ω sinφ (classic identity)."""
        from modules.apps.weather.geo import vorticity_from_uv, EARTH_RADIUS
        lat = np.linspace(-60, 60, 121)
        lon = np.linspace(0, 359, 360)
        omega = 7.29e-5
        u = omega * EARTH_RADIUS * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
        v = np.zeros_like(u)
        zeta = vorticity_from_uv(u, v, lat, lon)
        expected = 2 * omega * np.sin(np.deg2rad(lat))[:, None] * np.ones_like(u)
        np.testing.assert_allclose(zeta[2:-2], expected[2:-2], rtol=1e-3, atol=1e-8)

    def test_regrid_shapes_and_constants(self):
        from modules.apps.weather.geo import regrid
        fields = np.ones((5, 24, 48))
        out = regrid(fields, 16)
        self.assertEqual(out.shape, (5, 16, 16))
        np.testing.assert_allclose(out, 1.0)
        self.assertEqual(regrid(fields[0], 16).shape, (16, 16))   # single frame

    def test_rescale_and_full_chain(self):
        from modules.apps.weather.geo import rescale_to_model, reanalysis_truth
        model = _tiny_flow()
        rng = np.random.default_rng(3)
        fields = 1e-5 * rng.standard_normal((6, 24, 48)) + 3e-5   # SI-ish vorticity
        scaled, scale = rescale_to_model(fields, model)
        self.assertAlmostEqual(scaled.std(), model.climatology().std(), places=10)
        traj = reanalysis_truth(fields, model, stride=2)
        self.assertEqual(traj.shape, (3, model.n))
        # (u, v, lat, lon) source and replay integration
        lat = np.linspace(-30, 30, 24); lon = np.linspace(0, 359, 48)
        u = rng.standard_normal((6, 24, 48)); v = rng.standard_normal((6, 24, 48))
        traj2 = reanalysis_truth((u, v, lat, lon), model)
        osse = TwinExperiment(model, n_members=6, truth_trajectory=traj2)
        self.assertEqual(osse.n_cycles, 6)


# ── Earth representation (land/sea mask, global plot) ────────────────────────

class TestEarthMap(unittest.TestCase):

    def test_geometry_loads(self):
        from modules.apps.weather.geo import earth_geometry
        land = earth_geometry("land")
        coast = earth_geometry("coastline")
        self.assertGreater(len(land), 100)
        self.assertGreater(len(coast), 100)
        allv = np.vstack(land)
        self.assertTrue((-180.1 <= allv[:, 0]).all() and (allv[:, 0] <= 180.1).all())
        self.assertTrue((-90.1 <= allv[:, 1]).all() and (allv[:, 1] <= 90.1).all())

    def test_land_mask_known_points(self):
        from modules.apps.weather.geo import land_mask
        lat = np.array([55.7, 23.0, -30.0, 40.0])       # Moscow, Sahara, S-Pacific, N-Atlantic
        lon = np.array([37.6, 10.0, -120.0, -40.0])
        m = land_mask(lat, lon)                          # (4, 4) grid; diagonal = the pairs
        self.assertTrue(m[0, 0])                         # Moscow: ground
        self.assertTrue(m[1, 1])                         # Sahara: ground
        self.assertFalse(m[2, 2])                        # South Pacific: sea
        self.assertFalse(m[3, 3])                        # North Atlantic: sea

    def test_plot_global_field_smoke(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from modules.apps.weather.geo import plot_global_field, band_coordinates
        fig, ax = plot_global_field()                    # plain planet: sea + ground
        self.assertIsNotNone(fig)
        lat = np.linspace(-90, 90, 46)
        lon = np.arange(0, 360, 4.0)
        field = np.random.default_rng(0).standard_normal((46, 90))
        geo = band_coordinates(_tiny_flow(), lat_range=(10, 50))
        fig2, ax2 = plot_global_field(field, lat, lon, central_longitude=180,
                                      band=(10, 50),
                                      probes=(geo["probe_lat"], geo["probe_lon"]))
        self.assertIsNotNone(fig2)
        import matplotlib.pyplot as plt
        plt.close("all")


# ── Kraichnan-turbulence chain (doubly-periodic Cartesian data) ──────────────

class TestTurbulenceTruth(unittest.TestCase):

    def test_vorticity_cartesian_analytic(self):
        """u = sin y → ζ = −cos y;  v = sin x → ζ = cos x (spectral, exact)."""
        from modules.apps.weather.geo import vorticity_cartesian
        n = 64
        x = 2 * np.pi * np.arange(n) / n
        X, Y = np.meshgrid(x, x)
        np.testing.assert_allclose(
            vorticity_cartesian(np.sin(Y), np.zeros((n, n))), -np.cos(Y), atol=1e-10)
        np.testing.assert_allclose(
            vorticity_cartesian(np.zeros((n, n)), np.sin(X)), np.cos(X), atol=1e-10)

    def test_coarsen_block_mean(self):
        from modules.apps.weather.geo import coarsen
        f = np.arange(2 * 8 * 8, dtype=float).reshape(2, 8, 8)
        out = coarsen(f, 4)
        self.assertEqual(out.shape, (2, 4, 4))
        np.testing.assert_allclose(out[0, 0, 0], f[0, :2, :2].mean())
        self.assertEqual(coarsen(np.ones((10, 10)), 4).shape, (4, 4))   # non-divisible → regrid

    def test_load_turbulence_layouts(self):
        import tempfile, os
        from modules.apps.weather.geo import load_turbulence
        rng = np.random.default_rng(0)
        arr = rng.standard_normal((4, 3, 16, 16))            # NSKT layout (T, C, H, W)
        np.testing.assert_array_equal(load_turbulence(arr, variable="vorticity"), arr[:, 2])
        u, v = load_turbulence(arr, variable="uv")
        np.testing.assert_array_equal(u, arr[:, 0])
        np.testing.assert_array_equal(v, arr[:, 1])
        with tempfile.TemporaryDirectory() as d:
            npz = os.path.join(d, "t.npz")
            np.savez(npz, vorticity=arr[:, 2])
            np.testing.assert_array_equal(load_turbulence(npz), arr[:, 2])
            try:
                import h5py
                h5 = os.path.join(d, "t.h5")
                with h5py.File(h5, "w") as f:
                    f.create_dataset("fields", data=arr)
                np.testing.assert_array_equal(load_turbulence(h5, variable="vorticity"), arr[:, 2])
            except ImportError:
                pass

    def test_high_res_truth_replay(self):
        """Kraichnan-style protocol in miniature: 32² nature run block-coarsened
        onto the 16² model grid, replayed as truth — EnKF must beat free run."""
        from modules.apps.weather.geo import turbulence_truth
        from modules.apps.weather.flow import NS2D
        model = _tiny_flow()
        hires = NS2D(n=32, nu=1 / 40., kf=4, chi=2.0, dt=0.01, seed=11)
        w = hires.advance(hires.random_ic(), 400)
        frames = []
        for _ in range(8):
            w = hires.advance(w, 10)
            frames.append(w.copy())
        traj = turbulence_truth(np.asarray(frames), model)
        self.assertEqual(traj.shape, (8, model.n))
        osse = TwinExperiment(model, n_members=8, truth_trajectory=traj)
        osse.add_method("enkf", inflation=1.10)
        osse.add_free_run()
        hist = osse.run(verbose=False)
        self.assertLess(np.mean(hist["enkf"]["rmse_a"][-3:]),
                        np.mean(hist["free"]["rmse_a"][-3:]))


# ── CNN surrogate (rollout loss) ─────────────────────────────────────────────

class TestFlowSurrogate(unittest.TestCase):

    def test_train_rollout_and_assimilate(self):
        try:
            from modules.apps.weather import surrogate as fs
        except ImportError:
            self.skipTest("torch not installed")
        model = _tiny_flow()
        traj = fs.make_trajectory(model, n_cycles=30, burn=200)
        self.assertEqual(traj.shape, (31, 16, 16))
        net, (mu, sd), val = fs.train_surrogate(
            traj, width=8, nblocks=1, epochs=5, rollout=2, batch=8, verbose=False)
        self.assertTrue(np.isfinite(val))
        forecast = fs.as_forecaster(net, mu, sd)
        X = model.initial_ensemble(3, rng=0)
        self.assertEqual(forecast(X).shape, X.shape)
        errs = fs.rollout_rmse(forecast, model, n_cycles=3)
        self.assertEqual(len(errs), 3)
        self.assertTrue(np.all(np.isfinite(errs)))
        # surrogate forecast wired into the OSSE harness
        osse = TwinExperiment(model, n_cycles=2, n_members=6)
        osse.add_method("surrogate", inflation=1.15, forecast=forecast)
        hist = osse.run(verbose=False)
        self.assertEqual(len(hist["surrogate"]["rmse_a"]), 2)

    def test_reanchored_forecast_alternates(self):
        try:
            from modules.apps.weather.surrogate import ReanchoredForecast
        except ImportError:
            self.skipTest("torch not installed")
        model = _tiny_flow()
        X = model.initial_ensemble(2, rng=0)
        marker = lambda Z: np.full_like(Z, 7.0)              # obviously-not-solver map
        hybrid = ReanchoredForecast(model, marker, every=3)
        self.assertTrue(np.all(hybrid(X.copy()) == 7.0))     # cycle 1: surrogate
        self.assertTrue(np.all(hybrid(X.copy()) == 7.0))     # cycle 2: surrogate
        anchored = hybrid(X.copy())                          # cycle 3: true solver
        np.testing.assert_allclose(anchored, model.forecast(X), atol=1e-10)
        self.assertTrue(np.all(hybrid.reset()(X.copy()) == 7.0))


# ── ParticleFilter on the flow model (allowed engine, smoke) ─────────────────

class TestParticleOnFlow(unittest.TestCase):

    def test_pf_runs_on_flow_model(self):
        model = _tiny_flow()
        w0 = model.equilibrium_state(decorrelate=50)
        model.initial_state(w0.reshape(-1, 1), 0.25 * np.eye(model.n))
        truth, zs = w0.copy(), []
        rng = np.random.default_rng(11)
        for _ in range(3):
            truth = model.f(truth)
            zs.append(truth[model.obs_idx] + rng.normal(0, model.sig_o, model.m))
        pf = ParticleFilter(model, n_particles=20)
        rec = pf.forward(np.asarray(zs))
        xk = np.asarray(rec.xk)
        self.assertEqual(xk.shape, (3, model.n))
        self.assertTrue(np.all(np.isfinite(xk)))


if __name__ == "__main__":
    unittest.main()

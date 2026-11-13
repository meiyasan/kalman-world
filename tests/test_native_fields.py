"""A field model's ensemble pass runs in C++ (libKalmanApps: Weather::KolmogorovFlow).

With the ensemble engine, ``KolmogorovFlowModel`` and ``BarotropicFlowModel``
hand the whole pass to their twin: the forecast of every member, the
observation of the ensemble, the localization tapers, the analysis and the
loop over the cycles. The draws stay numpy's (the engine's generator, in the
order the Python steps take them), so the C++ pass is compared here with the
Python reference on the same seed, member for member.

How close: the reference steps the field with numpy's FFT and the twin with
the library's; one solver step agrees to ~1e-16 relative, and the chaotic flow
grows that to ~1e-13 over the 1,200 steps of notebook 13 (measured, then held
by the analysis). The tolerances below are 1e-9 relative to the largest entry.

Skipped when libKalmanApps is not built.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.models.native import apps_library_names
from modules.framework.engines.ensemble import EnsembleKalmanEngine
from modules.framework.filters import EnsembleKalmanFilter
from modules.apps.weather.flow import KolmogorovFlowModel, TwinExperiment, NS2D
from modules.apps.weather.barotropic import BarotropicFlowModel

needs_apps = unittest.skipUnless(bool(apps_library_names()), "libKalmanApps is not built")
RTOL = 1e-9


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.nanmax(np.abs(a - b)) / max(1e-300, float(np.nanmax(np.abs(b)))))


def mini(obs_type="vorticity", q_std=0.0, **kw):
    """Notebook 13's particle-filter comparison (step 5), at its size: 16 x 16."""
    model = KolmogorovFlowModel("mini", grid=16, cycle=10, dt=0.01, nu=1 / 40., n_obs=40, sig_o=0.30,
                                loc_radius=4.0, seed=7, obs_type=obs_type, q_std=q_std, **kw)
    model.climatology(n_snapshots=60, spinup=600, stride=5)
    return model


def measurements(model, steps=12, seed=21):
    rng = np.random.default_rng(seed)
    w = model.equilibrium_state(decorrelate=100)
    zs = []
    for _ in range(steps):
        w = model.f(w)
        zs.append(np.asarray(model.h(w)).ravel() + rng.normal(0, model.sig_o, model.m))
    zs = np.asarray(zs)
    zs[3, ::4] = np.nan                       # some probes silent
    zs[5, :] = np.nan                         # and one cycle with none
    return zs


def filter_pass(make, engine_kw, reference, expect_twin=True):
    def go():
        model = make()
        zs = measurements(model)
        f = EnsembleKalmanFilter(model, **engine_kw)
        if not reference:
            assert (f.engine._native_twin(model) is not None) == expect_twin
        rec = f.forward(zs)
        return (np.asarray(rec.xk), np.asarray(rec.Pk), f.engine.members.copy(),
                int(f.engine.rng.integers(0, 1 << 62)), [np.asarray(nu).ravel() for nu in f.innovation])
    if reference:
        with native.python_reference():
            return go()
    return go()


@needs_apps
class TestFieldFilterPass(unittest.TestCase):

    def compare(self, make, **engine_kw):
        got, want = filter_pass(make, engine_kw, False), filter_pass(make, engine_kw, True)
        self.assertLess(rel(got[0], want[0]), RTOL, "posterior means")
        self.assertLess(rel(got[1], want[1]), RTOL, "posterior covariances")
        self.assertLess(rel(got[2], want[2]), RTOL, "final members")
        self.assertLess(rel(np.concatenate(got[4]), np.concatenate(want[4])), RTOL, "innovations")
        self.assertEqual(got[3], want[3], "the engine's generator is left where the Python pass leaves it")
        return got

    def test_vorticity_probes(self):
        got = self.compare(mini, n_members=16, inflation=1.05, seed=0)
        self.assertTrue(np.all(np.isfinite(got[0])))

    def test_gradient_probes_with_process_noise(self):
        self.compare(lambda: mini("gradient", q_std=0.05), n_members=12, inflation=1.02, seed=3)

    def test_tapers_given_to_the_engine(self):
        model = mini()
        rho_xy, rho_yy = model.localization()
        loc = (0.5 * rho_xy, rho_yy)              # not the model's: the engine's win
        self.compare(mini, n_members=10, localization=loc, seed=4)

    def test_barotropic_flow(self):
        # sig_o = 1e-5 1/s against fields of ~4.5e-5. At 1e-6 the innovation covariance
        # of 10 members is so ill-conditioned that numpy's pinv and the library's
        # Cholesky inverse part at 6e-8 (the C++ kernel the Python loop already used
        # does too; the C++ pass agrees with that loop to 8e-15).
        def make():
            model = BarotropicFlowModel("b", grid=16, cycle=6, n_obs=30, sig_o=1e-5, seed=7)
            model.climatology(n_snapshots=40, spinup=60, stride=3)
            return model
        self.compare(make, n_members=10, inflation=1.03, seed=1)

    def test_the_twin_declares_what_python_does(self):
        model = mini("gradient")
        handle = model.native_handle()
        x = model.climatology()[:, 3].copy()
        h = np.empty(model.m)
        core.check(core.lib.kalman_model_h(handle, core.cdptr(np.ascontiguousarray(x)), 0.0, core.dptr(h)), "h")
        np.testing.assert_allclose(h, np.asarray(model.h(x)).ravel(), rtol=0, atol=1e-12)
        H = np.empty((model.m, model.n))
        core.check(core.lib.kalman_model_H(handle, 0.0, core.dptr(H)), "H")
        np.testing.assert_allclose(H @ x, h, rtol=0, atol=1e-12)          # H is the map h is
        f = np.empty(model.n)
        core.check(core.lib.kalman_model_f(handle, core.cdptr(np.ascontiguousarray(x)), core.ffi.NULL, 0.0,
                                           core.dptr(f)), "f")
        np.testing.assert_allclose(f, model.f(x), rtol=0, atol=1e-12)


@needs_apps
class TestFieldFallbacks(unittest.TestCase):
    """What has no C++ counterpart keeps the Python pass, and still gives its answer."""

    def test_python_callables_and_other_solvers(self):
        model = mini()
        self.assertIsNotNone(model.native_handle())
        model.surrogate = lambda X: X
        self.assertIsNone(model.native_name(), "a surrogate forecast is Python")
        model.surrogate = None

        model.solver = NS2D(n=16, nu=1 / 40., kf=4, chi=2.0, dt=0.01, backend="torch", device="cpu")
        self.assertIsNone(model.native_name(), "libTorch is not linked by libKalmanApps")

        class Damped(NS2D):
            def rhs_hat(self, wh, t=None):
                return super().rhs_hat(wh, t) - 0.1 * wh
        model.solver = Damped(n=16, nu=1 / 40., kf=4, chi=2.0, dt=0.01)
        self.assertIsNone(model.native_name(), "other physics")

    def test_a_changed_hook_or_a_full_q(self):
        class Observed(KolmogorovFlowModel):
            def observe(self, X, t=None):
                return super().observe(X, t) * 1.0
        model = Observed("o", grid=16, cycle=4, n_obs=10, seed=1)
        self.assertIsNone(model.native_handle(), "an observe() of its own")

        model = KolmogorovFlowModel("q", grid=8, cycle=4, n_obs=10, seed=1)
        Q = 0.01 * np.eye(model.n)
        Q[0, 1] = Q[1, 0] = 0.001
        model._Q = Q
        self.assertIsNone(model.native_handle(), "a Q with off-diagonal terms")
        self.assertIsNone(EnsembleKalmanEngine(n_members=4)._native_twin(model))

    def test_a_python_pass_still_gives_the_reference(self):
        def make():
            model = mini()
            model.surrogate = lambda X: np.column_stack([model.solver.advance(x, model.cycle) for x in X.T])
            return model
        got = filter_pass(make, dict(n_members=8, seed=2), False, expect_twin=False)
        want = filter_pass(make, dict(n_members=8, seed=2), True)
        self.assertLess(rel(got[0], want[0]), RTOL)


@needs_apps
class TestTwinExperiment(unittest.TestCase):
    """Notebook 13's assimilation, reduced: 32 x 32, 8 cycles, the same seeds."""

    @staticmethod
    def experiment(mode):
        def go():
            model = KolmogorovFlowModel("probes", grid=32, nu=1 / 150., chi=2.0, kf=4, dt=0.005, cycle=10,
                                        n_obs=60, sig_o=0.30, loc_radius=6.0, seed=7)
            model.climatology(n_snapshots=40, spinup=300, stride=5)
            osse = TwinExperiment(model, n_cycles=8, n_members=12, seed_obs=999, seed_setup=123, obs_periods=(2, 6))
            osse.add_method("enkf", inflation=1.06)
            osse.add_method("enkf+Q", inflation=1.05, q_std=0.3)
            osse.add_free_run()
            hist = osse.run(verbose=False, record_fields=True)
            return hist, osse
        if mode == "reference":
            with native.python_reference():
                return go()
        if mode == "loop":                       # the C++ solver and kernel, the Python loop
            saved, native.USE_NATIVE_TWINS = native.USE_NATIVE_TWINS, False
            try:
                return go()
            finally:
                native.USE_NATIVE_TWINS = saved
        return go()

    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for mode in ("cxx", "loop", "reference"):
            hist, osse = cls.experiment(mode)
            members = {name: m["engine"].members.copy() for name, m in osse.methods.items()}
            cls.runs[mode] = (hist, osse.fields, members, osse.truth.copy(), osse)

    def test_every_method_agrees(self):
        got, fields, members, truth, _ = self.runs["cxx"]
        for other in ("loop", "reference"):
            want, ref_fields, ref_members, ref_truth, _ = self.runs[other]
            for name in ("enkf", "enkf+Q", "free"):
                for key in ("rmse_f", "rmse_a", "spread"):
                    self.assertEqual(len(got[name][key]), 8)
                    self.assertLess(rel(got[name][key], want[name][key]), RTOL, f"{other} {name} {key}")
                for key in ("forecast", "analysis"):
                    self.assertLess(rel(fields[name][key], ref_fields[name][key]), RTOL, f"{other} {name} {key}")
                self.assertLess(rel(members[name], ref_members[name]), RTOL)
            np.testing.assert_allclose(truth, ref_truth, rtol=0, atol=1e-9)

    def test_a_free_run_is_its_forecasts(self):
        hist = self.runs["cxx"][0]["free"]
        self.assertEqual(hist["rmse_f"], hist["rmse_a"])

    def test_a_callback_keeps_the_python_loop(self):
        seen = []
        osse = self.runs["loop"][-1]
        osse.run(verbose=False, callback=lambda c, truth, hist: seen.append(len(hist["enkf"]["rmse_a"])))
        self.assertEqual(seen, list(range(1, 9)))       # the history grows cycle by cycle


if __name__ == "__main__":
    unittest.main()

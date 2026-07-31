"""Tests for the kalman.api facade: introspection, execution, KalmanSpec
authoring/scaffold, and scoped-JWT auth."""
import importlib.util
import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, ".")
import modules.api as api
from modules.api import auth

CV = {"base": "matrices", "sensor_id": "cv", "fs": 10,
      "dynamics": {"F": [[1, 0.1], [0, 1]], "H": [[1, 0]]},
      "noise": {"Q": [[1e-3, 0], [0, 1e-3]], "R": [[0.25]]},
      "initial": {"x0": [[0], [1]]}}


class TestIntrospection(unittest.TestCase):

    def test_catalog(self):
        cat = api.catalog()
        self.assertIn("OscillatorModel", cat["models"])
        self.assertIn("rk4", cat["integrators"])
        self.assertIn("spectral", cat["solvers"])
        self.assertIn("ensemble", cat["engines"])
        self.assertGreater(len(cat["controllers"]), 0)
        # every app domain is reachable by name, calorimeter included
        self.assertIn("CalorimeterLinearModel", cat["models"])

    def test_registered_kinds_appear_in_the_catalog(self):
        from modules.framework import (KalmanIntegrator, register_integrator,
                                       unregister_integrator)

        class _Noop(KalmanIntegrator):
            def step(self, f, x, u=None, t=None, dt=1.0): return x

        register_integrator("catalog_probe", _Noop)
        try:
            self.assertIn("catalog_probe", api.catalog()["integrators"])
        finally:
            unregister_integrator("catalog_probe")
        self.assertNotIn("catalog_probe", api.catalog()["integrators"])

    def test_describe(self):
        d = api.describe("OscillatorModel")
        self.assertEqual(d["kind"], "model")
        self.assertEqual(d["default_integrator"], "rk4")
        self.assertIn("default_engine", d)
        with self.assertRaises(KeyError):
            api.describe("NopeModel")


class TestExecute(unittest.TestCase):

    def test_spec_roundtrip_filter(self):
        truth = api.sample(CV, 2.0)
        self.assertEqual(np.array(truth["xk"]).shape, (20, 2))
        res = api.run_filter(CV, truth["zk"])
        self.assertEqual(np.array(res["xk"]).shape, (20, 2))
        self.assertIsInstance(res["xk"][0][0], float)          # JSON-native
        d = api.diagnostics(res["run_id"])
        self.assertEqual(len(d["nis"]), 20)

    def test_run_control_cost(self):
        spec = {"base": "matrices", "sensor_id": "c", "fs": 10,
                "dynamics": {"F": [[1, 0.1], [0, 1]], "H": [[1, 0]], "B": [[0], [0.1]]},
                "noise": {"Q": [[1e-3, 0], [0, 1e-3]], "R": [[0.25]]},
                "initial": {"x0": [[1.0], [0.0]]}}
        out = api.run_control(spec, {"controller": "lqr",
                                     "cost": {"Qcost": [[10, 0], [0, 1]], "Rcost": [[1]]}}, 3.0)
        self.assertIn("cost", out)
        self.assertGreater(out["cost"], 0.0)
        self.assertEqual(len(out["cost_log"]), 30)

    def test_run_osse_small(self):
        spec = {"base": "KolmogorovFlowModel",
                "dynamics": {"params": {"grid": 16, "cycle": 8, "dt": 0.01,
                                        "nu": 1/40., "n_obs": 30, "loc_radius": 4.0, "seed": 3}}}
        m = api.make_model(spec)
        m.climatology(n_snapshots=30, spinup=300, stride=5)     # prime cheaply
        # run_osse rebuilds the model; prime that one too by monkegreasing catalog spec
        out = api.run_osse(spec, n_cycles=3, n_members=8)
        self.assertIn("enkf", out["hist"])
        self.assertEqual(len(out["hist"]["enkf"]["rmse_a"]), 3)


class TestAuthoring(unittest.TestCase):

    def setUp(self):
        api._SESSION_MODELS.clear()

    def test_register_and_run_by_name(self):
        info = api.register_model(CV, name="my_cv")
        self.assertEqual((info["n"], info["m"]), (2, 1))
        self.assertIn("my_cv", api.catalog()["authored"])
        truth = api.sample(CV, 1.0)
        res = api.run_filter({"base": "my_cv"}, truth["zk"])
        self.assertEqual(np.array(res["xk"]).shape[1], 2)

    def test_rejects_bad_specs(self):
        with self.assertRaises(ValueError):
            api.register_model({"base": "nope"})
        with self.assertRaises(ValueError):
            api.register_model({"base": "matrices", "dynamics": {"H": [[1, 0]]},
                                "noise": {"Q": [[1, 0], [0, 1]], "R": [[-1]]},
                                "initial": {"x0": [[0], [0]]}})     # non-PSD R
        with self.assertRaises(ValueError):
            api.register_model({"base": "matrices", "dynamics": {"F": "os.system"},
                                "initial": {"x0": [[0]]}})          # code-bearing

    def test_save_reload(self):
        api.register_model(CV, name="rt")
        with tempfile.TemporaryDirectory() as d:
            path = api.save_spec("rt", d)
            self.assertTrue(os.path.exists(path))
            api._SESSION_MODELS.clear()
            api.load_spec(path)
            self.assertIn("rt", api.catalog()["authored"])

    def test_scaffold_imports(self):
        src = api.scaffold(CV, "MyCV")
        self.assertIn("class MyCV(LinearKalmanModel)", src)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.py")
            api.write_scaffold(CV, p, "MyCV")
            sp = importlib.util.spec_from_file_location("m", p)
            mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
            inst = mod.MyCV()
            self.assertEqual((inst.n, inst.m), (2, 1))

    def test_no_eval_exec_in_source(self):
        import glob
        for f in glob.glob(os.path.join(os.path.dirname(api.__file__), "*.py")):
            src = open(f).read()
            self.assertNotIn("eval(", src, f"eval( in {f}")
            self.assertNotIn("exec(", src, f"exec( in {f}")


class TestAuth(unittest.TestCase):

    def setUp(self):
        try:
            import jwt  # noqa
        except ImportError:
            self.skipTest("pyjwt not installed")
        os.environ["KALMAN_JWT_SECRET"] = "test-secret-123"

    def tearDown(self):
        os.environ.pop("KALMAN_JWT_SECRET", None)

    def test_disabled_allows(self):
        os.environ.pop("KALMAN_JWT_SECRET", None)
        self.assertTrue(auth.authorize(None, "write_models"))     # no secret -> allowed

    def test_scoped_tokens(self):
        writer = auth.mint_token("alice", ["write_models"])
        reader = auth.mint_token("bob", ["read"])
        self.assertTrue(auth.authorize(writer, "write_models"))
        self.assertTrue(auth.authorize(reader, "read"))
        with self.assertRaises(auth.AuthError):
            auth.authorize(reader, "write_models")
        with self.assertRaises(auth.AuthError):
            auth.authorize(None, "read")                          # missing token
        with self.assertRaises(auth.AuthError):
            auth.authorize("garbage.token.here", "read")          # bad signature

    def test_admin_scope_grants_all(self):
        admin = auth.mint_token("root", ["admin"])
        self.assertTrue(auth.authorize(admin, "write_models"))
        self.assertTrue(auth.authorize(admin, "send_measurement"))

    def test_expired_denied(self):
        tok = auth.mint_token("x", ["read"], exp_hours=-1.0)      # already expired
        with self.assertRaises(auth.AuthError):
            auth.authorize(tok, "read")

    def test_register_requires_scope(self):
        reader = auth.mint_token("bob", ["read"])
        with self.assertRaises(auth.AuthError):
            api.register_model(CV, name="x", credential=reader)
        writer = auth.mint_token("alice", ["write_models"])
        api._SESSION_MODELS.clear()
        info = api.register_model(CV, name="ok", credential=writer)
        self.assertEqual(info["name"], "ok")

    def test_htpasswd_roundtrip(self):
        try:
            import bcrypt  # noqa
        except ImportError:
            self.skipTest("bcrypt not installed")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "htpasswd")
            auth.add_user("alice", "pw123", path)
            self.assertTrue(auth.verify_user("alice", "pw123", path))
            self.assertFalse(auth.verify_user("alice", "wrong", path))
            self.assertFalse(auth.verify_user("nobody", "pw123", path))
            # Apache htpasswd format: one user:hash line, owner-only perms
            line = open(path).read().strip()
            self.assertTrue(line.startswith("alice:$2"))
            self.assertEqual(oct(os.stat(path).st_mode)[-3:], "600")
            # updating a password replaces the line rather than appending
            auth.add_user("alice", "pw456", path)
            self.assertEqual(len(open(path).read().strip().splitlines()), 1)
            self.assertTrue(auth.verify_user("alice", "pw456", path))
            self.assertFalse(auth.verify_user("alice", "pw123", path))
            # a >72-byte password is truncated, not an error (bcrypt's limit)
            auth.add_user("bob", "x" * 200, path)
            self.assertTrue(auth.verify_user("bob", "x" * 200, path))


class TestPlotAndForecast(unittest.TestCase):

    def test_render_plot_works_for_a_declarative_spec(self):
        """A spec-built model has no domain state labels; the figure must still
        render (the error panel is dropped when no truth is stored)."""
        import matplotlib
        matplotlib.use("Agg", force=True)
        for run_id in (api.sample(CV, 1.0)["run_id"],
                       api.run_filter(CV, [[0.1 * i] for i in range(10)])["run_id"]):
            png = api.render_plot(run_id)
            self.assertGreater(len(png), 1000)
            self.assertEqual(png[:4], b"\x89PNG")
        with self.assertRaises(KeyError):
            api.render_plot("no-such-run")

    def test_synthetic_state_labels(self):
        """An unlabelled model reports one label per state/observable."""
        m = api.make_model(CV)
        self.assertEqual(len(list(m.states)), m.n)
        self.assertEqual(len(list(m.observables)), m.m)

    def test_forecast_propagates_state_and_covariance(self):
        out = api.forecast(CV, [[0.0], [1.0]], None, 4)
        xk, Pk = np.asarray(out["xk"]), np.asarray(out["Pk"])
        self.assertEqual(xk.shape, (4, 2, 1))
        self.assertEqual(Pk.shape, (4, 2, 2))
        # constant velocity: position advances by v*dt each step, uncertainty grows
        np.testing.assert_allclose(xk[:, 0, 0], [0.1, 0.2, 0.3, 0.4], atol=1e-9)
        self.assertGreater(Pk[-1][0, 0], Pk[0][0, 0])

    def test_forecast_uses_a_models_own_cone_when_it_has_one(self):
        m = api.make_model({"base": "StormModel", "fs": 1 / 3600.})
        out = api.forecast({"base": "StormModel", "fs": 1 / 3600.},
                           np.asarray(m.x0).tolist(), None, 3)
        for key in ("lon", "lat", "I"):
            self.assertIn(key, out)


if __name__ == "__main__":
    unittest.main()

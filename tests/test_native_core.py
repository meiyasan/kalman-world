"""The C++ core seen from Python: the C API contract, the factory, plugins, and a
C++ model driven by the Python engines, the API and its auth.

Everything that needs ``libKalman`` is skipped when it is not built
(``make build install``); the header/cdef contract test always runs.
"""
import os
import re
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core
from modules.framework.model import KalmanModel
from modules.framework.constants import KalmanEngineKind
import modules.api as api
from modules.api import auth

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(REPO, "include", "Kalman", "CAPI.h")
PLUGIN = "KalmanPlugin_example"

def importlib_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


_PROTO = re.compile(r"^\s*(?:[A-Za-z_][\w ]*\s*\*?\s*)\b((?:kalman|kf|pf|ukf|diag)_\w+)\s*\(", re.M)


def _prototypes(text: str) -> set:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)      # block comments
    text = re.sub(r"//.*", "", text)                        # line comments
    return set(_PROTO.findall(text))


class TestHeaderContract(unittest.TestCase):
    """``native/core.py``'s cdef is a copy of ``CAPI.h``; they must not drift."""

    def test_cdef_matches_header(self):
        with open(HEADER) as fh:
            header = _prototypes(fh.read())
        cdef = _prototypes(core._CDEF)
        self.assertTrue(header, "no prototypes parsed from CAPI.h")
        self.assertEqual(header - cdef, set(), "in CAPI.h but missing from core._CDEF")
        self.assertEqual(cdef - header, set(), "in core._CDEF but not in CAPI.h")

    def test_info_struct_fields_match(self):
        with open(HEADER) as fh:
            header = fh.read()
        def fields(text):
            body = re.search(r"typedef struct kalman_model_info\s*\{(.*?)\}", text, re.S).group(1)
            body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
            return [ln.strip().rstrip(";").split()[-1].lstrip("*") for ln in body.splitlines() if ln.strip()]
        self.assertEqual(fields(header), fields(core._CDEF))


class TestPluginPolicy(unittest.TestCase):
    """What may be loaded is decided here, in Python, by directory — not by the caller."""

    def test_names_only_resolve_inside_allowed_dirs(self):
        with self.assertRaises(ValueError):
            core.resolve_plugin("")
        with self.assertRaises(ValueError):
            core.resolve_plugin("bad name with spaces")
        with self.assertRaises(FileNotFoundError):
            core.resolve_plugin("no-such-plugin-anywhere")

    def test_paths_outside_the_allow_list_are_refused(self):
        outside = os.path.abspath(__file__)                 # exists, but not a plugin dir
        with self.assertRaises(PermissionError):
            core.resolve_plugin(outside)
        with self.assertRaises(FileNotFoundError):
            core.resolve_plugin("/definitely/not/here" + core._EXT)

    def test_env_replaces_the_default_dirs(self):
        old = os.environ.get("KALMAN_PLUGIN_PATH")
        os.environ["KALMAN_PLUGIN_PATH"] = os.pathsep.join(["/tmp/a", "/tmp/b"])
        try:
            self.assertEqual(core.plugin_dirs(), [os.path.realpath("/tmp/a"), os.path.realpath("/tmp/b")])
        finally:
            if old is None:
                os.environ.pop("KALMAN_PLUGIN_PATH", None)
            else:
                os.environ["KALMAN_PLUGIN_PATH"] = old


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built — skipping C++ core tests")
class TestFactory(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loaded = api.load_plugin(PLUGIN)

    def test_library_and_version(self):
        self.assertTrue(os.path.isfile(core.LIBRARY_PATH))
        self.assertRegex(core.version(), r"^\d+\.\d+")

    def test_plugin_registered_its_models(self):
        self.assertIn("constant-velocity", core.names())
        self.assertIn("pendulum", core.names())
        self.assertIn(self.loaded["plugin"], core.plugins())
        # loading again is idempotent and still reports what the library owns
        again = api.load_plugin(PLUGIN)
        self.assertEqual(sorted(again["models"]), sorted(self.loaded["models"]))

    def test_describe(self):
        d = core.describe("constant-velocity")
        self.assertEqual((d["n"], d["m"], d["l"]), (2, 1, 0))
        self.assertEqual(d["title"], "Constant Velocity")
        self.assertEqual(d["engine"], 0)                    # Kalman::EngineKind::Linear
        self.assertTrue(d["analytic"])
        self.assertTrue(d["plugin"].endswith(core._EXT))
        with self.assertRaises(RuntimeError):
            core.describe("no-such-model")

    def test_errors_are_exceptions_not_crashes(self):
        with self.assertRaises(RuntimeError) as cm:
            core.load_plugin("/definitely/not/a/library" + core._EXT)
        self.assertIn("library", str(cm.exception))


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built — skipping C++ core tests")
class TestNativeModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        api.load_plugin(PLUGIN)
        from modules.framework.models.native import NativeModel
        cls.NativeModel = NativeModel

    def test_is_a_kalman_model_with_the_cxx_dimensions(self):
        m = self.NativeModel("constant-velocity", "cv")
        self.assertIsInstance(m, KalmanModel)
        self.assertEqual((m.n, m.m, m.l), (2, 1, 0))
        self.assertEqual(m.fs, 100.0)
        self.assertIs(m.DEFAULT_ENGINE, KalmanEngineKind.LINEAR)
        self.assertEqual(m.x0.shape, (2, 1))
        self.assertEqual(m.x0[1, 0], 1.0)                   # the C++ constructor's x0
        with self.assertRaises(KeyError):
            self.NativeModel("no-such-model")

    def test_coefficients_come_from_cxx(self):
        m = self.NativeModel("constant-velocity")
        np.testing.assert_allclose(m.F(), [[1, 0.01], [0, 1]])
        np.testing.assert_allclose(m.H(), [[1, 0]])
        Q = m.Q()
        self.assertEqual(Q.shape, (2, 2))
        np.testing.assert_allclose(Q, Q.T)
        self.assertGreater(Q[0, 1], 0.0)                    # the off-diagonal is the point
        np.testing.assert_allclose(m.R(), [[0.04]])
        self.assertIsNone(m.B())
        self.assertIsNone(m.L())

    def test_maps_agree_with_the_matrices(self):
        m = self.NativeModel("constant-velocity")
        x = np.array([[1.0], [2.0]])
        np.testing.assert_allclose(m.f(x), m.F() @ x)
        np.testing.assert_allclose(m.h(x), m.H() @ x)
        self.assertEqual(m.f(x).shape, (2, 1))
        self.assertEqual(m.f(x.ravel()).shape, (2,))        # shape follows the input
        w = np.array([0.5, -0.5])
        np.testing.assert_allclose(m.f(x.ravel(), w=w), (m.F() @ x).ravel() + w)

    def test_overrides_are_pushed_into_the_cxx_instance(self):
        m = self.NativeModel("constant-velocity", fs=10.0, x0=[3.0, -1.0], P0=np.eye(2) * 4)
        self.assertEqual(m.fs, 10.0)
        np.testing.assert_allclose(m.F(), [[1, 0.1], [0, 1]])   # dt followed fs down
        np.testing.assert_allclose(m.x0.ravel(), [3.0, -1.0])
        np.testing.assert_allclose(m.P0, np.eye(2) * 4)
        self.assertEqual(m.describe_native()["fs"], 100.0)   # a fresh instance keeps the C++ default

    def test_nonlinear_model_uses_numerical_jacobian(self):
        p = self.NativeModel("pendulum")
        self.assertIs(p.DEFAULT_ENGINE, KalmanEngineKind.EXTENDED)
        x = np.array([[0.3], [0.0]])
        F = p.F(x)
        self.assertEqual(F.shape, (2, 2))
        # d(theta_next)/d(omega) ~ dt for a small step; d(omega_next)/d(theta) ~ -omega^2 cos(theta) dt
        self.assertAlmostEqual(F[0, 1], 1.0 / p.fs, places=4)
        self.assertLess(F[1, 0], 0.0)
        # f is genuinely nonlinear: f(2x) != 2 f(x)
        self.assertFalse(np.allclose(p.f(2 * x), 2 * p.f(x)))

    def test_filters_with_the_python_engines(self):
        np.random.seed(0)
        m = self.NativeModel("constant-velocity", "cv")
        sampled = m.sample(2.0)                             # truth xk + noisy zk, 100 Hz
        rec = m.filter(sampled.zk)
        self.assertEqual(len(rec.xk), 200)
        err = np.asarray(rec.xk)[:, 0].ravel() - np.asarray(sampled.xk)[:, 0].ravel()
        self.assertLess(np.sqrt(np.mean(err[50:] ** 2)), 0.2)   # better than the 0.2 m sensor

        p = self.NativeModel("pendulum", "pendulum")
        sampled = p.sample(2.0)
        rec = p.filter(sampled.zk)
        self.assertTrue(np.all(np.isfinite(np.asarray(rec.xk))))


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built — skipping C++ core tests")
class TestApiIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        api.load_plugin(PLUGIN)

    def test_native_models_are_in_the_catalogue(self):
        cat = api.catalog()
        self.assertIn("constant-velocity", cat["models"])
        self.assertIn("pendulum", cat["models"])
        self.assertIn("OscillatorModel", cat["models"])        # Python ones untouched
        self.assertEqual(cat["native"]["library"], core.LIBRARY_PATH)
        self.assertIn("pendulum", cat["native"]["models"])

    def test_describe_marks_the_backend(self):
        d = api.describe("pendulum")
        self.assertEqual(d["backend"], "native")
        self.assertEqual(d["kind"], "model")
        self.assertEqual(d["default_engine"], "extended")
        self.assertEqual((d["n"], d["m"]), (2, 1))
        self.assertTrue(d["plugin"].endswith(core._EXT))
        self.assertEqual(api.describe("OscillatorModel")["backend"], "python")

    def test_spec_with_a_native_base_runs_the_filter(self):
        spec = {"base": "constant-velocity", "sensor_id": "cv", "fs": 20,
                "initial": {"x0": [[0.0], [1.0]]}}
        sampled = api.sample(spec, 1.0)
        self.assertEqual(len(sampled["zk"]), 20)
        out = api.run_filter(spec, sampled["zk"])
        self.assertEqual(len(out["xk"]), 20)
        self.assertEqual(len(out["xk"][0]), 2)
        self.assertIn("run_id", out)

    def test_load_plugin_refuses_paths_outside_the_allow_list(self):
        with self.assertRaises(PermissionError):
            api.load_plugin(os.path.abspath(__file__))
        with self.assertRaises(FileNotFoundError):
            api.load_plugin("not-a-plugin")


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built — skipping C++ core tests")
class TestNativeEngine(unittest.TestCase):
    """The C++ estimation step behind the Python filter, and the fused pass."""

    @classmethod
    def setUpClass(cls):
        api.load_plugin(PLUGIN)
        from modules.framework.models.native import NativeModel
        from modules.framework.engines import NativeEngine
        from modules.framework.filter import KalmanFilter
        cls.NativeModel, cls.NativeEngine, cls.KalmanFilter = NativeModel, NativeEngine, KalmanFilter

    def test_engine_identity(self):
        e = self.NativeEngine("unscented", alpha=1e-2)
        self.assertIs(e.KIND, KalmanEngineKind.UNSCENTED)
        self.assertTrue(e.has_smoother)
        self.assertFalse(self.NativeEngine("particle", n_particles=50, seed=1).has_smoother)
        with self.assertRaises(ValueError):
            self.NativeEngine("no-such-engine")
        with self.assertRaises(TypeError):
            self.NativeEngine("linear", alpha=1.0)              # the linear engine takes no parameters

    def test_refuses_a_python_model(self):
        from modules.framework.model import KalmanModel
        py = KalmanModel("py", F=np.eye(2), H=np.eye(2)[:1], Q=np.eye(2) * 1e-3, R=np.eye(1) * 0.1, fs=10)
        with self.assertRaises(TypeError):
            self.KalmanFilter(py, engine=self.NativeEngine("linear")).forward(np.zeros((5, 1)))

    def test_cxx_step_matches_the_python_step_on_a_linear_model(self):
        np.random.seed(1)
        m = self.NativeModel("constant-velocity", "cv")
        zk = m.sample(2.0).zk
        native = self.KalmanFilter(m, engine=self.NativeEngine("linear")).forward(zk)
        python = self.KalmanFilter(m).forward(zk)                 # the Python linear engine, same C++ model
        np.testing.assert_allclose(np.asarray(native.xk), np.asarray(python.xk), atol=1e-10)
        np.testing.assert_allclose(np.asarray(native.Pk), np.asarray(python.Pk), atol=1e-10)

    def test_cxx_ekf_matches_the_python_ekf_on_the_pendulum(self):
        np.random.seed(2)
        p = self.NativeModel("pendulum", "pendulum")
        zk = p.sample(2.0).zk
        native = self.KalmanFilter(p, engine=self.NativeEngine("extended")).forward(zk)
        python = self.KalmanFilter(p).forward(zk)
        # The two Jacobians are both central differences, with slightly different
        # step conventions, so agreement is to 1e-5 rather than round-off.
        np.testing.assert_allclose(np.asarray(native.xk), np.asarray(python.xk), atol=1e-5)

    def test_fused_pass_equals_the_stepwise_pass(self):
        np.random.seed(3)
        p = self.NativeModel("pendulum", "pendulum")
        zk = p.sample(1.5).zk
        stepwise = self.KalmanFilter(p, engine=self.NativeEngine("extended")).forward(zk)
        fused = p.run_native(zk, engine="extended")
        np.testing.assert_allclose(fused["x_post"], np.asarray(stepwise.xk).reshape(len(zk), -1), atol=1e-12)
        self.assertEqual(fused["innovations"].shape, (len(zk), 1))
        self.assertEqual(fused["transitions"].shape, (len(zk), 2, 2))

    def test_filter_native_returns_a_record_and_smooths(self):
        np.random.seed(4)
        m = self.NativeModel("constant-velocity", "cv")
        sampled = m.sample(2.0)
        post = m.filter_native(sampled.zk, engine="linear")
        smooth = m.filter_native(sampled.zk, engine="linear", smooth=True)
        self.assertEqual(np.asarray(post.xk).shape[0], 200)
        err_post = np.asarray(post.xk)[:, 0].ravel() - np.asarray(sampled.xk)[:, 0].ravel()
        err_smooth = np.asarray(smooth.xk)[:, 0].ravel() - np.asarray(sampled.xk)[:, 0].ravel()
        self.assertLessEqual(np.sqrt(np.mean(err_smooth ** 2)), np.sqrt(np.mean(err_post ** 2)) + 1e-9)
        with self.assertRaises(RuntimeError):
            m.filter_native(sampled.zk, engine="particle", smooth=True, n_particles=50)   # no smoother

    def test_sample_set_engines_run_and_seed(self):
        np.random.seed(5)
        m = self.NativeModel("constant-velocity", "cv")
        zk = m.sample(1.0).zk
        a = m.run_native(zk, engine="particle", n_particles=200, seed=7)["x_post"]
        b = m.run_native(zk, engine="particle", n_particles=200, seed=7)["x_post"]
        c = m.run_native(zk, engine="ensemble", n_members=40, inflation=1.02, seed=7)["x_post"]
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(np.isfinite(a)) and np.all(np.isfinite(c)))

    def test_api_binding_selects_the_native_backend(self):
        spec = {"base": "pendulum", "sensor_id": "p", "fs": 50,
                "bindings": {"engine": "extended", "backend": "native"}}
        filt = api.make_filter(spec)
        self.assertEqual(type(filt.engine).__name__, "NativeEngine")
        sampled = api.sample(spec, 1.0)
        out = api.run_filter(spec, sampled["zk"])
        self.assertEqual(len(out["xk"]), 50)


@unittest.skipUnless(core.HAS_CXX_CORE and importlib_available("jwt"), "needs libKalman + PyJWT")
class TestLoadPluginAuth(unittest.TestCase):
    """Loading a plugin runs code, so it has its own scope."""

    def setUp(self):
        os.environ["KALMAN_JWT_SECRET"] = "test-secret-native"

    def tearDown(self):
        os.environ.pop("KALMAN_JWT_SECRET", None)

    def test_scope(self):
        writer = auth.mint_token("alice", ["write_models"])     # can author specs ...
        loader = auth.mint_token("carol", ["load_plugins"])
        with self.assertRaises(auth.AuthError):
            api.load_plugin(PLUGIN)                             # no token
        with self.assertRaises(auth.AuthError):
            api.load_plugin(PLUGIN, credential=writer)          # ... but not load code
        self.assertIn("plugin", api.load_plugin(PLUGIN, credential=loader))
        self.assertIn("plugin", api.load_plugin(PLUGIN, credential=auth.mint_token("root", ["admin"])))


if __name__ == "__main__":
    unittest.main()

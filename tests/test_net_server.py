"""The C++ servers (libKalmanNet, ``kalman-serve``) against the Python facade.

Starts ``kalman-serve`` -- skipped when it is not built -- calls its REST routes
with urllib and has one MCP exchange over stdio, and compares the C++ answers
with :mod:`kalman.api`'s for the same request on the same C++ model (the
example plugin's, loaded in both processes) and on a ``matrices`` spec.
States agree to 1e-9. Tokens minted by the Python side are accepted by the C++
server, and a token revoked through the Python registry is refused by it.
"""
import json
import os
import re
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core, net  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(REPO, "net", "include", "Kalman", "Net", "CAPI.h")
PLUGIN = "KalmanPlugin_example"
SECRET = "a-secret-key-that-is-long-enough-for-hs256"

_PROTO = re.compile(r"^\s*(?:[A-Za-z_][\w ]*\s*\*?\s*)\b(kalman_\w+)\s*\(", re.M)


def _prototypes(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return set(_PROTO.findall(text))


def _serve_binary():
    for d in ("var/install/bin", "var/build/bin"):
        p = os.path.join(REPO, d, "kalman-serve")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _plugin_path():
    if not core.HAS_CXX_CORE:
        return None
    try:
        return core.resolve_plugin(PLUGIN)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


SERVE = _serve_binary()
HAVE = SERVE is not None and _plugin_path() is not None
WHY = "kalman-serve or the example plugin is not built (make build examples, cmake --install)"

CV = {"base": "matrices", "sensor_id": "cv", "fs": 10,
      "dynamics": {"F": [[1, 0.1], [0, 1]], "H": [[1, 0]]},
      "noise": {"Q": [[1e-3, 0], [0, 1e-3]], "R": [[0.25]]},
      "initial": {"x0": [[0], [1]], "P0": [[1, 0], [0, 1]]}}

PENDULUM = {"base": "pendulum", "fs": 50.0,
            "initial": {"x0": [0.4, 0.0], "P0": [[0.1, 0.0], [0.0, 0.1]]}}

CONSTANT_VELOCITY = {"base": "constant-velocity", "fs": 10.0,
                     "initial": {"x0": [0.0, 1.0], "P0": [[1.0, 0.0], [0.0, 1.0]]}}


class Server:
    """``kalman-serve --rest`` on a free port, until stop()."""

    def __init__(self, env=None, *args):
        e = dict(os.environ)
        e.pop("KALMAN_JWT_SECRET", None)
        e.update(env or {})
        self.proc = subprocess.Popen([SERVE, "--rest", "--port", "0", "--plugin", PLUGIN, *args],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=e, text=True)
        line = self.proc.stderr.readline()
        m = re.search(r":(\d+) ", line)
        if not m:
            self.proc.kill()
            raise RuntimeError(f"kalman-serve did not start: {line!r} {self.proc.stderr.read()}")
        self.url = f"http://127.0.0.1:{m.group(1)}"

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def request(self, method, path, body=None, token=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "null")


class TestHeaderContract(unittest.TestCase):
    """``native/net.py``'s cdef is a copy of ``Kalman/Net/CAPI.h``."""

    def test_cdef_matches_header(self):
        with open(HEADER) as fh:
            header = _prototypes(fh.read())
        self.assertTrue(header)
        self.assertEqual(header, _prototypes(net._CDEF))


@unittest.skipUnless(HAVE, WHY)
class TestRestAgainstFacade(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ.pop("KALMAN_JWT_SECRET", None)
        core.load_plugin(_plugin_path())                 # the same C++ model in this process
        cls.server = Server()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def _data(self, spec, n=40, seed=3):
        from modules import api
        rec = api.sample(spec, n / spec["fs"], seed=seed)
        return rec["zk"]

    def assertClose(self, a, b, tol, what):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        self.assertEqual(a.shape, b.shape, what)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.assertLessEqual(err, tol, f"{what}: max |C++ - Python| = {err:.3g}")
        return err

    def test_health_and_catalog(self):
        status, h = self.server.request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual((h["ok"], h["auth"]), (True, False))
        status, cat = self.server.request("GET", "/catalog")
        self.assertEqual(status, 200)
        for name in ("constant-velocity", "pendulum"):
            self.assertIn(name, cat["models"])
        self.assertEqual(set(cat), {"models", "controllers", "engines", "integrators", "solvers",
                                    "providers", "authored", "sources", "native"})

    def test_describe_matches(self):
        from modules import api
        for name in ("pendulum", "constant-velocity"):
            status, d = self.server.request("GET", f"/describe/{name}")
            self.assertEqual(status, 200)
            p = api.describe(name)
            for k in ("n", "m", "l", "fs", "analytic", "backend", "kind", "default_engine", "description"):
                self.assertEqual(d[k], p[k], f"describe({name})[{k}]")
        self.assertEqual(self.server.request("GET", "/describe/Nope")[0], 404)

    def test_filter_matches_on_a_cxx_model(self):
        from modules import api
        for spec in (CONSTANT_VELOCITY, PENDULUM):
            zk = self._data(spec)
            for smooth in (False, True):
                status, c = self.server.request("POST", "/filter", {"spec": spec, "data": zk, "smooth": smooth})
                self.assertEqual(status, 200, c)
                p = api.run_filter(spec, zk, smooth=smooth)
                if smooth and spec is PENDULUM:
                    # The Python RTS pass on a nonlinear model gains with
                    # transition[k] (the Jacobian at x_post[k-1]); Kalman::Filter
                    # with transition[k+1] (at x_post[k]), the RTS gain. They differ
                    # by ~5e-5 here, so the reference is the C++ pass driven from
                    # Python (NativeModel.filter_native), the same Kalman::Filter.
                    rec = api.make_model(spec).filter_native(zk, engine="extended", smooth=True)
                    p = {"xk": rec.xk, "Pk": rec.Pk, "time": rec.time}
                self.assertClose(c["xk"], p["xk"], 1e-9, f"{spec['base']} xk smooth={smooth}")
                self.assertClose(c["Pk"], p["Pk"], 1e-9, f"{spec['base']} Pk smooth={smooth}")
                self.assertClose(c["time"], p["time"], 1e-12, f"{spec['base']} time")

    def test_filter_matches_on_a_matrices_spec(self):
        from modules import api
        zk = [[0.1 * i + 0.05 * np.sin(i)] for i in range(30)]
        status, c = self.server.request("POST", "/filter", {"spec": CV, "data": zk})
        self.assertEqual(status, 200, c)
        p = api.run_filter(CV, zk)
        self.assertClose(c["xk"], p["xk"], 1e-9, "matrices xk")
        self.assertClose(c["Pk"], p["Pk"], 1e-9, "matrices Pk")

    def test_diagnostics_match(self):
        from modules import api
        zk = self._data(PENDULUM)
        _, c = self.server.request("POST", "/filter", {"spec": PENDULUM, "data": zk})
        status, dc = self.server.request("GET", f"/diagnostics/{c['run_id']}")
        self.assertEqual(status, 200, dc)
        dp = api.diagnostics(api.run_filter(PENDULUM, zk)["run_id"])
        self.assertClose(dc["innovation"], dp["innovation"], 1e-9, "innovation")
        self.assertClose(dc["innovation_covariance"], dp["innovation_covariance"], 1e-9, "S")
        self.assertClose(dc["nis"], dp["nis"], 1e-7, "NIS")
        self.assertEqual(self.server.request("GET", "/diagnostics/run-nope")[0], 404)

    def test_forecast_matches(self):
        from modules import api
        for spec in (CONSTANT_VELOCITY, PENDULUM, CV):
            body = {"spec": spec, "x0": [0.3, 0.2], "P0": [[0.5, 0.1], [0.1, 0.4]], "steps": 5}
            status, c = self.server.request("POST", "/forecast", body)
            self.assertEqual(status, 200, c)
            p = api.forecast(spec, body["x0"], body["P0"], 5)
            self.assertClose(c["xk"], p["xk"], 1e-9, f"forecast {spec['base']} xk")
            self.assertClose(c["Pk"], p["Pk"], 1e-9, f"forecast {spec['base']} Pk")

    def test_noise_free_sample_matches(self):
        from modules import api
        for spec in (PENDULUM, CV):
            kw = {"duration": 0.5, "process_noise": False, "measurement_noise": False}
            status, c = self.server.request("POST", "/sample", {"spec": spec, **kw})
            self.assertEqual(status, 200, c)
            p = api.sample(spec, **kw)
            self.assertClose(c["xk"], p["xk"], 1e-9, f"sample {spec['base']} xk")
            self.assertClose(c["zk"], p["zk"], 1e-9, f"sample {spec['base']} zk")
            self.assertClose(c["time"], p["time"], 1e-12, f"sample {spec['base']} time")

    def test_errors_match(self):
        bad = {"spec": {"F": [[1]]}, "data": [[1.0]]}
        status, c = self.server.request("POST", "/filter", bad)
        self.assertEqual(status, 400)
        self.assertIn("'F' belongs in dynamics['F']", c["detail"])
        self.assertEqual(self.server.request("POST", "/filter", {"spec": CV})[0], 400)
        self.assertEqual(self.server.request("POST", "/osse", {"spec": CV})[0], 501)
        self.assertEqual(self.server.request("GET", "/providers")[0], 501)

    def test_shared_registry(self):
        """A plugin loaded through the C++ server is in the Python registry."""
        from modules import api
        status, out = self.server.request("POST", "/plugins", {"name": PLUGIN})
        self.assertEqual(status, 200, out)
        self.assertIn("pendulum", out["models"])
        self.assertIn(PLUGIN, [p["name"] for p in api.plugins()["plugins"]])
        status, evs = self.server.request("GET", "/events?since=0&limit=1000")
        self.assertEqual(status, 200)
        self.assertEqual(evs, api.events(0, 1000))

    def test_in_process_call(self):
        """The same verbs through the C interface (native/net.py)."""
        if not net.available():
            self.skipTest(net.LOAD_ERROR)
        from modules import api
        zk = self._data(CONSTANT_VELOCITY)
        c = net.call("run_filter", {"spec": CONSTANT_VELOCITY, "data": zk})
        p = api.run_filter(CONSTANT_VELOCITY, zk)
        self.assertClose(c["xk"], p["xk"], 1e-9, "in-process xk")
        with self.assertRaises(net.NetError) as ctx:
            net.call("describe", {"name": "Nope"})
        self.assertEqual(ctx.exception.status, 404)


@unittest.skipUnless(HAVE, WHY)
class TestMcpStdio(unittest.TestCase):

    def test_one_exchange(self):
        from modules import api
        core.load_plugin(_plugin_path())
        env = dict(os.environ)
        env.pop("KALMAN_JWT_SECRET", None)
        zk = api.sample(CONSTANT_VELOCITY, 3.0, seed=5)["zk"]
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "pytest", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "run_filter", "arguments": {"spec": CONSTANT_VELOCITY, "data": zk}}},
        ]
        out = subprocess.run([SERVE, "--mcp", "--plugin", PLUGIN], env=env, text=True, timeout=60,
                             input="".join(json.dumps(m) + "\n" for m in msgs), capture_output=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        replies = [json.loads(line) for line in out.stdout.splitlines() if line.strip()]
        self.assertEqual([r["id"] for r in replies], [1, 2, 3])        # no answer to the notification
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "Kalman-AI")
        names = {t["name"] for t in replies[1]["result"]["tools"]}
        python_tools = {"list_models", "describe_model", "list_registered_models", "get_diagnostics",
                        "list_sources", "list_events", "run_filter", "forecast", "load_plugin"}
        self.assertTrue(python_tools <= names, python_tools - names)
        result = replies[2]["result"]
        self.assertFalse(result["isError"], result)
        p = api.run_filter(CONSTANT_VELOCITY, zk)
        err = np.max(np.abs(np.asarray(result["structuredContent"]["xk"]) - np.asarray(p["xk"])))
        self.assertLessEqual(err, 1e-9)


@unittest.skipUnless(HAVE and core.HAS_CXX_CORE, WHY)
class TestTokensFromPython(unittest.TestCase):
    """A token minted by kalman.api.auth works on the C++ server; scopes and
    revocation hold there too."""

    def test_python_tokens(self):
        try:
            import jwt  # noqa: F401
        except ImportError:
            self.skipTest("PyJWT is not installed")
        from modules.api import auth, store
        os.environ["KALMAN_JWT_SECRET"] = SECRET
        try:
            server = Server({"KALMAN_JWT_SECRET": SECRET})
            try:
                reader = auth.mint_token("bob", ["read"])
                self.assertEqual(server.request("GET", "/catalog")[0], 401)
                self.assertEqual(server.request("GET", "/catalog", token=reader)[0], 200)
                status, who = server.request("GET", "/whoami", token=reader)
                self.assertEqual((status, who["user"], who["scopes"]), (200, "bob", ["read"]))
                from modules import api
                self.assertEqual(who, api.whoami(reader))
                self.assertEqual(server.request("POST", "/forecast", {"spec": CV, "x0": [0, 0]},
                                                token=reader)[0], 403)
                self.assertEqual(server.request("GET", "/catalog", token=reader + "x")[0], 401)
                # Revoked through the Python registry: refused by the C++ server at once.
                c = auth.claims(reader)
                store.revoke_token(c["jti"], c["sub"], c["exp"], "test")
                self.assertEqual(server.request("GET", "/catalog", token=reader)[0], 401)
                admin = auth.mint_token("alice", ["admin"])
                self.assertEqual(server.request("POST", "/forecast", {"spec": CV, "x0": [0, 0]},
                                                token=admin)[0], 200)
            finally:
                server.stop()
        finally:
            os.environ.pop("KALMAN_JWT_SECRET", None)


if __name__ == "__main__":
    unittest.main()

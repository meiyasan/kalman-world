"""The shared registry: one set of models, plugins, sources and events for the
REST API, the MCP server and every other process on the same state directory,
with bearer-token auth in front of it."""
import importlib
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, ".")
import modules.api as api
from modules.api import auth, store

CV = {"base": "matrices", "sensor_id": "cv", "fs": 10,
      "dynamics": {"F": [[1, 0.1], [0, 1]], "H": [[1, 0]]},
      "noise": {"Q": [[1e-3, 0], [0, 1e-3]], "R": [[0.25]]},
      "initial": {"x0": [[0], [1]]}}
CV_NOISY = {**CV, "noise": {"Q": [[1e-2, 0], [0, 1e-2]], "R": [[1.0]]}}
SECRET = "a-registry-test-secret-that-is-long-enough"


def _has(mod):
    try:
        importlib.import_module(mod); return True
    except ImportError:
        return False


def _subprocess_env():
    return dict(os.environ, PYTHONPATH=os.getcwd())


class _Auth:
    """Turn auth on for one test."""

    def setUp(self):
        os.environ["KALMAN_JWT_SECRET"] = SECRET

    def tearDown(self):
        for k in ("KALMAN_JWT_SECRET", "KALMAN_JWT_ISSUER", "KALMAN_JWT_AUDIENCE"):
            os.environ.pop(k, None)

    def tok(self, user, *scopes):
        return auth.mint_token(user, list(scopes))


class TestRegistry(unittest.TestCase):

    def test_same_spec_is_a_no_op_and_a_different_one_conflicts(self):
        a = api.register_model(CV, "reg-cv")
        self.assertEqual(a["action"], "created")
        self.assertEqual(api.register_model(CV, "reg-cv")["action"], "unchanged")
        with self.assertRaises(api.ConflictError):
            api.register_model(CV_NOISY, "reg-cv")
        self.assertEqual(api.register_model(CV_NOISY, "reg-cv", replace=True)["action"],
                         "replaced")
        self.assertEqual(api.describe("reg-cv")["spec"]["noise"]["R"], [[1.0]])

    def test_a_name_cannot_shadow_a_catalog_model(self):
        with self.assertRaises(ValueError):
            api.register_model(CV, "OscillatorModel")
        with self.assertRaises(ValueError):
            api.register_model(CV, "../etc/passwd")

    def test_unnamed_specs_are_named_by_content(self):
        a = api.register_model(CV)
        self.assertTrue(a["name"].startswith("authored-"))
        self.assertEqual(api.register_model(CV)["name"], a["name"])

    def test_another_process_sees_the_model(self):
        """The point of the store: two processes, one registry."""
        api.register_model(CV, "seen-elsewhere")
        out = subprocess.run(
            [sys.executable, "-c",
             "import json, modules.api as a; print(json.dumps(a.catalog()['authored']))"],
            env=_subprocess_env(), capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("seen-elsewhere", json.loads(out.stdout.strip().splitlines()[-1]))

    def test_events_record_each_change(self):
        start = store.last_event()
        api.register_model(CV, "evented")
        api.delete_model("evented")
        types = [e["type"] for e in api.events(start)["events"]]
        self.assertEqual(types, ["model.created", "model.deleted"])

    def test_run_ids_do_not_repeat(self):
        zk = [[0.1 * i] for i in range(5)]
        ids = {api.run_filter(CV, zk)["run_id"] for _ in range(3)}
        self.assertEqual(len(ids), 3)
        self.assertTrue(all(i.startswith("run-") for i in ids))


class TestPushSources(unittest.TestCase):

    def test_push_then_fetch_then_filter(self):
        api.register_source("bench", {"kind": "push", "m": 1})
        z = np.asarray(api.make_model(CV).sample(2.0).zk)
        out = api.push_measurements("bench", z.tolist(), time=list(range(len(z))))
        self.assertEqual(out["count"], len(z))
        got = api.fetch("bench")
        np.testing.assert_allclose(np.asarray(got["zk"]), z)
        self.assertEqual(got["time"][:3], [0.0, 1.0, 2.0])
        # read on from the cursor: nothing new yet
        self.assertEqual(api.fetch({"source": "bench"}, {"since": got["meta"]["next"]})["zk"], [])
        res = api.run_filter(CV, source="bench")
        self.assertEqual(np.asarray(res["xk"]).shape, (len(z), 2))

    def test_bad_rows_are_refused(self):
        api.register_source("narrow", {"kind": "push", "m": 1})
        with self.assertRaises(ValueError):
            api.push_measurements("narrow", [[1.0, 2.0]])            # width 2, source says 1
        with self.assertRaises(ValueError):
            api.push_measurements("narrow", [[float("nan")]])
        api.register_source("synth", {"kind": "synthetic", "init": {"spec": CV}})
        with self.assertRaises(ValueError):
            api.push_measurements("synth", [[1.0]])                  # not a push source
        with self.assertRaises(KeyError):
            api.push_measurements("missing", [[1.0]])

    def test_a_query_cannot_move_a_file_source(self):
        api.register_source("fixed-file", {"kind": "file", "init": {"path": "/nonexistent.npz"}})
        with self.assertRaises(ValueError):
            api.fetch("fixed-file", {"path": "/etc/passwd"})


class TestScopesAndOwnership(_Auth, unittest.TestCase):

    def test_only_the_owner_or_an_admin_replaces(self):
        alice = self.tok("alice", "write_models")
        bob = self.tok("bob", "write_models")
        root = self.tok("root", "admin")
        self.assertEqual(api.register_model(CV, "owned", credential=alice)["owner"], "alice")
        with self.assertRaises(auth.AuthError):
            api.register_model(CV_NOISY, "owned", replace=True, credential=bob)
        with self.assertRaises(auth.AuthError):
            api.delete_model("owned", credential=bob)
        api.register_model(CV_NOISY, "owned", replace=True, credential=alice)
        api.delete_model("owned", credential=root)

    def test_raw_file_and_remote_providers_need_admin(self):
        sender = self.tok("ada", "send_measurement")
        with self.assertRaises(auth.AuthError):
            api.fetch({"kind": "remote", "init": {"base_url": "http://example.invalid",
                                                  "credential_name": "cds"}},
                      credential=sender)
        with self.assertRaises(auth.AuthError):
            api.fetch({"kind": "file", "init": {"path": "/etc/passwd"}}, credential=sender)
        writer = self.tok("al", "write_models")
        with self.assertRaises(auth.AuthError):
            api.register_source("lab", {"kind": "remote", "init": {"base_url": "http://x"}},
                                credential=writer)
        api.register_source("lab", {"kind": "remote", "init": {"base_url": "http://x"}},
                            credential=self.tok("root", "admin"))

    def test_issuer_and_audience_are_enforced(self):
        os.environ["KALMAN_JWT_AUDIENCE"] = "kalman-prod"
        prod = self.tok("al", "read")
        self.assertTrue(auth.authorize(prod, "read"))
        os.environ["KALMAN_JWT_AUDIENCE"] = "kalman-staging"
        with self.assertRaises(auth.AuthError):
            auth.authorize(prod, "read")

    def test_whoami(self):
        info = api.whoami(self.tok("carol", "read", "send_measurement"))
        self.assertEqual((info["user"], info["scopes"]), ("carol", ["read", "send_measurement"]))
        with self.assertRaises(auth.AuthError):
            api.whoami("not-a-jwt")


class TestServerCheck(unittest.TestCase):

    def tearDown(self):
        for k in ("KALMAN_JWT_SECRET", "KALMAN_ALLOW_ANONYMOUS"):
            os.environ.pop(k, None)

    def test_open_server_on_a_network_interface_is_refused(self):
        os.environ.pop("KALMAN_JWT_SECRET", None)
        auth.check_server("127.0.0.1", service="t")                  # loopback: fine
        with self.assertRaises(auth.AuthError):
            auth.check_server("0.0.0.0", service="t")
        auth.check_server("0.0.0.0", service="t", verified_elsewhere=True)
        os.environ["KALMAN_ALLOW_ANONYMOUS"] = "1"
        auth.check_server("0.0.0.0", service="t")

    def test_short_secret_is_refused(self):
        os.environ["KALMAN_JWT_SECRET"] = "short"
        with self.assertRaises(auth.AuthError):
            auth.check_server("127.0.0.1", service="t")


@unittest.skipUnless(_has("fastapi"), "fastapi not installed")
class TestRestRegistry(_Auth, unittest.TestCase):

    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from modules.rest.app import create_app
        self.client = TestClient(create_app())

    def h(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_status_codes(self):
        w = self.h(self.tok("al", "write_models"))
        self.assertEqual(self.client.post("/models", json={"spec": CV, "name": "rest-cv"},
                                          headers=w).status_code, 200)
        self.assertEqual(self.client.post("/models", json={"spec": CV_NOISY, "name": "rest-cv"},
                                          headers=w).status_code, 409)
        self.assertEqual(self.client.post("/models", json={"name": "x"}, headers=w).status_code, 400)
        self.assertEqual(self.client.delete("/models/nope", headers=w).status_code, 404)
        other = self.h(self.tok("bob", "write_models"))
        self.assertEqual(self.client.delete("/models/rest-cv", headers=other).status_code, 403)
        r = self.client.get("/whoami")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.headers.get("www-authenticate"), "Bearer")

    def test_measure_from_a_device_and_filter(self):
        """What a Unity client does: push rows, then ask for the estimate."""
        admin = self.h(self.tok("root", "admin"))
        dev = self.h(self.tok("unity", "send_measurement"))
        self.assertEqual(self.client.post("/sources", json={"name": "scene-1",
                         "provider": {"kind": "push", "m": 1}}, headers=admin).status_code, 200)
        z = np.asarray(api.make_model(CV).sample(2.0).zk).tolist()
        self.assertEqual(self.client.post("/sources/scene-1/measurements", json={"rows": z},
                                          headers=dev).status_code, 200)
        r = self.client.post("/filter", json={"spec": CV, "source": "scene-1"}, headers=dev)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(np.asarray(r.json()["xk"]).shape, (len(z), 2))
        back = self.client.get("/sources/scene-1/measurements?since=0", headers=dev).json()
        self.assertEqual(len(back["zk"]), len(z))
        listing = self.client.get("/sources", headers=self.h(self.tok("x", "read"))).json()
        scene = [s for s in listing["sources"] if s["name"] == "scene-1"][0]
        self.assertEqual(scene["rows"], len(z))

    def test_event_stream(self):
        start = store.last_event()
        self.client.post("/models", json={"spec": CV, "name": "streamed"},
                         headers=self.h(self.tok("al", "write_models")))
        r = self.client.get(f"/events/stream?since={start}&timeout=0",
                            headers=self.h(self.tok("x", "read")))
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: model.created", r.text)
        self.assertIn('"name": "streamed"', r.text)
        self.assertEqual(self.client.get("/events/stream?timeout=0").status_code, 401)


@unittest.skipUnless(_has("fastapi"), "fastapi not installed")
class TestCors(unittest.TestCase):

    def tearDown(self):
        os.environ.pop("KALMAN_CORS_ORIGINS", None)

    def _preflight(self):
        from fastapi.testclient import TestClient
        from modules.rest.app import create_app
        return TestClient(create_app()).options("/catalog", headers={
            "Origin": "https://game.example", "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization"})

    def test_off_by_default_on_when_configured(self):
        os.environ.pop("KALMAN_CORS_ORIGINS", None)
        self.assertNotIn("access-control-allow-origin", self._preflight().headers)
        os.environ["KALMAN_CORS_ORIGINS"] = "https://game.example"
        r = self._preflight()
        self.assertEqual(r.headers.get("access-control-allow-origin"), "https://game.example")


@unittest.skipUnless(_has("fastapi") and _has("mcp"), "fastapi + mcp needed")
class TestMcpRemoteBackend(_Auth, unittest.TestCase):
    """The MCP server as a client of the REST API: one registry, one auth."""

    def setUp(self):
        super().setUp()
        from fastapi.testclient import TestClient
        from modules.rest.app import create_app
        from modules.mcp.backend import RemoteBackend, LocalBackend
        self.remote = RemoteBackend("http://testserver", client=TestClient(create_app()))
        self.local = LocalBackend()

    def test_registered_through_mcp_listed_through_rest(self):
        w = self.tok("agent", "write_models")
        self.remote.register_model(CV, "via-mcp", False, w)
        self.assertIn("via-mcp", self.local.catalog(self.tok("x", "read"))["authored"])
        from modules.mcp.backend import RemoteError
        with self.assertRaises(RemoteError) as e:
            self.remote.register_model(CV_NOISY, "via-mcp", False, w)
        self.assertEqual(e.exception.status, 409)
        with self.assertRaises(auth.AuthError):
            self.remote.register_model(CV, "nope", False, self.tok("x", "read"))

    def test_a_run_made_through_mcp_is_diagnosed_through_rest(self):
        s = self.tok("agent", "send_measurement", "read")
        out = self.remote.run_filter(CV, [[0.1 * i] for i in range(10)], None, None, False, s)
        self.assertIn("nis", self.remote.diagnostics(out["run_id"], s))

    def test_verifier(self):
        import asyncio
        from modules.mcp.server import JwtVerifier
        v = JwtVerifier(self.remote)
        at = asyncio.run(v.verify_token(self.tok("eve", "read")))
        self.assertEqual((at.subject, at.scopes), ("eve", ["read"]))
        self.assertIsNone(asyncio.run(v.verify_token("forged.token.value")))


def _native():
    try:
        from modules.framework.native import core
        core.resolve_plugin("KalmanPlugin_example")
        return core.HAS_CXX_CORE
    except Exception:
        return False


@unittest.skipUnless(_native(), "libKalmanFramework or the example plugin not built")
class TestPluginsAreShared(unittest.TestCase):

    def test_loaded_once_known_everywhere(self):
        out = api.load_plugin("KalmanPlugin_example")
        self.assertIn("pendulum", out["native"])
        rec = {p["name"]: p for p in api.plugins()["plugins"]}
        self.assertIn("pendulum", rec["KalmanPlugin_example"]["models"])
        # A fresh process never called load_plugin, yet has the plugin's models.
        r = subprocess.run(
            [sys.executable, "-c",
             "import json, modules.api as a; print(json.dumps(a.catalog()['models']))"],
            env=_subprocess_env(), capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pendulum", json.loads(r.stdout.strip().splitlines()[-1]))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(_has("mcp") and _has("jwt"), "mcp + pyjwt needed")
class TestMcpHttpBearer(unittest.TestCase):
    """End to end: the MCP server over streamable HTTP takes the token from the
    Authorization header, and refuses a client without one."""

    def test_header_token(self):
        import asyncio
        port = _free_port()
        env = dict(_subprocess_env(), KALMAN_JWT_SECRET=SECRET,
                   KALMAN_MCP_TRANSPORT="streamable-http", KALMAN_MCP_PORT=str(port))
        env.pop("KALMAN_API_URL", None)
        proc = subprocess.Popen([sys.executable, "-m", "modules.mcp"], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.time() + 60
            while time.time() < deadline:
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                    break
                except OSError:
                    if proc.poll() is not None:
                        self.fail(proc.stderr.read().decode())
                    time.sleep(0.2)
            url = f"http://127.0.0.1:{port}/mcp"
            token = auth.mint_token("agent", ["read", "send_measurement"], secret=SECRET)

            def transport(headers):
                try:                                   # SDK >= 2.0
                    from mcp.client.streamable_http import streamable_http_client
                    import httpx2
                    return streamable_http_client(url, http_client=httpx2.AsyncClient(
                        headers=headers, timeout=60))
                except ImportError:                    # SDK 1.x
                    from mcp.client.streamable_http import streamablehttp_client
                    return streamablehttp_client(url, headers=headers)

            async def call(headers):
                from mcp import ClientSession
                async with transport(headers) as (r, w, *_):
                    async with ClientSession(r, w) as s:
                        await s.initialize()
                        return await s.call_tool("run_filter", {
                            "spec": CV, "data": [[0.1 * i] for i in range(10)]})

            ok = asyncio.run(call({"Authorization": f"Bearer {token}"}))
            self.assertFalse(ok.is_error, ok.content)
            payload = json.loads("".join(c.text for c in ok.content if c.type == "text"))
            self.assertEqual(len(payload["xk"]), 10)

            import httpx
            init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "t", "version": "0"}}}
            hdrs = {"Accept": "application/json, text/event-stream"}
            for bad in ({}, {"Authorization": "Bearer forged.token.value"}):
                r = httpx.post(url, json=init, headers={**hdrs, **bad}, timeout=10)
                self.assertEqual(r.status_code, 401, bad)       # refused before any tool
                self.assertIn("Bearer", r.headers.get("www-authenticate", ""))
        finally:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()

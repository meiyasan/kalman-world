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
        # 401 = no usable token (none, forged, expired); 403 = a good token, not allowed
        self.assertEqual(self.client.get("/catalog", headers=self.h("forged.token.x")).status_code, 401)
        expired = auth.mint_token("al", ["read"], exp_hours=-1.0)
        self.assertEqual(self.client.get("/catalog", headers=self.h(expired)).status_code, 401)
        self.assertEqual(self.client.get("/catalog", headers=self.h(self.tok("al", "write_models"))).status_code, 403)

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

    def test_a_revoked_token_is_refused_at_once(self):
        """No cached "yes": the MCP server asks the API on every request."""
        import asyncio
        from modules.mcp.server import JwtVerifier
        v, tok = JwtVerifier(self.remote), self.tok("mallory", "read")
        self.assertIsNotNone(asyncio.run(v.verify_token(tok)))
        auth.revoke(tok)
        self.assertIsNone(asyncio.run(v.verify_token(tok)))

    def test_watcher_needs_no_token(self):
        """With a read token the watcher names the resource that changed.
        Without one it still notices the change (GET /events/cursor is open)
        and reports every resource."""
        import asyncio
        from modules.mcp.watch import RegistryWatcher, MODELS, SOURCES, EVENTS
        writer = self.tok("al", "write_models")
        for token, expected in ((self.tok("w", "read"), {MODELS, EVENTS}),
                                (None, {MODELS, SOURCES, EVENTS})):
            w = RegistryWatcher(self.remote, bus=None, token=token)
            cursor, stale = asyncio.run(w._poll(None))
            self.assertEqual(stale, set())                       # start from now
            self.assertEqual(asyncio.run(w._poll(cursor)), (cursor, set()))   # nothing new
            self.remote.register_model(CV, f"watched-{bool(token)}", False, writer)
            cursor2, stale = asyncio.run(w._poll(cursor))
            self.assertGreater(cursor2, cursor)
            self.assertEqual(stale, expected)
            self.assertEqual(w._precise, token is not None)


@unittest.skipUnless(_has("fastapi") and _has("bcrypt") and _has("jwt"), "fastapi + bcrypt + pyjwt needed")
class TestLoginAndRevocation(_Auth, unittest.TestCase):
    """POST /token: an account's password for a scoped JWT; POST /token/revoke."""

    def setUp(self):
        super().setUp()
        import tempfile
        from fastapi.testclient import TestClient
        from modules.rest.app import create_app
        self.dir = tempfile.mkdtemp()
        os.environ["KALMAN_HTPASSWD"] = os.path.join(self.dir, "htpasswd")
        store.clear_login_failures()
        auth.add_user("alice", "correct horse", scopes="read,send_measurement,write_models")
        auth.add_user("bob", "battery staple")                       # no grant: read only
        auth.add_user("root", "tr0ub4dor&3", scopes="admin")
        self.client = TestClient(create_app())

    def tearDown(self):
        super().tearDown()
        for k in ("KALMAN_HTPASSWD", "KALMAN_TOKEN_MAX_HOURS"):
            os.environ.pop(k, None)

    def login(self, user, password, **kw):
        return self.client.post("/token", json={"user": user, "password": password, **kw})

    def h(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_login_gives_a_working_token(self):
        r = self.login("alice", "correct horse")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["token_type"], "bearer")
        self.assertEqual(body["scopes"], ["read", "send_measurement", "write_models"])
        who = self.client.get("/whoami", headers=self.h(body["access_token"])).json()
        self.assertEqual(who["user"], "alice")
        self.assertEqual(self.client.post("/models", json={"spec": CV, "name": "by-login"},
                                          headers=self.h(body["access_token"])).json()["owner"],
                         "alice")

    def test_scopes_are_limited_to_the_grant(self):
        narrow = self.login("alice", "correct horse", scopes=["read"]).json()
        self.assertEqual(narrow["scopes"], ["read"])
        self.assertEqual(self.login("alice", "correct horse", scopes=["admin"]).status_code, 403)
        self.assertEqual(self.login("bob", "battery staple").json()["scopes"], ["read"])
        self.assertEqual(self.login("bob", "battery staple",
                                    scopes=["write_models"]).status_code, 403)
        # an admin grant covers every scope
        self.assertEqual(self.login("root", "tr0ub4dor&3",
                                    scopes=["load_plugins"]).json()["scopes"], ["load_plugins"])

    def test_lifetime_is_capped(self):
        os.environ["KALMAN_TOKEN_MAX_HOURS"] = "2"
        self.assertEqual(self.login("alice", "correct horse", exp_hours=1000).json()["expires_in"],
                         7200)
        self.assertEqual(self.login("alice", "correct horse", exp_hours=0.5).json()["expires_in"],
                         1800)

    def test_wrong_password_then_lockout(self):
        for _ in range(auth.LOGIN_ATTEMPTS):
            self.assertEqual(self.login("alice", "nope").status_code, 401)
        r = self.login("alice", "correct horse")                     # right, but locked out
        self.assertEqual(r.status_code, 429)
        self.assertGreater(int(r.headers["retry-after"]), 0)
        # the same answer for an account that does not exist, and no lockout of others
        self.assertEqual(self.login("nobody", "x").status_code, 401)
        self.assertEqual(self.login("bob", "battery staple").status_code, 200)

    def test_a_lockout_holds_in_another_process_and_an_admin_ends_it(self):
        """The count is in the registry: a second worker, or the server after a
        restart, does not hand out five more tries."""
        for _ in range(auth.LOGIN_ATTEMPTS):
            self.login("alice", "nope")
        out = subprocess.run(
            [sys.executable, "-c",
             "import modules.api.auth as a\n"
             "try:\n    a.login('alice', 'correct horse', client='testclient'); print('IN')\n"
             "except a.RateLimited as e:\n    print('LOCKED', int(e.retry_after) > 0)"],
            env=_subprocess_env(), capture_output=True, text=True, timeout=120)
        self.assertIn("LOCKED True", out.stdout, out.stderr)
        bob = self.login("bob", "battery staple").json()["access_token"]
        root = self.login("root", "tr0ub4dor&3").json()["access_token"]
        self.assertEqual(self.client.post("/token/unlock", json={"user": "alice"},
                                          headers=self.h(bob)).status_code, 403)
        self.assertEqual(self.client.post("/token/unlock", json={"user": "alice"},
                                          headers=self.h(root)).status_code, 200)
        self.assertEqual(self.login("alice", "correct horse").status_code, 200)

    def test_cli_revokes_through_the_api(self):
        """`make api-revoke API=...`: the registry is the server's, not this
        machine's, so the CLI asks the API with an admin token it mints."""
        tok = self.login("alice", "correct horse").json()["access_token"]
        calls = []

        def post(base_url, path, token, body):
            calls.append((base_url, path, body))
            r = self.client.post(path, json=body, headers=self.h(token))
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
            return r.json()

        real, auth._api_post = auth._api_post, post
        try:
            self.assertEqual(auth._main(["revoke", "--user", "alice", "--api", "http://api"]), 0)
            self.assertEqual(auth._main(["unlock", "--user", "alice", "--api", "http://api"]), 0)
        finally:
            auth._api_post = real
        self.assertEqual([c[1] for c in calls], ["/token/revoke", "/token/unlock"])
        self.assertEqual(self.client.get("/catalog", headers=self.h(tok)).status_code, 401)

    def test_changing_the_password_keeps_the_grant(self):
        auth.add_user("alice", "new password")
        self.assertEqual(auth.user_scopes("alice"), ["read", "send_measurement", "write_models"])
        self.assertTrue(auth.verify_user("alice", "new password"))
        self.assertFalse(auth.verify_user("alice", "correct horse"))

    def test_logout_revokes_this_token_only(self):
        a = self.login("alice", "correct horse").json()["access_token"]
        b = self.login("alice", "correct horse").json()["access_token"]
        self.assertEqual(self.client.post("/token/revoke", headers=self.h(a)).status_code, 200)
        self.assertEqual(self.client.get("/catalog", headers=self.h(a)).status_code, 401)
        self.assertEqual(self.client.get("/catalog", headers=self.h(b)).status_code, 200)
        self.assertEqual(self.client.post("/token/revoke").status_code, 401)

    def test_admin_revokes_all_tokens_of_a_user(self):
        a = self.login("alice", "correct horse").json()["access_token"]
        b = self.login("bob", "battery staple").json()["access_token"]
        root = self.login("root", "tr0ub4dor&3").json()["access_token"]
        self.assertEqual(self.client.post("/token/revoke", json={"user": "alice"},
                                          headers=self.h(b)).status_code, 403)
        self.assertEqual(self.client.post("/token/revoke", json={"user": "alice"},
                                          headers=self.h(root)).status_code, 200)
        self.assertEqual(self.client.get("/catalog", headers=self.h(a)).status_code, 401)
        self.assertEqual(self.client.get("/catalog", headers=self.h(b)).status_code, 200)
        time.sleep(0.01)
        again = self.login("alice", "correct horse").json()["access_token"]
        self.assertEqual(self.client.get("/catalog", headers=self.h(again)).status_code, 200)

    def test_another_process_refuses_a_revoked_token(self):
        tok = self.login("alice", "correct horse").json()["access_token"]
        self.client.post("/token/revoke", headers=self.h(tok))
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys, modules.api.auth as a\n"
             "try:\n    a.authorize(sys.argv[1], 'read'); print('ALLOWED')\n"
             "except a.AuthError as e:\n    print('DENIED', e)", tok],
            env=_subprocess_env(), capture_output=True, text=True, timeout=120)
        self.assertIn("DENIED this token was revoked", out.stdout, out.stderr)


@unittest.skipUnless(_has("mcp"), "mcp SDK not installed")
class TestMcpNotifications(unittest.TestCase):
    """A client listening on catalog://models hears about a model that another
    process registers."""

    def setUp(self):
        os.environ["KALMAN_EVENTS_POLL"] = "0.1"
        os.environ.pop("KALMAN_JWT_SECRET", None)

    def tearDown(self):
        os.environ.pop("KALMAN_EVENTS_POLL", None)

    def test_the_sdk_still_has_the_subscribe_hook(self):
        """attach() reaches the low-level server through an attribute that is not
        public SDK API. If a release moves it, this fails — not a client."""
        from modules.mcp.server import _server_class
        from modules.mcp.watch import RegistryWatcher
        try:
            from mcp.server.subscriptions import InMemorySubscriptionBus  # noqa: F401
        except ImportError:
            self.skipTest("MCP SDK without subscriptions")
        self.assertTrue(RegistryWatcher(None, None).attach(_server_class()("t")))

    def test_uris(self):
        from modules.mcp.watch import uris_for
        self.assertEqual(uris_for("model.created"), {"catalog://models", "registry://events"})
        self.assertEqual(uris_for("plugin.loaded"), {"catalog://models", "registry://events"})
        self.assertEqual(uris_for("measurement.received"),
                         {"catalog://sources", "registry://events"})

    def test_listen_hears_a_change_made_elsewhere(self):
        import asyncio
        try:
            from mcp.client import Client
            from mcp.server.subscriptions import ResourceUpdated
        except ImportError:
            self.skipTest("MCP SDK without subscriptions/listen")
        from modules.mcp.server import build_server
        from modules.mcp.backend import LocalBackend
        name = f"heard-{os.getpid()}-{int(time.time())}"
        code = ("import json, sys, modules.api as a; "
                "a.register_model(json.loads(sys.argv[1]), sys.argv[2])")

        async def run():
            async with Client(build_server(LocalBackend())) as client:
                async with client.listen(resource_subscriptions=["catalog://models"]) as sub:
                    await asyncio.sleep(0.5)                 # the watcher has its cursor
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-c", code, json.dumps(CV), name, env=_subprocess_env())
                    self.assertEqual(await proc.wait(), 0)
                    event = await asyncio.wait_for(sub.__anext__(), timeout=20)
                # the notification says "read again"; the content comes from the resource
                models = await client.read_resource("catalog://models")
                return event, models

        event, models = asyncio.run(run())
        self.assertEqual(event, ResourceUpdated(uri="catalog://models"))
        text = "".join(getattr(c, "text", "") for c in models.contents)
        self.assertIn(name, json.loads(text)["authored"])

    def test_resources_subscribe_for_clients_before_listen(self):
        """A real server process over stdio, and a client that speaks the
        protocol most clients speak today: resources/subscribe, then
        notifications/resources/updated when this process changes the registry."""
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        name = f"legacy-{os.getpid()}-{int(time.time())}"
        got = []

        async def on_message(message):
            params = getattr(message, "params", None)
            if getattr(params, "uri", None):
                got.append(str(params.uri))

        async def run():
            params = StdioServerParameters(command=sys.executable, args=["-m", "modules.mcp"],
                                           env=_subprocess_env(), cwd=os.getcwd())
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w, message_handler=on_message) as s:
                    init = await s.initialize()
                    self.assertTrue(init.capabilities.resources.subscribe)
                    await s.subscribe_resource("catalog://sources")
                    await asyncio.sleep(0.5)                  # the watcher has its cursor
                    await asyncio.to_thread(api.register_model, CV, name)     # not subscribed
                    await asyncio.to_thread(api.register_source, name, {"kind": "push"})
                    for _ in range(100):
                        if got:
                            break
                        await asyncio.sleep(0.1)
                    await s.unsubscribe_resource("catalog://sources")

        asyncio.run(run())
        self.assertEqual(got, ["catalog://sources"])       # only what was subscribed


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

"""Tests for the MCP + REST transports (guarded — skip when their optional
deps are absent; the facade they wrap is covered by test_api.py)."""
import importlib
import os
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")

CV = {"base": "matrices", "sensor_id": "cv", "fs": 10,
      "dynamics": {"F": [[1, 0.1], [0, 1]], "H": [[1, 0]]},
      "noise": {"Q": [[1e-3, 0], [0, 1e-3]], "R": [[0.25]]},
      "initial": {"x0": [[0], [1]]}}


def _has(mod):
    try:
        importlib.import_module(mod); return True
    except ImportError:
        return False


@unittest.skipUnless(_has("fastapi"), "fastapi not installed")
class TestRest(unittest.TestCase):

    def setUp(self):
        from fastapi.testclient import TestClient
        from modules.rest.app import create_app
        os.environ.pop("KALMAN_JWT_SECRET", None)   # auth disabled -> open
        self.client = TestClient(create_app())

    def test_health_and_catalog(self):
        self.assertEqual(self.client.get("/healthz").json(), {"ok": True})
        cat = self.client.get("/catalog").json()
        self.assertIn("OscillatorModel", cat["models"])

    def test_describe_404(self):
        self.assertEqual(self.client.get("/describe/Nope").status_code, 404)

    def test_filter_roundtrip(self):
        zk = [[float(i) * 0.1] for i in range(20)]
        r = self.client.post("/filter", json={"spec": CV, "data": zk})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(np.array(r.json()["xk"]).shape, (20, 2))

    def test_openapi_served(self):
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)

    def test_auth_scopes(self):
        import jwt  # noqa
        from modules.api import auth
        os.environ["KALMAN_JWT_SECRET"] = "a-secret-key-that-is-long-enough-32"
        try:
            from fastapi.testclient import TestClient
            from modules.rest.app import create_app
            c = TestClient(create_app())
            # no token -> 401 on a gated route
            self.assertEqual(c.get("/catalog").status_code, 401)
            reader = auth.mint_token("bob", ["read"])
            self.assertEqual(c.get("/catalog", headers={"Authorization": f"Bearer {reader}"}).status_code, 200)
            # read token cannot register a model -> 403
            r = c.post("/models", json={"spec": CV, "name": "x"},
                       headers={"Authorization": f"Bearer {reader}"})
            self.assertEqual(r.status_code, 403)
            writer = auth.mint_token("al", ["write_models"])
            r2 = c.post("/models", json={"spec": CV, "name": "ok"},
                        headers={"Authorization": f"Bearer {writer}"})
            self.assertEqual(r2.status_code, 200)
        finally:
            os.environ.pop("KALMAN_JWT_SECRET", None)


class TestMcp(unittest.TestCase):

    @unittest.skipIf(_has("mcp"), "mcp installed — main() would start the server")
    def test_main_prints_hint_without_sdk(self):
        """Without the SDK, the entry point prints an install hint and exits 0
        rather than crashing (the core never hard-depends on mcp)."""
        from modules.mcp.__main__ import main
        self.assertEqual(main([]), 0)

    @unittest.skipUnless(_has("mcp"), "mcp SDK not installed")
    def test_build_server_registers_tools(self):
        """The server must actually construct against the installed SDK — this
        caught a real breakage when SDK 2.0 renamed FastMCP -> MCPServer."""
        from modules.mcp.server import build_server, _server_class
        self.assertIsNotNone(_server_class())
        server = build_server()
        self.assertIsNotNone(server)

    @unittest.skipUnless(_has("mcp"), "mcp SDK not installed")
    def test_live_stdio_session(self):
        """End-to-end: drive the server as a real MCP client over stdio —
        handshake, tool listing, a scoped call, and scope enforcement."""
        import asyncio, json, os, subprocess

        async def drive():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from modules.api.auth import mint_token
            secret = "a-long-enough-integration-test-secret-key"
            env = dict(os.environ, PYTHONPATH=os.getcwd(), KALMAN_JWT_SECRET=secret)
            params = StdioServerParameters(command=sys.executable,
                                           args=["-m", "kalman.mcp"], env=env,
                                           cwd=os.getcwd())
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    tools = {t.name for t in (await s.list_tools()).tools}
                    reader = mint_token("bob", ["read"], secret=secret)
                    sender = mint_token("ada", ["send_measurement"], secret=secret)
                    ok = await s.call_tool("run_filter",
                                           {"spec": CV, "data": [[0.1 * i] for i in range(10)],
                                            "token": sender})
                    denied = await s.call_tool("run_filter",
                                               {"spec": CV, "data": [[0.0]], "token": reader})
                    return tools, ok, denied

        tools, ok, denied = asyncio.run(drive())
        self.assertIn("run_filter", tools)
        self.assertIn("register_model", tools)
        self.assertFalse(ok.is_error)
        payload = json.loads("".join(c.text for c in ok.content if getattr(c, "type", "") == "text"))
        self.assertEqual(len(payload["xk"]), 10)
        self.assertTrue(denied.is_error)          # read-only token refused


class TestRestMainHint(unittest.TestCase):

    def test_main_hint_without_uvicorn(self):
        if _has("uvicorn"):
            self.skipTest("uvicorn installed")
        from modules.rest.__main__ import main
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()

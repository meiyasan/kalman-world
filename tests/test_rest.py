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

    def test_import_and_main_hint(self):
        # __main__.main() must not raise even when the mcp SDK is absent.
        from modules.mcp.__main__ import main
        self.assertEqual(main([]), 0)

    @unittest.skipUnless(_has("mcp"), "mcp SDK not installed")
    def test_build_server_registers_tools(self):
        from modules.mcp.server import build_server
        server = build_server()
        self.assertIsNotNone(server)


class TestRestMainHint(unittest.TestCase):

    def test_main_hint_without_uvicorn(self):
        if _has("uvicorn"):
            self.skipTest("uvicorn installed")
        from modules.rest.__main__ import main
        self.assertEqual(main([]), 0)


if __name__ == "__main__":
    unittest.main()

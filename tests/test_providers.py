"""Tests for the kalman.providers data layer + the facade fetch verb."""
import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, ".")
import modules.api as api
from modules.providers import (KalmanProvider, SyntheticProvider, FileProvider,
                               SensorProvider, RemoteProvider, ERA5Provider,
                               TurbulenceProvider, PROVIDER_BY_KIND, resolve_provider)
from modules.providers.base import KalmanProvider as Base
from modules.framework.constants import KalmanProviderKind, ALL_PROVIDERS

CV = {"base": "matrices", "sensor_id": "cv", "fs": 10,
      "dynamics": {"F": [[1, 0.1], [0, 1]], "H": [[1, 0]]},
      "noise": {"Q": [[1e-3, 0], [0, 1e-3]], "R": [[0.25]]},
      "initial": {"x0": [[0], [1]]}}


class TestRegistry(unittest.TestCase):

    def test_coverage_and_resolve(self):
        self.assertEqual(set(PROVIDER_BY_KIND), ALL_PROVIDERS)
        self.assertIsInstance(resolve_provider("file"), FileProvider)
        self.assertIsInstance(resolve_provider(KalmanProviderKind.SENSOR), SensorProvider)
        with self.assertRaises(ValueError):
            resolve_provider("nope")


class TestSynthetic(unittest.TestCase):

    def test_feeds_run_filter(self):
        p = SyntheticProvider(spec=CV)
        payload = p.fetch(duration=2.0)
        self.assertEqual(np.asarray(payload["zk"]).shape, (20, 1))
        res = api.run_filter(CV, payload["zk"])
        self.assertEqual(np.asarray(res["xk"]).shape, (20, 2))
        # measurements()/truth() conveniences
        self.assertEqual(p.measurements(duration=1.0).shape[0], 10)
        self.assertEqual(p.truth(duration=1.0).shape[1], 2)


class TestFile(unittest.TestCase):

    def test_save_load_roundtrip(self):
        zk = np.random.default_rng(0).standard_normal((12, 1))
        with tempfile.TemporaryDirectory() as d:
            for ext in (".npz", ".json"):
                path = os.path.join(d, "m" + ext)
                FileProvider.save(path, zk)
                out = FileProvider(path).fetch()
                np.testing.assert_allclose(out["zk"], zk)

    def test_needs_path(self):
        with self.assertRaises(ValueError):
            FileProvider().fetch()


class TestSensor(unittest.TestCase):

    def test_stream_bounded_and_consumed_by_filter(self):
        from modules.filters import LinearKalmanFilter
        rows = [[float(i)] for i in range(5)]
        p = SensorProvider(rows)
        streamed = list(p.stream())
        self.assertEqual(len(streamed), 5)
        # callback source with a stop sentinel
        seq = iter([[1.0], [2.0], None, [3.0]])
        cb = SensorProvider(lambda: next(seq))
        self.assertEqual(len(list(cb.stream())), 2)      # stops at None
        # fetch drains to an array a filter can consume
        zk = p.fetch()["zk"]
        self.assertEqual(zk.shape, (5, 1))


class _MockResp:
    def __init__(self, payload): self._p = payload; self.calls = 0
    def raise_for_status(self): pass
    def json(self): return self._p


class _MockClient:
    def __init__(self, payload): self.payload = payload; self.calls = 0
    def get(self, url, params=None, headers=None):
        self.calls += 1
        return _MockResp(self.payload)


class TestRemote(unittest.TestCase):

    def test_cache_and_headers(self):
        client = _MockClient({"zk": [[1.0], [2.0]]})
        p = RemoteProvider("http://x", client=client)
        a = p.fetch("data"); b = p.fetch("data")
        np.testing.assert_allclose(a["zk"], [[1.0], [2.0]])
        self.assertEqual(client.calls, 1)                # 2nd fetch served from cache

    def test_fail_closed_without_credential(self):
        client = _MockClient({"zk": [[1.0]]})
        p = RemoteProvider("http://x", credential_name="acme", client=client)
        os.environ.pop("KALMAN_ACME_KEY", None)
        with self.assertRaises(PermissionError):
            p.fetch("data")
        os.environ["KALMAN_ACME_KEY"] = "secret"
        try:
            out = p.fetch("data")                        # now authorized
            self.assertIn("zk", out)
        finally:
            os.environ.pop("KALMAN_ACME_KEY", None)


class TestReanalysisProviders(unittest.TestCase):

    def _flow(self):
        from modules.apps.weather.flow import KolmogorovFlowModel
        m = KolmogorovFlowModel("f", grid=16, cycle=8, dt=0.01, nu=1/40., n_obs=20, seed=3)
        m.climatology(n_snapshots=20, spinup=200, stride=5)
        return m

    def test_turbulence_transform(self):
        m = self._flow()
        fields = np.stack([m.solver.random_ic() for _ in range(4)])   # (4, 16, 16)
        out = TurbulenceProvider(m).fetch(source=fields, variable="vorticity")
        self.assertEqual(np.asarray(out["truth"]).shape, (4, m.n))

    def test_era5_guarded(self):
        try:
            import cdsapi, xarray  # noqa
        except ImportError:
            with self.assertRaises((ImportError, ValueError)):
                ERA5Provider(self._flow()).fetch(dates=("2024-01-01", "2024-01-02"),
                                                 area=(50, 0, 10, 360))
            return
        self.skipTest("cdsapi present — needs live CDS credentials")


class TestFacadeFetch(unittest.TestCase):

    def test_fetch_verb_and_scope(self):
        out = api.fetch({"kind": "synthetic", "init": {"spec": CV}}, {"duration": 1.0})
        self.assertEqual(np.asarray(out["zk"]).shape, (10, 1))
        # scope enforcement when auth enabled
        try:
            import jwt  # noqa
            from modules.api import auth
        except ImportError:
            return
        os.environ["KALMAN_JWT_SECRET"] = "secret-key-for-tests-32bytes!!"
        try:
            reader = auth.mint_token("bob", ["read"])
            with self.assertRaises(auth.AuthError):
                api.fetch({"kind": "synthetic", "init": {"spec": CV}}, {"duration": 1.0},
                          credential=reader)
            sender = auth.mint_token("al", ["send_measurement"])
            ok = api.fetch({"kind": "synthetic", "init": {"spec": CV}}, {"duration": 1.0},
                           credential=sender)
            self.assertIn("zk", ok)
        finally:
            os.environ.pop("KALMAN_JWT_SECRET", None)


if __name__ == "__main__":
    unittest.main()

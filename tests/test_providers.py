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
        from modules.framework.filters import LinearKalmanFilter
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


class TestProviderDiscovery(unittest.TestCase):
    """A provider must advertise what it can supply, not only that it exists.

    ``catalog()['providers']`` answers "which sources are there"; without the
    measurement declaration a caller still has to read the source to learn that
    ERA5 yields 500 hPa wind, which is exactly the question ``fetch`` needs
    answered before it can be written.
    """

    def test_every_provider_declares_its_measurements(self):
        for name, record in api.providers()["providers"].items():
            self.assertTrue(record["measurements"],
                            f"provider {name!r} advertises no measurements")
            for measurement in record["measurements"]:
                self.assertIn("name", measurement)
                self.assertIn("description", measurement)

    def test_era5_declares_the_ecmwf_variables_and_its_prerequisites(self):
        era5 = api.describe_provider("era5")
        names = [m["name"] for m in era5["measurements"]]
        self.assertIn("u_component_of_wind", names)
        self.assertIn("v_component_of_wind", names)
        self.assertIn("vorticity", names)
        # The credentials/packages a fetch needs are stated up front rather than
        # surfacing as an ImportError mid-download.
        requires = " ".join(era5["requires"]).lower()
        self.assertIn("cdsapi", requires)
        self.assertIn("cdsapirc", requires)

    def test_query_keys_are_read_off_the_real_fetch_signature(self):
        era5 = api.describe_provider("era5")
        keys = {q["name"] for q in era5["query"]}
        for expected in ("path", "dates", "area", "level", "stride"):
            self.assertIn(expected, keys)

    def test_streaming_providers_are_marked(self):
        self.assertTrue(api.describe_provider("sensor")["streams"])
        self.assertFalse(api.describe_provider("era5")["streams"])

    def test_unknown_provider_lists_the_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            api.describe_provider("ecmwf")
        self.assertIn("era5", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class TestCatalogSpecInitialState(unittest.TestCase):
    """initial.x0/P0 must reach catalog models, not just the matrices path.

    Regression for a SILENT failure: make_model filtered kw against the class's NAMED
    signature, so a model taking **kwargs (as the app models do) never received x0/P0. The
    filter then started from the origin with a default covariance and returned confident
    nonsense instead of raising -- a satellite spec carrying its true initial state diverged
    by 892 km before anyone noticed the state had been dropped.
    """

    def _spec(self, order, n, x0_first=6771e3):
        MU, G = 3.986004418e14, 6.674e-11
        x0 = [0.0] * n
        x0[0] = x0_first
        return {
            "base": "BallisticModel", "fs": 1.0,
            "dynamics": {"params": {"G": G, "M": MU / G, "m": 1.0, "c_drag": 0, "g": None,
                                    "order": order, "fs": 1.0,
                                    "process_uncertainty": 0.05,
                                    "measurement_uncertainty": 500.0}},
            "initial": {"x0": x0,
                        "P0": [[(1234.0 ** 2 if i == j else 0.0) for j in range(n)]
                               for i in range(n)]},
        }

    def test_x0_and_p0_reach_a_catalog_model(self):
        m = api.make_model(self._spec(3, 9))
        self.assertAlmostEqual(float(np.ravel(m.x0)[0]), 6771e3, places=3)
        self.assertAlmostEqual(float(np.asarray(m.P0)[0, 0]), 1234.0 ** 2, places=3)

    def test_ballistic_order_2_is_not_an_index_error(self):
        """order=2 is documented and used to raise IndexError from a hardcoded [0,3,6]."""
        for order, n in ((3, 9), (2, 6)):
            with self.subTest(order=order):
                m = api.make_model(self._spec(order, n))
                z = np.ravel(m.h(m.x0))
                self.assertEqual(z.shape, (3,))
                self.assertAlmostEqual(float(z[0]), 6771e3, places=3)

    def test_orbital_gravity_is_order_independent(self):
        """g=None selects GM/r^3; acceleration must not depend on the state layout."""
        for order, n in ((3, 9), (2, 6)):
            with self.subTest(order=order):
                m = api.make_model(self._spec(order, n))
                x = np.zeros((n, 1))
                x[0, 0] = 6771e3
                x[order + 1, 0] = 7660.0
                self.assertAlmostEqual(float(np.ravel(m.f(x))[1]), -8.694, places=3)

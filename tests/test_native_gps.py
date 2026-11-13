"""The GPS receiver and its C++ twin (Kalman::Apps::GPS::Receiver), and the named
arrays a twin is configured with (one orbit per satellite).

Skipped when libKalmanApps is not built (``make build``).
"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

import modules.framework.models.native as native
from modules.framework.native import core
from modules.framework.models.native import NativeModel, apps_library_names
from modules.apps.gps import GPSModel

HAVE = "gps" in apps_library_names()
needs_apps = unittest.skipUnless(HAVE, "libKalmanApps is not built")


def both(make, z):
    with native.python_reference():
        python = make().filter(z)
    model = make()
    assert model.native_handle() is not None, "the twin was expected to be used"
    return python, model.filter(z)


def apart(a, b):
    """Largest difference of two records: on the state in metres, on the covariance relative."""
    xa, xb = np.asarray(a.xk, float), np.asarray(b.xk, float)
    Pa, Pb = np.asarray(a.Pk, float), np.asarray(b.Pk, float)
    return float(np.nanmax(np.abs(xa - xb))), float(np.nanmax(np.abs(Pa - Pb)) / np.nanmax(np.abs(Pa)))


@needs_apps
class TestGPSTwin(unittest.TestCase):
    # The state is 6.4e6 m: 1e-6 m is 1.6e-13 of it.
    def check(self, **kw):
        make = lambda: GPSModel("rx", **kw)
        z = make().simulate(600.0, seed=3)
        state, covariance = apart(*both(make, z))
        self.assertLess(state, 1e-6, kw)
        self.assertLess(covariance, 1e-12, kw)
        return z

    def test_fixed_constellation(self):
        self.check()

    def test_fewer_satellites_another_rate_no_correction(self):
        self.check(n_sat=5, fs=2.0, apply_relativistic=False)

    def test_satellites_rising_and_setting(self):
        """The whole pool of 24, each masked while below the cutoff: the twin has 24
        channels and the missing pseudoranges are skipped."""
        z = self.check(elevation_mask=10.0)
        self.assertEqual(z.shape[1], 24)
        self.assertGreater(int(np.isnan(z).sum()), 1000)

    def test_the_twin_takes_this_instance_constellation(self):
        model = GPSModel("rx", n_sat=6, receiver_lat=-30.0, receiver_lon=120.0)
        handle = model.native_handle()
        n, m, l = (core.ffi.new("int *") for _ in range(3))
        core.check(core.lib.kalman_model_dims(handle, n, m, l), "dims")
        self.assertEqual((n[0], m[0]), (5, 6))
        # h of the twin against h of the Python model, at a later time
        x = np.asarray(model.x0, float).reshape(-1) + np.array([10.0, -20.0, 5.0, 3.0, 0.1])
        z = np.empty(6)
        core.check(core.lib.kalman_model_h(handle, core.cdptr(np.ascontiguousarray(x)), 1234.5, core.dptr(z)), "h")
        self.assertLess(np.abs(z - model.h(x, t=1234.5).reshape(-1)).max(), 1e-6)
        H = np.empty((6, 5))
        core.check(core.lib.kalman_model_jacobian_h(handle, core.cdptr(np.ascontiguousarray(x)), 1234.5, core.dptr(H)),
                   "H")
        self.assertLess(np.abs(H - model.H(x, t=1234.5)).max(), 1e-12)

    def test_a_python_only_configuration_keeps_the_python_physics(self):
        model = GPSModel("rx", dtype=np.float32)
        self.assertIsNone(model.native_handle())


@needs_apps
class TestNamedArrays(unittest.TestCase):
    def test_read_set_and_resize(self):
        model = NativeModel("gps")
        arrays = model.arrays()
        self.assertEqual(sorted(arrays), ["arg_perigee", "eccentricity", "inclination", "mean_anomaly", "raan",
                                          "semi_major_axis"])
        self.assertEqual((model.n, model.m), (5, 8))
        for name, values in arrays.items():
            model.set_array(name, values[:5])
        self.assertEqual(model.m, 5)
        self.assertEqual(model.R().shape, (5, 5))
        with self.assertRaises(KeyError):
            model.set_array("no_such_array", [1.0])

    def test_arrays_that_disagree_are_said_before_a_run(self):
        model = NativeModel("gps")
        model.set_array("eccentricity", [0.01, 0.02])
        with self.assertRaises(RuntimeError) as raised:
            model.run_native(np.zeros((3, model.m)))
        self.assertIn("one entry per satellite", str(raised.exception))

    def test_arrays_go_through_a_model_file_and_its_settings(self):
        import tempfile
        import kalman.native as kn
        model = NativeModel("gps")
        for name, values in model.arrays().items():
            model.set_array(name, values[:6])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "gps.kw")
            document = model.export(path)
            self.assertEqual(document["kind"], "reference")
            self.assertEqual(len(document["spec"]["dynamics"]["params"]["eccentricity"]), 6)
            self.assertEqual(len(document["measurement"]), 6)
            back = kn.open_model(path)
            self.assertEqual(back.m, 6)
            np.testing.assert_array_equal(back.arrays()["raan"], model.arrays()["raan"])
            # settings: another eccentricity for each satellite
            with open(path + ".json", "w") as handle:
                json.dump({"format": "kalman-settings", "for": "gps",
                           "parameters": {"eccentricity": [0.02] * 6, "pseudorange_uncertainty": 2.0}}, handle)
            changed = kn.open_model(path)
            np.testing.assert_array_equal(changed.arrays()["eccentricity"], [0.02] * 6)
            self.assertEqual(changed.R()[0, 0], 4.0)


if __name__ == "__main__":
    unittest.main()

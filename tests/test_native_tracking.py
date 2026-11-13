"""The drift chamber and its C++ twin (Kalman::Apps::Tracking::DriftChamber): track
finding propagates in C++ instead of calling back into the Python model.

Skipped when libKalmanApps is not built (``make build``).
"""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, ".")

import modules.framework.models.native as native
from modules.framework.native import core
from modules.framework.models.native import apps_library_names
from modules.framework.filters import CombinatorialKalmanFilter
from modules.apps.tracking import DriftChamberModel

HAVE = "drift-chamber" in apps_library_names()
needs_apps = unittest.skipUnless(HAVE, "libKalmanApps is not built")


def find(model, event):
    ckf = CombinatorialKalmanFilter(model)
    ckf.forward(event.hits, layer_radii=event.radii)
    ckf.backward()
    return ckf


@needs_apps
class TestDriftChamberTwin(unittest.TestCase):
    # The search compares chi-squares: both sides must make the same choices, and
    # then agree on each track. One propagation ends in a bisection onto the
    # layer and both sides difference it with a step of 1e-6, which leaves about
    # 1e-6 cm on a track of 100 cm (the callback path shows the same).
    def check(self, model_kw, event_kw):
        make = lambda: DriftChamberModel(**model_kw)
        event = make().generate_event(**event_kw)
        with native.python_reference():
            python = find(make(), event)
        model = make()
        self.assertIsNotNone(model.native_handle())
        twin = find(model, event)
        self.assertEqual([t.hit_ids for t in python.tracks_raw], [t.hit_ids for t in twin.tracks_raw])
        self.assertEqual([t.hit_ids for t in python.candidates], [t.hit_ids for t in twin.candidates])
        for a, b in zip(python.tracks_raw, twin.tracks_raw):
            self.assertLess(np.abs(np.asarray(a.xk) - np.asarray(b.xk)).max(), 1e-5)
            self.assertLess(abs(a.chi2 - b.chi2), 1e-3 * max(1.0, a.chi2))
        for a, b in zip(python.tracks, twin.tracks):            # smoothed records
            self.assertLess(np.abs(np.asarray(a.xk) - np.asarray(b.xk)).max(), 1e-5)
        return python, twin

    def test_three_tracks_in_background(self):
        _, twin = self.check({}, dict(n_tracks=3, n_background=4, seed=0))
        self.assertGreaterEqual(len(twin.tracks_raw), 3)

    def test_holes_another_field_more_layers(self):
        self.check(dict(n_layers=12, Bz=1.0), dict(n_tracks=6, n_background=10, efficiency=0.9, seed=5))

    def test_one_propagation_agrees(self):
        model = DriftChamberModel(Bz=1.2, q=-1.0, step=0.25)
        handle = model.native_handle()
        x = np.array([3.0, 1.0, 0.6, 0.5])
        out = np.empty(4)
        core.check(core.lib.kalman_model_f(handle, core.cdptr(x), core.ffi.NULL, 68.0, core.dptr(out)), "f")
        self.assertLess(np.abs(out - model.f(x, t=68.0).reshape(-1)).max(), 1e-12)
        self.assertAlmostEqual(float(np.hypot(out[0], out[1])), 68.0, places=9)
        # a track that curls back before the layer
        slow = np.array([3.0, 1.0, 0.02, 0.01])
        core.check(core.lib.kalman_model_f(handle, core.cdptr(slow), core.ffi.NULL, 104.0, core.dptr(out)), "f")
        self.assertTrue(np.all(np.isnan(out)) and np.all(np.isnan(model.f(slow, t=104.0))))

    def test_faster_than_calling_back_into_python(self):
        make = lambda: DriftChamberModel(n_layers=12, Bz=1.0)
        event = make().generate_event(n_tracks=6, n_background=10, efficiency=0.9, seed=5)
        find(make(), event)                                     # load and warm up
        start = time.perf_counter(); find(make(), event); twin = time.perf_counter() - start
        os.environ["KALMAN_NATIVE_TWINS"] = "0"
        try:
            start = time.perf_counter(); find(make(), event); callback = time.perf_counter() - start
        finally:
            os.environ["KALMAN_NATIVE_TWINS"] = "1"
        self.assertLess(twin * 5, callback)


if __name__ == "__main__":
    unittest.main()

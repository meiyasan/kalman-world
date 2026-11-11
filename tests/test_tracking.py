"""Tests for the Combinatorial Kalman Filter (CKF) and the tracking detector model."""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.apps import DriftChamberModel
from modules.framework.filters import CombinatorialKalmanFilter
from modules.framework.engines.combinatorial import CombinatorialKalmanEngine


def _no_nan_inf(arr, label="array"):
    arr = np.asarray(arr)
    assert not np.any(np.isnan(arr)), f"{label} contains NaN"
    assert not np.any(np.isinf(arr)), f"{label} contains Inf"


def _majority_truth(raw, event):
    ids = [int(event.truth[l][j]) for (l, j) in raw.hit_ids]
    sig = [i for i in ids if i >= 0]
    return (max(set(sig), key=sig.count) if sig else -1), ids


# ===========================================================================
# Model
# ===========================================================================
class TestTrackingModel(unittest.TestCase):

    def test_dims(self):
        m = DriftChamberModel(n_layers=8)
        self.assertEqual((m.n, m.m, m.l), (4, 2, 0))
        self.assertEqual(len(m.radii), 8)

    def test_f_h_shapes(self):
        m = DriftChamberModel(n_layers=6)
        x = np.array([[1.0], [0.0], [1.0], [0.2]])
        xp = m.f(x, t=float(m.radii[2]))
        self.assertEqual(xp.shape, (4, 1))
        self.assertEqual(m.h(xp).shape, (2, 1))

    def test_propagation_lands_on_radius(self):
        m = DriftChamberModel(n_layers=8, Bz=1.5)
        s = np.array([0.0, 0.0, 1.0, 0.3])
        for r in m.radii:
            s = m.propagate_to_radius(s, float(r))
            self.assertIsNotNone(s)
            self.assertAlmostEqual(float(np.hypot(s[0], s[1])), float(r), places=4)

    def test_default_engine_is_combinatorial(self):
        from modules.framework.constants import KalmanEngineKind
        self.assertEqual(DriftChamberModel().DEFAULT_ENGINE, KalmanEngineKind.COMBINATORIAL)


# ===========================================================================
# Engine gate
# ===========================================================================
class TestGate(unittest.TestCase):

    def test_gate_separates_on_and_off_track(self):
        m = DriftChamberModel(n_layers=8, Bz=1.5)
        ev = m.generate_event(n_tracks=1, n_background=0, seed=3)
        eng = CombinatorialKalmanEngine()
        x0, P0 = m.seed_state(ev.hits[0][0])
        r1 = float(m.radii[1])
        xp, Pp, _ = eng._predict(m, x0, P0, t=r1)
        on = ev.tracks[0][1][:2].reshape(2, 1)             # true position at layer 1
        off = on + np.array([[6.0], [6.0]])
        chi2_on, _, _ = eng.gate(m, xp, Pp, on, t=r1)
        chi2_off, _, _ = eng.gate(m, xp, Pp, off, t=r1)
        cut = CombinatorialKalmanFilter(m).chi2_cut
        self.assertLess(chi2_on, cut)
        self.assertGreater(chi2_off, cut)


# ===========================================================================
# Track finding
# ===========================================================================
class TestCKFFinding(unittest.TestCase):

    def test_clean_event_recovers_every_track(self):
        m = DriftChamberModel(n_layers=8, Bz=1.5)
        ev = m.generate_event(n_tracks=3, n_background=0, efficiency=1.0, seed=2)
        ckf = CombinatorialKalmanFilter(m, max_candidates=8, min_hits=4)
        tracks = ckf.forward(ev.hits, layer_radii=ev.radii)
        self.assertEqual(len(tracks), ev.n_tracks)
        found = {_majority_truth(raw, ev)[0] for raw in ckf.tracks_raw}
        self.assertEqual(found, set(range(ev.n_tracks)))
        for t in tracks:
            self.assertEqual(len(t), 8)                    # every layer found
            _no_nan_inf(t.xk, "xk"); _no_nan_inf(t.Pk, "Pk")

    def test_multitrack_with_background(self):
        # All three signal tracks recovered despite background, across several events.
        for seed in range(5):
            m = DriftChamberModel(n_layers=8, Bz=1.5)
            ev = m.generate_event(n_tracks=3, n_background=4, seed=seed)
            ckf = CombinatorialKalmanFilter(m, max_candidates=8, min_hits=4)
            ckf.forward(ev.hits, layer_radii=ev.radii)
            found = {_majority_truth(raw, ev)[0] for raw in ckf.tracks_raw}
            self.assertTrue(set(range(3)).issubset(found), f"seed {seed}: found {found}")

    def test_arbitration_yields_disjoint_tracks(self):
        m = DriftChamberModel(n_layers=8, Bz=1.5)
        ev = m.generate_event(n_tracks=3, n_background=4, seed=0)
        ckf = CombinatorialKalmanFilter(m, max_candidates=8, min_hits=4, shared_hit_frac=0.0)
        ckf.forward(ev.hits, layer_radii=ev.radii)
        seen = set()
        for raw in ckf.tracks_raw:
            ids = set(raw.hit_ids)
            self.assertEqual(len(ids & seen), 0, "accepted tracks share a hit")
            seen |= ids

    def test_pure_background_makes_no_full_length_track(self):
        # No genuine track exists; requiring (almost) all layers rejects random background.
        for seed in range(5):
            m = DriftChamberModel(n_layers=8, Bz=1.5)
            ev = m.generate_event(n_tracks=0, n_background=5, seed=seed)
            ckf = CombinatorialKalmanFilter(m, max_candidates=8, min_hits=7)
            tracks = ckf.forward(ev.hits, layer_radii=ev.radii)
            self.assertEqual(len(tracks), 0, f"seed {seed}: background faked {len(tracks)} tracks")


# ===========================================================================
# Smoothing
# ===========================================================================
class TestCKFSmooth(unittest.TestCase):

    def test_backward_runs_and_is_finite(self):
        m = DriftChamberModel(n_layers=8, Bz=1.5)
        ev = m.generate_event(n_tracks=2, n_background=3, seed=1)
        ckf = CombinatorialKalmanFilter(m, min_hits=4)
        ckf.forward(ev.hits, layer_radii=ev.radii)
        smoothed = ckf.backward()
        self.assertEqual(len(smoothed), len(ckf.tracks_raw))
        for s in smoothed:
            self.assertEqual(s.xk.shape[1], 4)
            _no_nan_inf(s.xk, "smoothed xk"); _no_nan_inf(s.Pk, "smoothed Pk")


if __name__ == "__main__":
    unittest.main()

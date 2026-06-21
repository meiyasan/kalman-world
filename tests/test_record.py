"""Tests for the IO data model: ``KalmanRecord`` (the posterior-centric container every
filter/sample returns) and ``KalmanRecordCollection``.

Covers the access contract notebooks rely on — dimensions, indexing, iteration,
immutability and the construction-time validation — independently of plotting.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")
from modules.framework import LinearKalmanModel
from modules.framework.io.record import KalmanRecord
from modules.framework.io.collection import KalmanRecordCollection
from modules.apps import SuspensionModel


DT = 0.1
F_CV = np.array([[1.0, DT], [0.0, 1.0]])
H_POS = np.array([[1.0, 0.0]])


def _cv_record(seed=0):
    np.random.seed(seed)
    m = LinearKalmanModel("s", F=F_CV, H=H_POS,
                          Q=np.eye(2) * 1e-3, R=np.array([[0.1]]), fs=10)
    return m, m.filter(m.sample(2.0).zk)


# ===========================================================================
# KalmanRecord data model
# ===========================================================================
class TestRecordDataModel(unittest.TestCase):

    def test_dimensions(self):
        _, rec = _cv_record()
        self.assertEqual(rec.n, 2)
        self.assertEqual(rec.m, 1)
        self.assertEqual(rec.dim, 1)
        self.assertEqual(len(rec), 20)

    def test_time_properties(self):
        _, rec = _cv_record()
        self.assertAlmostEqual(rec.sample_rate, 10.0, places=6)
        self.assertAlmostEqual(rec.duration, 2.0, places=6)

    def test_indexing_and_iteration(self):
        _, rec = _cv_record()
        self.assertEqual(np.asarray(rec[0]).shape, (2,))            # __getitem__ -> state row
        first = next(iter(rec))                                     # (t, xk, Pk, zk) tuple
        self.assertEqual(len(first), 4)
        self.assertEqual(first[0], rec.time[0])
        self.assertEqual(sum(1 for _ in rec), len(rec))

    def test_arrays_are_frozen(self):
        _, rec = _cv_record()
        with self.assertRaises(ValueError):
            rec.xk[0, 0] = 1.234                                    # frozen=True dataclass + read-only arrays

    def test_validation_rejects_nan(self):
        m, _ = _cv_record()
        with self.assertRaises(AssertionError):
            KalmanRecord(np.arange(3), xk=np.array([[1.0], [np.nan], [2.0]]), model=m)

    def test_validation_rejects_wrong_state_dim(self):
        m, _ = _cv_record()                                         # model has n=2
        with self.assertRaises(AssertionError):
            KalmanRecord(np.arange(3), xk=np.zeros((3, 5)), model=m)

    def test_repr_and_print_do_not_crash(self):
        _, rec = _cv_record()
        self.assertIsInstance(repr(rec), str)
        self.assertIsInstance(str(rec), str)


# ===========================================================================
# Component resolution (enum / name / int)
# ===========================================================================
class TestComponentResolution(unittest.TestCase):

    def test_resolve_int_and_enum(self):
        s = SuspensionModel("road", fs=256)
        rec = s.filter(0.001 * np.random.RandomState(0).randn(64, 1),
                       np.zeros((64, 1)))
        state0 = list(s.states)[0]
        self.assertEqual(rec._resolve_component_symbol(0), 0)        # int passes through
        self.assertIs(rec._resolve_component_symbol(state0), state0)  # enum passes through

    def test_normalize_component_accepts_enum_member(self):
        s = SuspensionModel("road", fs=256)
        rec = s.filter(0.001 * np.random.RandomState(1).randn(32, 1), np.zeros((32, 1)))
        member = list(s.states)[0]
        normalized = rec._normalize_component(member)
        self.assertEqual(len(normalized), 1)
        self.assertIs(normalized[0], member)


# ===========================================================================
# KalmanRecordCollection
# ===========================================================================
class TestRecordCollection(unittest.TestCase):

    def test_add_remove_clear_len(self):
        _, r1 = _cv_record(0)
        _, r2 = _cv_record(1)
        col = KalmanRecordCollection([r1])
        self.assertEqual(len(col), 1)
        col.add(r2)
        self.assertEqual(len(col), 2)
        self.assertEqual(col.list(), [0, 1])
        col.remove(r1)
        self.assertEqual(len(col), 1)
        col.clear()
        self.assertEqual(len(col), 0)

    def test_getitem_iter_and_bounds(self):
        _, r1 = _cv_record(0)
        _, r2 = _cv_record(1)
        col = KalmanRecordCollection([r1, r2])
        self.assertIs(col[0], r1)
        self.assertIs(col[1], r2)
        self.assertIsNone(col[5])                # out-of-range returns None (not IndexError)
        self.assertEqual(list(col), [r1, r2])

    def test_repr_and_str(self):
        _, r1 = _cv_record(0)
        col = KalmanRecordCollection([r1])
        self.assertEqual(repr(col), "KalmanRecordCollection(N=1)")
        self.assertIn("1 records", str(col))


if __name__ == "__main__":
    unittest.main()

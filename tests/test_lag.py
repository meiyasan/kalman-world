"""lag: a model that carries its last states.

``model.lag(L)`` (``lag(model, L)``, ``Lag``) is ``Kalman::Models::Lag`` seen from
Python. Pinned here: the three ways in, the stacked state and its names, a
sensor that reports late (``KalmanSensor(delay=...)``) and what happens without
the lag, fixed-lag smoothing, and nesting with the other operations. The same
scenarios are in ``tests/cxx/26_lag.cc``.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework import KalmanModel, KalmanSensor, Lag, lag
from modules.framework.constants import KalmanStructureKind
from modules.framework.native import core

needs_core = unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")

FS = 10.0


def cart():
    dt, q = 1 / FS, 1.0
    Q = q * np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    return KalmanModel("cart", fs=FS, F=np.array([[1, dt], [0, 1]]), H=np.array([[1.0, 0.0]]), Q=Q,
                       R=np.array([[0.25]]), x0=np.array([[0.0], [1.0]]), P0=np.eye(2))


def late(delay):
    return KalmanSensor("late", G=[[1.0, 0.0]], R=0.01, labels={"X_LATE": "m"}, delay=delay)


def simulate(model, steps, seed):
    """Truth and readings of a lagged model, from its own f and h."""
    rng = np.random.default_rng(seed)
    n0 = model.n_inner
    Lq = np.linalg.cholesky(model.Q()[:n0, :n0] + 1e-15 * np.eye(n0))
    Lr = np.linalg.cholesky(model.R())
    x, xs, zs = np.asarray(model.x0, dtype=float).copy(), [], []
    for _ in range(steps):
        x = model.f(x)
        x[:n0] += Lq @ rng.standard_normal((n0, 1))
        xs.append(x.ravel().copy())
        zs.append((model.h(x) + Lr @ rng.standard_normal((model.m, 1))).ravel())
    return np.array(xs), np.array(zs)


rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))


@needs_core
class TestLag(unittest.TestCase):

    def test_three_ways(self):
        for model in (cart().lag(3), lag(cart(), 3), Lag(cart(), 3)):
            self.assertIsInstance(model, Lag)
            self.assertEqual((model.n, model.m, model.depth, model.n_inner), (8, 1, 3, 2))

    def test_state_and_names(self):
        model = cart().lag(2)
        names = list(model.states.__members__)
        self.assertEqual(len(names), 6)
        self.assertEqual(names[2:], [names[0] + "@-1", names[1] + "@-1", names[0] + "@-2", names[1] + "@-2"])
        np.testing.assert_array_equal(model.x0.ravel(), [0, 1, 0, 1, 0, 1])
        self.assertIs(model.structure, KalmanStructureKind.MATRICES)       # a model of matrices stays one
        self.assertEqual(model.block(2), slice(4, 6))
        with self.assertRaises(IndexError):
            model.block(3)

    def test_step_shifts(self):
        model = cart().lag(2)
        x = np.arange(1.0, 7.0).reshape(-1, 1)
        out = model.f(x).ravel()
        np.testing.assert_allclose(out, [1 + 2 / FS, 2, 1, 2, 3, 4])
        np.testing.assert_array_equal(model.h(x).ravel(), [1.0])
        self.assertEqual(model.Q()[2, 2], 0.0)

    def test_refusals(self):
        for depth in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                cart().lag(depth)
        with self.assertRaises(TypeError):
            Lag("cart", 2)
        with self.assertRaises(ValueError):
            KalmanSensor("s", G=[[1.0, 0.0]], R=1.0, delay=-0.1)
        with self.assertRaises(ValueError):                                 # later than the model carries
            cart().add_sensor(late(0.3)).lag(2)
        with self.assertRaises(ValueError):                                 # declared with the model: no past state
            KalmanModel([late(0.3)], fs=FS, F=np.eye(2), Q=np.eye(2), x0=np.zeros((2, 1)), P0=np.eye(2))

    def test_fixed_lag_smoothing(self):
        model = cart().lag(3)
        truth, zk = simulate(model, 2000, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rec = model.filter(zk, smooth=False, reset=True)
        xk = np.asarray(rec.xk).reshape(len(zk), -1)
        now, then = rmse(xk[:, 0], truth[:, 0]), rmse(xk[:, model.block(3)][:, 0], truth[:, 6])
        self.assertLess(then, 0.9 * now)


@needs_core
class TestLateSensor(unittest.TestCase):

    def test_reads_the_state_as_it_was(self):
        model = cart().add_sensor(late(0.3)).lag(3)
        self.assertEqual([s.delay for s in model.inner.sensors], [None, 0.3])
        self.assertEqual([s.delay for s in model.sensors], [None, None])     # on the lagged state none is late
        x = np.arange(1.0, 9.0).reshape(-1, 1)
        np.testing.assert_array_equal(model.h(x).ravel(), [1.0, 7.0])
        H = model.H()
        self.assertEqual((H[0, 0], H[1, 6], H[1, 0]), (1.0, 1.0, 0.0))
        self.assertEqual(float(model.sensors[1].measure(x)[0, 0]), 7.0)

    def test_without_the_lag_it_is_refused_at_the_pass(self):
        model = cart().add_sensor(late(0.3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(Exception) as raised:
                model.filter(np.zeros((5, 2)), smooth=False, reset=True)
        self.assertIn("late", str(raised.exception))

    def test_the_late_sensor_helps_and_ignoring_its_delay_hurts(self):
        model = cart().add_sensor(late(0.3)).lag(3)
        truth, zk = simulate(model, 2000, 2)
        naive = cart().add_sensor(late(None))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            good = np.asarray(model.filter(zk, smooth=False, reset=True).xk).reshape(len(zk), -1)
            bad = np.asarray(naive.filter(zk, smooth=False, reset=True).xk).reshape(len(zk), -1)
            alone = np.asarray(cart().filter(zk[:, :1], smooth=False, reset=True).xk).reshape(len(zk), -1)
        e_lag, e_naive, e_alone = (rmse(x[:, 0], truth[:, 0]) for x in (good, bad, alone))
        self.assertLess(e_lag, 0.8 * e_alone)
        self.assertLess(e_lag, 0.8 * e_naive)

    def test_a_function_sensor(self):
        squared = KalmanSensor("squared", lambda x: [float(x[0, 0]) ** 2], R=1.0, delay=0.1)
        model = cart().add_sensor(squared).lag(1)
        x = np.array([[1.0], [0.0], [3.0], [0.0]])
        np.testing.assert_array_equal(model.h(x).ravel(), [1.0, 9.0])
        self.assertAlmostEqual(float(model.H(x)[1, 2]), 6.0, places=4)

    def test_they_nest(self):
        station = cart().add_sensor(late(0.2)).lag(2)
        twice = station.lag(1)
        self.assertEqual(twice.n, 12)
        x = np.zeros((12, 1))
        x[4, 0] = 7.0
        self.assertEqual(float(twice.h(x)[1, 0]), 7.0)                      # the delay is applied once


if __name__ == "__main__":
    unittest.main()

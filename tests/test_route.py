"""route: several models of one system, with a probability on each.

``route(models, Pi)`` (``KalmanModel.route``, ``Router``) is ``Kalman::Models::Router``
seen from Python, run by the ordinary filter on the interacting engine
(``Kalman::Engines::Interacting``, the IMM of the literature). Pinned here: the
ways in, the record and the mode probabilities, the decision layers and their
training, the accuracy on a switching process, and the refusals. The same in
C++: ``tests/cxx/05_route.cc``.
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.framework.model import KalmanModel
from modules.framework import Router, route
from modules.framework.constants import KalmanEngineKind
from modules.framework.routers          import (BayesianRouter, ThresholdRouter,
                                                 NeuralRouter, HybridRouter)
from modules.framework.routers.nn.mlp  import MLP


# ── shared fixtures ───────────────────────────────────────────────────────────

def _two_models(q_slow=0.1, q_fast=2.0):
    """Two 1-D random-walk models with very different process noise."""
    def _m(name, q):
        return KalmanModel(name,
                           F=np.eye(1), H=np.eye(1),
                           Q=np.array([[q ** 2]]),
                           R=np.array([[1.0 ** 2]]),
                           fs=1.0,
                           x0=np.array([[0.0]]))
    return _m("slow", q_slow), _m("fast", q_fast)


def _sim(regime_seq, q_slow=0.1, q_fast=2.0, obs_std=1.0, seed=0):
    """Simulate a 1-D switching process and noisy observations."""
    rng = np.random.default_rng(seed)
    x = 0.0
    zk, labels = [], []
    for re in regime_seq:
        x += rng.normal(0, q_slow if re == 0 else q_fast)
        zk.append([x + rng.normal(0, obs_std)])
        labels.append(re)
    return np.array(zk), np.array(labels, dtype=int)


PI = np.array([[0.95, 0.05], [0.05, 0.95]])

REGIME_200 = np.r_[np.zeros(100, int), np.ones(100, int)]
REGIME_200_TEST = np.r_[np.zeros(70, int), np.ones(60, int), np.zeros(70, int)]


# ── basic API ─────────────────────────────────────────────────────────────────

class TestRouteBasicAPI(unittest.TestCase):

    def _ikf(self, **kw):
        m0, m1 = _two_models()
        return route([m0, m1], PI, **kw)

    def test_forward_returns_record(self):
        from modules.framework.io.record import KalmanRecord
        ikf = self._ikf()
        zk, _ = _sim(REGIME_200)
        rec = ikf.filter(zk)
        self.assertIsInstance(rec, KalmanRecord)

    def test_record_shape(self):
        ikf = self._ikf()
        T   = 60
        zk, _ = _sim(np.zeros(T, int))
        rec   = ikf.filter(zk)
        self.assertEqual(rec.xk.shape, (T, 1))

    def test_mode_probs_shape_and_sum(self):
        T   = 50
        ikf = self._ikf()
        zk, _ = _sim(np.zeros(T, int))
        ikf.filter(zk)
        mu = ikf.mode_probabilities
        self.assertEqual(mu.shape, (T, 2))
        np.testing.assert_allclose(mu.sum(axis=1), 1.0, atol=1e-9)

    def test_mode_probs_non_negative(self):
        ikf = self._ikf()
        zk, _ = _sim(REGIME_200)
        ikf.filter(zk)
        self.assertTrue(np.all(ikf.mode_probabilities >= 0))

    def test_default_router_is_bayesian(self):
        ikf = self._ikf()
        self.assertIsInstance(ikf.router, BayesianRouter)

    def test_every_pass_starts_again(self):
        ikf = self._ikf()
        zk, _ = _sim(REGIME_200)
        first = ikf.filter(zk).xk.copy()
        mu = ikf.mode_probabilities.copy()
        np.testing.assert_array_equal(ikf.filter(zk).xk, first)
        np.testing.assert_array_equal(ikf.mode_probabilities, mu)
        self.assertEqual(ikf.nis_history.shape, (200, 2))

    def test_three_ways_and_what_it_is(self):
        m0, m1 = _two_models()
        for model in (route([m0, m1], PI), KalmanModel.route([m0, m1], PI), Router([m0, m1], PI)):
            self.assertIsInstance(model, Router)
            self.assertIs(model.DEFAULT_ENGINE, KalmanEngineKind.INTERACTING)
            self.assertEqual((model.n, model.m), (1, 1))
            self.assertEqual(model.mode_names, ["slow", "fast"])
        zk, _ = _sim(REGIME_200)
        with self.assertRaises(Exception):                       # no smoother on several modes
            model.filter(zk, smooth=True)

    def test_initial_probabilities(self):
        m0, m1 = _two_models()
        zk, _ = _sim(np.zeros(3, int))
        sure = route([m0, m1], PI, [0.999, 0.001])
        sure.filter(zk)
        even = route(list(_two_models()), PI)
        even.filter(zk)
        self.assertGreater(sure.mode_probabilities[0, 0], even.mode_probabilities[0, 0])

    def test_refusals(self):
        m0, m1 = _two_models()
        with self.assertRaises(ValueError):
            route([m0], np.eye(1))
        with self.assertRaises(ValueError):
            route([m0, m1], np.eye(3))
        with self.assertRaises(ValueError):
            route([m0, m1], np.full((2, 2), 0.7))                # rows do not sum to 1
        with self.assertRaises(ValueError):
            route([m0, m1], PI, [1.0])
        with self.assertRaises(TypeError):
            route([m0, "fast"], PI)
        wide = KalmanModel("wide", F=np.eye(2), H=np.eye(2), Q=np.eye(2), R=np.eye(2), fs=1.0, x0=np.zeros((2, 1)))
        with self.assertRaises(ValueError):
            route([m0, wide], PI)

    def test_a_router_that_raises_is_reported_as_itself(self):
        m0, m1 = _two_models()

        def broken(nis, c, likelihoods):
            raise KeyError("no such regime")
        zk, _ = _sim(np.zeros(5, int))
        with self.assertRaises(KeyError):
            route([m0, m1], PI, router=broken).filter(zk)

    def test_mixing_starts_every_mode_from_the_beliefs_before_the_step(self):
        # One step by hand, for two random walks: mode j starts from the blend
        # of both beliefs as they were, with weights Pi_ij mu_i / c_j.
        m0, m1 = _two_models()
        z = np.array([[0.7]])
        model = route([m0, m1], PI)
        model.filter(z)
        mu, c = np.array([0.5, 0.5]), PI.T @ np.array([0.5, 0.5])
        P = [float(m0.P0[0, 0]), float(m1.P0[0, 0])]
        post, like = [], []
        for j, q in enumerate((0.1 ** 2, 2.0 ** 2)):
            w = PI[:, j] * mu / c[j]
            S = w @ P + q + 1.0
            like.append(np.exp(-0.5 * 0.7 ** 2 / S) / np.sqrt(2 * np.pi * S))
            post.append((w @ P + q) / S * 0.7)
        want = c * np.array(like)
        want /= want.sum()
        np.testing.assert_allclose(model.mode_probabilities[0], want, rtol=1e-12)
        np.testing.assert_allclose(float(model._filter.x[0, 0]), want @ np.array(post), rtol=1e-12)

    def test_mode_probs_empty_before_forward(self):
        ikf = self._ikf()
        mu = ikf.mode_probabilities
        self.assertEqual(mu.shape, (0, 2))

    def test_filter_alias_works(self):
        ikf = self._ikf()
        zk, _ = _sim(np.zeros(20, int))
        rec = ikf.filter(zk)
        from modules.framework.io.record import KalmanRecord
        self.assertIsInstance(rec, KalmanRecord)

    def test_no_nan_inf_in_estimates(self):
        ikf = self._ikf()
        zk, _ = _sim(REGIME_200)
        rec = ikf.filter(zk)
        self.assertFalse(np.any(np.isnan(rec.xk)))
        self.assertFalse(np.any(np.isinf(rec.xk)))


# ── router fit API ────────────────────────────────────────────────────────────

class TestRouteFitAPI(unittest.TestCase):

    def _data(self, n_sequences=3, seed_start=0):
        train_zk, train_lbl = [], []
        for s in range(n_sequences):
            zk, lbl = _sim(REGIME_200, seed=seed_start + s)
            train_zk.append(zk)
            train_lbl.append(lbl)
        return train_zk, train_lbl

    def test_fit_returns_self(self):
        m0, m1 = _two_models()
        ikf = route([m0, m1], PI, router=NeuralRouter(MLP(4), window=3))
        train_zk, train_lbl = self._data()
        result = ikf.fit(train_zk, train_lbl, epochs=20)
        self.assertIs(result, ikf)

    def test_fit_trains_neural_router(self):
        m0, m1 = _two_models()
        router  = NeuralRouter(MLP(4), window=3)
        ikf     = route([m0, m1], PI, router=router)
        train_zk, train_lbl = self._data()
        ikf.fit(train_zk, train_lbl, epochs=30)
        self.assertTrue(router._is_trained)

    def test_fit_noop_for_bayesian(self):
        m0, m1 = _two_models()
        router  = BayesianRouter()
        ikf     = route([m0, m1], PI, router=router)
        train_zk, train_lbl = self._data()
        ikf.fit(train_zk, train_lbl)
        # No error; BayesianRouter.fit is a no-op
        self.assertIsInstance(router, BayesianRouter)


# ── routing accuracy ──────────────────────────────────────────────────────────

class TestRoutingAccuracy(unittest.TestCase):
    """Routing accuracy on the 2-model Gaussian benchmark."""

    def _run(self, router, epochs=200, n_train=3, seed=42):
        m0, m1 = _two_models()
        ikf = route([m0, m1], PI, router=router)

        train_zk, train_lbl = [], []
        for s in range(n_train):
            zk, lbl = _sim(REGIME_200, seed=s)
            train_zk.append(zk); train_lbl.append(lbl)

        ikf.fit(train_zk, train_lbl, epochs=epochs, seed=seed)

        zk_test, lbl_test = _sim(REGIME_200_TEST, seed=99)
        ikf.filter(zk_test)
        mu = ikf.mode_probabilities
        acc = float((np.argmax(mu, axis=1) == lbl_test).mean())
        return acc

    def test_bayesian_accuracy_high(self):
        # The Bayesian decision should reach ≥85% on clean Gaussian data
        acc = self._run(BayesianRouter())
        self.assertGreater(acc, 0.85, f"Bayesian accuracy {acc:.1%} too low")

    def test_threshold_accuracy_reasonable(self):
        acc = self._run(ThresholdRouter(m=1, window=5, soft=True))
        self.assertGreater(acc, 0.70, f"Threshold accuracy {acc:.1%} too low")

    def test_neural_accuracy_reasonable(self):
        # After 3 training sequences, MLP should reach ≥70%
        acc = self._run(NeuralRouter(MLP(hidden=16), window=5), epochs=300)
        self.assertGreater(acc, 0.70, f"Neural accuracy {acc:.1%} too low")

    def test_hybrid_close_to_bayesian(self):
        r_bay = BayesianRouter()
        r_mlp = NeuralRouter(MLP(hidden=8), window=5)
        acc = self._run(HybridRouter(r_bay, r_mlp, alpha=0.3), epochs=200)
        # Bayesian anchor keeps accuracy high
        self.assertGreater(acc, 0.80, f"Hybrid accuracy {acc:.1%} too low")


# ── control input (uk= callable) ──────────────────────────────────────────────

class TestRouteWithControl(unittest.TestCase):

    def test_uk_callable_accepted(self):
        m0, m1 = _two_models()
        ikf = route([m0, m1], PI)
        zk, _ = _sim(np.zeros(20, int))
        zero_ctrl = lambda x, P, t, nu=None: np.zeros((1, 1))
        rec = ikf.filter(zk, uk=zero_ctrl)
        self.assertFalse(np.any(np.isnan(rec.xk)))


if __name__ == "__main__":
    unittest.main()

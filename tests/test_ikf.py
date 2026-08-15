"""Tests for InteractiveKalmanFilter and the full IMM pipeline."""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.framework.model import KalmanModel
from modules.framework.filters.ikf import InteractiveKalmanFilter
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

class TestIKFBasicAPI(unittest.TestCase):

    def _ikf(self, **kw):
        m0, m1 = _two_models()
        return InteractiveKalmanFilter([m0, m1], PI, **kw)

    def test_forward_returns_record(self):
        from modules.framework.io.record import KalmanRecord
        ikf = self._ikf()
        zk, _ = _sim(REGIME_200)
        rec = ikf.forward(zk)
        self.assertIsInstance(rec, KalmanRecord)

    def test_record_shape(self):
        ikf = self._ikf()
        T   = 60
        zk, _ = _sim(np.zeros(T, int))
        rec   = ikf.forward(zk)
        self.assertEqual(rec.xk.shape, (T, 1))

    def test_mode_probs_shape_and_sum(self):
        T   = 50
        ikf = self._ikf()
        zk, _ = _sim(np.zeros(T, int))
        ikf.forward(zk)
        mu = ikf.mode_probabilities
        self.assertEqual(mu.shape, (T, 2))
        np.testing.assert_allclose(mu.sum(axis=1), 1.0, atol=1e-9)

    def test_mode_probs_non_negative(self):
        ikf = self._ikf()
        zk, _ = _sim(REGIME_200)
        ikf.forward(zk)
        self.assertTrue(np.all(ikf.mode_probabilities >= 0))

    def test_default_router_is_bayesian(self):
        ikf = self._ikf()
        self.assertIsInstance(ikf.router, BayesianRouter)

    def test_reset_clears_state(self):
        ikf = self._ikf()
        zk, _ = _sim(REGIME_200)
        ikf.forward(zk)
        self.assertGreater(len(ikf.mu_log), 0)
        ikf.reset()
        self.assertEqual(len(ikf.mu_log), 0)
        self.assertEqual(len(ikf.nis_log), 0)
        self.assertEqual(len(ikf.combined_posterior), 0)

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
        rec = ikf.forward(zk)
        self.assertFalse(np.any(np.isnan(rec.xk)))
        self.assertFalse(np.any(np.isinf(rec.xk)))


# ── router fit API ────────────────────────────────────────────────────────────

class TestIKFFitAPI(unittest.TestCase):

    def _data(self, n_sequences=3, seed_start=0):
        train_zk, train_lbl = [], []
        for s in range(n_sequences):
            zk, lbl = _sim(REGIME_200, seed=seed_start + s)
            train_zk.append(zk)
            train_lbl.append(lbl)
        return train_zk, train_lbl

    def test_fit_returns_self(self):
        m0, m1 = _two_models()
        ikf = InteractiveKalmanFilter([m0, m1], PI,
                                      router=NeuralRouter(MLP(4), window=3))
        train_zk, train_lbl = self._data()
        result = ikf.fit(train_zk, train_lbl, epochs=20)
        self.assertIs(result, ikf)

    def test_fit_trains_neural_router(self):
        m0, m1 = _two_models()
        router  = NeuralRouter(MLP(4), window=3)
        ikf     = InteractiveKalmanFilter([m0, m1], PI, router=router)
        train_zk, train_lbl = self._data()
        ikf.fit(train_zk, train_lbl, epochs=30)
        self.assertTrue(router._is_trained)

    def test_fit_noop_for_bayesian(self):
        m0, m1 = _two_models()
        router  = BayesianRouter()
        ikf     = InteractiveKalmanFilter([m0, m1], PI, router=router)
        train_zk, train_lbl = self._data()
        ikf.fit(train_zk, train_lbl)
        # No error; BayesianRouter.fit is a no-op
        self.assertIsInstance(router, BayesianRouter)


# ── routing accuracy ──────────────────────────────────────────────────────────

class TestRoutingAccuracy(unittest.TestCase):
    """Routing accuracy on the 2-model Gaussian benchmark."""

    def _run(self, router, epochs=200, n_train=3, seed=42):
        m0, m1 = _two_models()
        ikf = InteractiveKalmanFilter([m0, m1], PI, router=router)

        train_zk, train_lbl = [], []
        for s in range(n_train):
            zk, lbl = _sim(REGIME_200, seed=s)
            train_zk.append(zk); train_lbl.append(lbl)

        ikf.fit(train_zk, train_lbl, epochs=epochs, seed=seed)

        zk_test, lbl_test = _sim(REGIME_200_TEST, seed=99)
        ikf.forward(zk_test)
        mu = ikf.mode_probabilities
        acc = float((np.argmax(mu, axis=1) == lbl_test).mean())
        return acc

    def test_bayesian_accuracy_high(self):
        # Classical IMM should reach ≥85% on clean Gaussian data
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

class TestIKFWithControl(unittest.TestCase):

    def test_uk_callable_accepted(self):
        m0, m1 = _two_models()
        ikf = InteractiveKalmanFilter([m0, m1], PI)
        zk, _ = _sim(np.zeros(20, int))
        zero_ctrl = lambda x, P, t, nu=None: np.zeros((1, 1))
        rec = ikf.forward(zk, uk=zero_ctrl)
        self.assertFalse(np.any(np.isnan(rec.xk)))


if __name__ == "__main__":
    unittest.main()

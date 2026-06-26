"""Tests for modules/filters/routers/ — router hierarchy, MLP, and nn stubs."""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")
from modules.framework.routers.base      import KalmanRouter
from modules.framework.routers.bayesian  import BayesianRouter
from modules.framework.routers.threshold import ThresholdRouter
from modules.framework.routers.hybrid    import HybridRouter
from modules.framework.routers.neural    import NeuralRouter
from modules.framework.routers.nn.mlp   import MLP
from modules.framework.routers.nn.transformer import TransformerNN
from modules.framework.routers.nn.lstm        import LSTM
from modules.framework.routers.nn.cnn         import CNN


# ── helpers ──────────────────────────────────────────────────────────────────

def _dummy_nis(T=5, M=2, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.uniform(0, 3, (T, M))


def _is_prob_vector(mu, M):
    """Check mu is a non-negative (M,) array summing to 1."""
    mu = np.asarray(mu)
    return (mu.shape == (M,)
            and np.all(mu >= 0)
            and np.isclose(mu.sum(), 1.0, atol=1e-9))


# ── KalmanRouter contract ─────────────────────────────────────────────────────

class TestKalmanRouterContract(unittest.TestCase):

    def test_all_routers_are_subclasses(self):
        for cls in (BayesianRouter, ThresholdRouter, HybridRouter):
            self.assertTrue(issubclass(cls, KalmanRouter))

    def test_neural_router_is_subclass(self):
        self.assertTrue(issubclass(NeuralRouter, KalmanRouter))

    def test_normalise_standard(self):
        mu = KalmanRouter._normalise(np.array([1.0, 3.0]))
        np.testing.assert_allclose(mu, [0.25, 0.75])

    def test_normalise_clips_negative(self):
        mu = KalmanRouter._normalise(np.array([-1.0, 1.0]))
        self.assertTrue(np.all(mu > 0))
        self.assertAlmostEqual(mu.sum(), 1.0)

    def test_normalise_all_zeros(self):
        mu = KalmanRouter._normalise(np.array([0.0, 0.0]))
        self.assertAlmostEqual(mu.sum(), 1.0)
        np.testing.assert_allclose(mu, [0.5, 0.5])

    def test_call_delegates_to_route(self):
        router = BayesianRouter()
        nis = _dummy_nis()
        c = np.array([0.6, 0.4])
        L = np.array([0.8, 0.2])
        mu_route = router.route(nis, c, L)
        mu_call  = router(nis, c, L)
        np.testing.assert_array_equal(mu_route, mu_call)


# ── BayesianRouter ────────────────────────────────────────────────────────────

class TestBayesianRouter(unittest.TestCase):

    def setUp(self):
        self.r = BayesianRouter()

    def test_returns_prob_vector(self):
        nis = _dummy_nis(M=3)
        c   = np.array([0.3, 0.4, 0.3])
        L   = np.array([0.5, 0.3, 0.2])
        self.assertTrue(_is_prob_vector(self.r.route(nis, c, L), 3))

    def test_known_values(self):
        nis = _dummy_nis()
        c   = np.array([0.5, 0.5])
        L   = np.array([10.0, 1.0])
        mu  = self.r.route(nis, c, L)
        self.assertAlmostEqual(mu[0], 10 / 11, places=8)

    def test_fit_is_noop(self):
        nis_arr = [_dummy_nis()]
        lbl_arr = [np.array([0, 0, 1, 1, 1])]
        result = self.r.fit(nis_arr, lbl_arr)
        self.assertIs(result, self.r)


# ── ThresholdRouter ───────────────────────────────────────────────────────────

class TestThresholdRouter(unittest.TestCase):

    def test_hard_picks_closest_model(self):
        # model 0 NIS ≈ 1.0 (expected), model 1 NIS ≈ 5.0 (wrong)
        nis = np.array([[1.0, 5.0]] * 5)
        r   = ThresholdRouter(m=1, window=5, soft=False)
        mu  = r.route(nis, np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        self.assertEqual(mu.argmax(), 0)
        self.assertAlmostEqual(mu[0], 1.0)

    def test_soft_gives_continuous_weights(self):
        nis = np.array([[1.0, 5.0]] * 5)
        r   = ThresholdRouter(m=1, window=5, soft=True)
        mu  = r.route(nis, np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        self.assertTrue(_is_prob_vector(mu, 2))
        self.assertGreater(mu[0], mu[1])

    def test_window_averages_last_n(self):
        # First 5 steps model 1 is good (NIS≈1), last step model 0 flips to good
        # window=1 → should react immediately to the latest step only
        nis = np.vstack([np.tile([5.0, 1.0], (5, 1)),
                         np.array([[1.0, 5.0]])])
        r = ThresholdRouter(m=1, window=1, soft=False)
        mu = r.route(nis, np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        self.assertEqual(mu.argmax(), 0)   # latest step: model 0 NIS=1 closest to m=1

    def test_returns_prob_vector(self):
        r   = ThresholdRouter()
        nis = _dummy_nis()
        c   = np.array([0.5, 0.5])
        L   = np.array([1.0, 1.0])
        self.assertTrue(_is_prob_vector(r.route(nis, c, L), 2))


# ── HybridRouter ──────────────────────────────────────────────────────────────

class TestHybridRouter(unittest.TestCase):

    def _call(self, alpha, c, L):
        nis = np.ones((3, 2))
        r_a = BayesianRouter()
        r_b = ThresholdRouter(m=1, soft=True)
        hr  = HybridRouter(r_a, r_b, alpha=alpha)
        return hr.route(nis, c, L)

    def test_alpha_zero_is_pure_a(self):
        c = np.array([0.7, 0.3])
        L = np.array([0.8, 0.2])
        mu_hybrid = self._call(0.0, c, L)
        mu_bay    = BayesianRouter().route(np.ones((3, 2)), c, L)
        np.testing.assert_allclose(mu_hybrid, mu_bay, atol=1e-10)

    def test_alpha_one_is_pure_b(self):
        nis = np.ones((3, 2))
        c   = np.array([0.5, 0.5])
        L   = np.array([1.0, 1.0])
        r_b = ThresholdRouter(m=1, soft=True)
        mu_b = r_b.route(nis, c, L)
        hr   = HybridRouter(BayesianRouter(), r_b, alpha=1.0)
        mu_h = hr.route(nis, c, L)
        np.testing.assert_allclose(mu_h, mu_b, atol=1e-10)

    def test_blend_is_between(self):
        c = np.array([0.5, 0.5])
        L = np.array([5.0, 1.0])
        mu_a = BayesianRouter().route(np.ones((3, 2)), c, L)
        mu   = self._call(0.5, c, L)
        # blended weight for model 0 should be between pure_b[0] and mu_a[0]
        # (both agree model 0 is better, so mu[0] > 0.5 at minimum)
        self.assertGreater(mu[0], 0.5)
        self.assertTrue(_is_prob_vector(mu, 2))

    def test_fit_delegates_to_children(self):
        mlp_a = MLP(hidden=4)
        mlp_b = MLP(hidden=4)
        r_a = NeuralRouter(mlp_a, window=2)
        r_b = NeuralRouter(mlp_b, window=2)
        hr  = HybridRouter(r_a, r_b, alpha=0.5)

        nis_arr = [np.random.default_rng(0).uniform(0, 2, (10, 2))]
        lbl_arr = [np.array([0]*5 + [1]*5)]
        hr.fit(nis_arr, lbl_arr, epochs=5)
        # both children should now be trained
        self.assertTrue(r_a._is_trained)
        self.assertTrue(r_b._is_trained)


# ── NeuralRouter ──────────────────────────────────────────────────────────────

class TestNeuralRouter(unittest.TestCase):

    def _make_data(self, seed=0):
        rng = np.random.default_rng(seed)
        # model 0: low NIS ≈ 1, model 1: high NIS ≈ 5
        nis_0 = rng.normal(1.0, 0.3, (20, 2))
        nis_0[:, 1] = rng.normal(5.0, 0.3, 20)
        nis_1 = rng.normal(5.0, 0.3, (20, 2))
        nis_1[:, 0] = rng.normal(1.0, 0.3, 20)
        lbl_0 = np.zeros(20, int)
        lbl_1 = np.ones(20, int)
        return [nis_0, nis_1], [lbl_0, lbl_1]

    def test_untrained_falls_back_to_bayesian(self):
        r   = NeuralRouter(MLP(hidden=4), window=3)
        nis = _dummy_nis()
        c   = np.array([0.6, 0.4])
        L   = np.array([2.0, 1.0])
        mu  = r.route(nis, c, L)
        mu_bay = BayesianRouter().route(nis, c, L)
        np.testing.assert_allclose(mu, mu_bay, atol=1e-10)

    def test_window_feat_shape(self):
        r   = NeuralRouter(MLP(hidden=4), window=4)
        nis = _dummy_nis(T=10, M=3)
        feat = r._window_feat(nis)
        self.assertEqual(feat.shape, (12,))  # 3 × 4

    def test_window_feat_padded_at_start(self):
        r   = NeuralRouter(MLP(hidden=4), window=5)
        nis = _dummy_nis(T=2, M=2)
        feat = r._window_feat(nis)
        self.assertEqual(feat.shape, (10,))
        # first 3 rows (padding) should be 1.0
        np.testing.assert_allclose(feat[:6], 1.0)

    def test_fit_sets_trained(self):
        r = NeuralRouter(MLP(hidden=4), window=2)
        nis_arrs, lbl_arrs = self._make_data()
        r.fit(nis_arrs, lbl_arrs, epochs=50)
        self.assertTrue(r._is_trained)

    def test_trained_returns_prob_vector(self):
        r = NeuralRouter(MLP(hidden=4), window=2)
        nis_arrs, lbl_arrs = self._make_data()
        r.fit(nis_arrs, lbl_arrs, epochs=50)
        nis = _dummy_nis()
        mu  = r.route(nis, np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        self.assertTrue(_is_prob_vector(mu, 2))

    def test_any_callable_model_works(self):
        # A lambda that always prefers model 0
        model = lambda x: np.array([0.9, 0.1])
        r = NeuralRouter(model=model, window=3)
        # Manually set normalisation params to skip build/fit
        r._x_mean = np.zeros(6)
        r._x_std  = np.ones(6)
        mu = r.route(_dummy_nis(), np.array([0.5, 0.5]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(mu, [0.9, 0.1], atol=1e-10)


# ── MLP ───────────────────────────────────────────────────────────────────────

class TestMLP(unittest.TestCase):

    def test_build_creates_correct_shapes(self):
        mlp = MLP(hidden=8)
        mlp.build(d_in=10, d_out=3, seed=0)
        self.assertEqual(mlp._weights["W1"].shape, (8, 10))
        self.assertEqual(mlp._weights["W2"].shape, (8, 8))
        self.assertEqual(mlp._weights["W3"].shape, (3, 8))
        self.assertEqual(mlp._weights["b3"].shape, (3,))

    def test_call_returns_prob_vector(self):
        mlp = MLP(hidden=8)
        mlp.build(10, 3, seed=1)
        x   = np.random.default_rng(1).normal(size=10)
        out = mlp(x)
        self.assertEqual(out.shape, (3,))
        self.assertAlmostEqual(out.sum(), 1.0, places=10)
        self.assertTrue(np.all(out >= 0))

    def test_call_raises_before_build(self):
        mlp = MLP()
        with self.assertRaises(RuntimeError):
            mlp(np.zeros(10))

    def test_fit_raises_before_build(self):
        mlp = MLP()
        with self.assertRaises(RuntimeError):
            mlp.fit(np.zeros((5, 10)), np.zeros(5, int))

    def test_fit_reduces_loss(self):
        rng = np.random.default_rng(42)
        N, d_in, M = 80, 10, 2
        X = rng.normal(size=(N, d_in))
        y = (X[:, 0] > 0).astype(int)      # linearly separable
        mlp = MLP(hidden=16)
        mlp.build(d_in, M, seed=0)

        def _cross_entropy():
            preds = np.array([mlp(x) for x in X])
            return -np.log(preds[np.arange(N), y] + 1e-12).mean()

        loss_before = _cross_entropy()
        mlp.fit(X, y, epochs=300, lr=5e-3)
        loss_after  = _cross_entropy()
        self.assertLess(loss_after, loss_before)

    def test_build_is_reproducible_with_seed(self):
        m1 = MLP(16); m1.build(6, 2, seed=7)
        m2 = MLP(16); m2.build(6, 2, seed=7)
        np.testing.assert_array_equal(m1._weights["W1"], m2._weights["W1"])


# ── nn stubs ──────────────────────────────────────────────────────────────────

class TestNNStubs(unittest.TestCase):

    def _check_stub(self, cls):
        obj = cls()
        with self.assertRaises(NotImplementedError):
            obj.build(10, 2)
        with self.assertRaises(NotImplementedError):
            obj.fit(np.zeros((5, 10)), np.zeros(5, int))
        with self.assertRaises(NotImplementedError):
            obj(np.zeros(10))

    def test_transformer_nn_raises(self):
        self._check_stub(TransformerNN)

    def test_lstm_raises(self):
        self._check_stub(LSTM)

    def test_cnn_raises(self):
        self._check_stub(CNN)


if __name__ == "__main__":
    unittest.main()

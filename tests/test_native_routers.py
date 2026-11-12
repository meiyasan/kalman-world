"""The rule-based IKF routers decide in C++ (Kalman::Routers); the Python arithmetic is the reference.

Bayesian, Threshold and Hybrid routers are compared with their Python reference
on random inputs and inside a whole IKF pass, then the error paths: a Python
child of a Hybrid called back from C++ and raising, configurations the C++
classes do not take, and the C API refusing bad arguments.

Skipped when libKalman is not built.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.routers import BayesianRouter, ThresholdRouter, HybridRouter, KalmanRouter


def inputs(rows, M, seed=0):
    rng = np.random.default_rng(seed)
    nis = rng.chisquare(2, size=(rows, M)) * rng.uniform(0.5, 3.0, size=M)
    c = rng.dirichlet(np.ones(M))
    likelihoods = rng.uniform(0.0, 2.0, size=M)
    likelihoods[rng.integers(M)] = 0.0                          # one mode written off
    return nis, c, likelihoods


def reference_route(router, nis, c, likelihoods):
    with native.python_reference():
        return router(nis, c, likelihoods)


class _Softmax(KalmanRouter):
    """A router written in Python only (stands in for a NeuralRouter)."""

    def route(self, nis_arr, c, likelihoods):
        z = -nis_arr[-1]
        return np.exp(z - z.max()) / np.exp(z - z.max()).sum()


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestRoutersAgreeWithTheirReference(unittest.TestCase):

    def agree(self, router, atol=1e-15, shapes=((1, 2), (3, 2), (12, 3), (40, 5))):
        for seed, (rows, M) in enumerate(shapes):
            nis, c, L = inputs(rows, M, seed)
            mu = router(nis, c, L)
            ref = reference_route(router, nis, c, L)
            np.testing.assert_allclose(mu, ref, rtol=0, atol=atol, err_msg=f"{rows} x {M}")
            self.assertAlmostEqual(mu.sum(), 1.0, places=12)
        self.assertIsNotNone(router.__dict__.get("_native_state"), "the C++ router ran")

    def test_bayesian(self):
        self.agree(BayesianRouter())

    def test_threshold_hard_and_soft(self):
        for window in (1, 5, 20, 0):            # 0 = the whole history, as nis[-0:] is
            self.agree(ThresholdRouter(m=2, window=window, soft=False))
            self.agree(ThresholdRouter(m=2, window=window, soft=True))

    def test_threshold_on_a_history_shorter_than_its_window(self):
        router = ThresholdRouter(m=1, window=10, soft=True)
        nis = np.array([[0.9, 4.0], [1.2, 6.0]])
        np.testing.assert_allclose(router(nis, np.ones(2) / 2, np.ones(2)),
                                   reference_route(router, nis, np.ones(2) / 2, np.ones(2)), rtol=0, atol=1e-15)

    def test_threshold_before_anything_is_observed(self):
        router = ThresholdRouter(m=1, window=5)
        for path in (router, lambda *a: reference_route(router, *a)):
            np.testing.assert_allclose(path(np.empty((0, 3)), np.ones(3) / 3, np.ones(3)), np.ones(3) / 3)

    def test_hybrid_of_two_cxx_routers(self):
        for alpha in (0.0, 0.3, 1.0):
            self.agree(HybridRouter(BayesianRouter(), ThresholdRouter(m=2, window=4, soft=True), alpha=alpha),
                       atol=1e-12)

    def test_nested_hybrid(self):
        inner = HybridRouter(BayesianRouter(), ThresholdRouter(m=2, window=3, soft=True), alpha=0.5)
        self.agree(HybridRouter(inner, ThresholdRouter(m=2, window=6), alpha=0.25), atol=1e-12)

    def test_hybrid_with_a_python_child_calls_it_back(self):
        self.agree(HybridRouter(BayesianRouter(), _Softmax(), alpha=0.4), atol=1e-12)
        self.agree(HybridRouter(lambda nis, c, L: c, BayesianRouter(), alpha=0.6), atol=1e-12)

    def test_a_subclass_that_overrides_route_is_obeyed(self):
        class Flat(BayesianRouter):
            def route(self, nis_arr, c, likelihoods):
                return np.ones(len(c)) / len(c)
        nis, c, L = inputs(4, 3)
        np.testing.assert_allclose(Flat()(nis, c, L), np.ones(3) / 3)
        hybrid = HybridRouter(Flat(), BayesianRouter(), alpha=0.0)
        np.testing.assert_allclose(hybrid(nis, c, L), np.ones(3) / 3, atol=1e-12)

    def test_a_changed_parameter_rebuilds(self):
        router = ThresholdRouter(m=1, window=5)
        nis = np.array([[1.0, 3.0]] * 5)
        self.assertEqual(int(np.argmax(router(nis, np.ones(2) / 2, np.ones(2)))), 0)
        router.m = 3
        self.assertEqual(int(np.argmax(router(nis, np.ones(2) / 2, np.ones(2)))), 1)


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestInsideTheIKF(unittest.TestCase):
    """A whole IKF pass with the C++ router, against the same pass with the reference router."""

    def test_mode_probabilities_agree(self):
        from modules.framework.filters.ikf import InteractiveKalmanFilter
        from tests.test_ikf import _two_models, _sim, PI, REGIME_200_TEST
        zk, _ = _sim(REGIME_200_TEST, seed=3)
        for make in (BayesianRouter,
                     lambda: ThresholdRouter(m=1, window=10, soft=True),
                     lambda: HybridRouter(BayesianRouter(), ThresholdRouter(m=1, window=5, soft=True), alpha=0.3)):
            router = make()
            cxx = InteractiveKalmanFilter(list(_two_models()), PI, router=router)
            cxx.forward(zk, record=False)
            ref_router = make()
            ref = InteractiveKalmanFilter(list(_two_models()), PI,
                                          router=lambda n, c, L, r=ref_router: r._reference_route(n, c, L))
            ref.forward(zk, record=False)
            self.assertIsNotNone(router.__dict__.get("_native_state"))
            np.testing.assert_allclose(cxx.mode_probabilities, ref.mode_probabilities, rtol=0, atol=1e-12)
            self.assertGreater(np.ptp(cxx.mode_probabilities[:, 1]), 0.5)      # it did switch


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestErrors(unittest.TestCase):

    def test_a_python_child_error_comes_back_as_itself(self):
        def broken(nis, c, L):
            raise ArithmeticError("the child gave up")
        router = HybridRouter(BayesianRouter(), broken, alpha=0.5)
        with self.assertRaisesRegex(ArithmeticError, "the child gave up"):
            router(*inputs(3, 2))

    def test_a_child_returning_the_wrong_size(self):
        router = HybridRouter(BayesianRouter(), lambda nis, c, L: np.ones(5), alpha=0.5)
        with self.assertRaisesRegex(ValueError, "5 probabilities for 2 models"):
            router(*inputs(3, 2))

    def test_what_cxx_does_not_take_runs_the_reference(self):
        nis, c, L = inputs(3, 2)
        for router in (HybridRouter(BayesianRouter(), BayesianRouter(), alpha=1.5),   # outside [0, 1]
                       ThresholdRouter(m=1, window=-2)):
            router(nis, c, L)
            self.assertIsNone(router.__dict__.get("_native_state"))
        bayes = BayesianRouter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                     # the reference casts it
            bayes(nis, c.astype(complex), L)                    # not real
        self.assertIsNone(bayes.__dict__.get("_native_state"))

    def test_c_api_refuses_bad_arguments(self):
        with self.assertRaisesRegex(RuntimeError, "M must be positive"):
            core.check(core.lib.kalman_router_route(core.create_router("bayesian"), 0, 0, core.ffi.NULL,
                                                    core.ffi.NULL, core.ffi.NULL, core.ffi.NULL), "route")
        with self.assertRaisesRegex(RuntimeError, "alpha must lie in"):
            a = core.create_router("bayesian")
            core.create_router("hybrid", a=a, b=a, alpha=2.0)
        with self.assertRaisesRegex(RuntimeError, "null kalman_router_t"):
            core.create_router("hybrid", a=core.ffi.NULL, b=core.create_router("bayesian"), alpha=0.5)
        with self.assertRaisesRegex(RuntimeError, "must not be negative"):
            core.create_router("threshold", m=1, window=-1, soft=False)
        c = core.ffi.new("double[]", [0.5, 0.5])
        h = core.create_router("bayesian")
        with self.assertRaisesRegex(RuntimeError, "null output buffer for mu"):
            core.check(core.lib.kalman_router_route(h, 0, 2, core.ffi.NULL, c, c, core.ffi.NULL), "route")
        with self.assertRaisesRegex(RuntimeError, "null input for nis"):
            core.check(core.lib.kalman_router_route(h, 3, 2, core.ffi.NULL, c, c, core.ffi.new("double[2]")), "route")
        with self.assertRaisesRegex(KeyError, "no C\\+\\+ router"):
            core.create_router("oracle")


if __name__ == "__main__":
    unittest.main()

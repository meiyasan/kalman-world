"""The controllers forward to Kalman::Controllers (C++); the Python arithmetic is the reference.

Every shipped controller (P, PID, LQR, MPPI, Boost, AdaptiveQR) is run on its C++
path and on its Python reference (inside ``native.python_reference()``, or with
the C++ controller switched off on the instance) and the two are compared: the
commands, the logged costs, the adaptation factors, and the model they retune.
The notebook 12 cells that use them are replayed in miniature. Then the C API's
error paths, and the Riccati solver against scipy.

Skipped when libKalman is not built.
"""
import copy
import pickle
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import core
import modules.framework.models.native as native
from modules.framework.models.extended import ExtendedKalmanModel
from modules.framework.controllers import (ProportionalKalmanController, PIDKalmanController,
                                           LQRKalmanController, MPPIKalmanController,
                                           BoostKalmanController, AdaptiveQRKalmanController)
from modules.apps.vehicles import CarLaneModel

LATERAL_NOISE_STD = 0.3


def lane(x0=(1.0, 0.0, 15.0), **kw):
    return CarLaneModel("camera", fs=50, x0=np.array(x0, dtype=float).reshape(-1, 1),
                        lateral_uncertainty=LATERAL_NOISE_STD, **kw)


def cost_of(model):
    Qcost, Rcost, x_ref = model.control_cost()
    return dict(Qcost=Qcost, Rcost=Rcost, x_ref=x_ref)


def reference(controller):
    """The same controller on its Python reference path, everything else unchanged."""
    controller._native_handle = lambda: None
    return controller


def beliefs(n_steps, seed=0):
    rng = np.random.default_rng(seed)
    for k in range(n_steps):
        x = np.array([[0.8], [0.05], [15.0]]) + 0.1 * rng.standard_normal((3, 1))
        A = 0.05 * rng.standard_normal((3, 3))
        yield x, 0.01 * np.eye(3) + A @ A.T, 0.02 * k, 0.3 * rng.standard_normal((1, 1))


def drive(controller, n_steps=20, seed=0):
    return np.array([controller(x, P, t, nu=nu).ravel() for x, P, t, nu in beliefs(n_steps, seed)])


def closed_loop_run(controller, N, seed=0):
    """Notebook 12's closed loop (cell 5), Gaussian noise: the true car and an EKF steered by
    ``controller``. Returns the true states."""
    from modules.framework.filters import ExtendedKalmanFilter
    rng = np.random.default_rng(seed)
    model = controller.model
    ekf = ExtendedKalmanFilter(model)
    ekf.reset()
    x_true, nu, t_now = model.x0.copy(), np.zeros((model.m, 1)), 0.0
    q_std, r_std = np.sqrt(np.diag(model.Q())), np.sqrt(np.diag(model.R()))
    xs = np.zeros((N, model.n))
    for k in range(N):
        u = controller(ekf.x, ekf.P, t_now, nu=nu)
        x_true = np.asarray(model.f(x_true, u=u, t=t_now), dtype=float).reshape(-1, 1)
        x_true = x_true + rng.normal(0, q_std, model.n).reshape(-1, 1)
        z = np.asarray(model.h(x_true, t=t_now), dtype=float).reshape(-1, 1) + rng.normal(0, r_std).reshape(-1, 1)
        ekf.x, ekf.P, _ = ekf.engine._predict(model, ekf.x, ekf.P, u=u, t=t_now)
        nu = z - np.asarray(model.h(ekf.x, t=t_now), dtype=float).reshape(-1, 1)
        ekf.x, ekf.P = ekf.engine._update(model, ekf.x, ekf.P, z, t=t_now)
        xs[k] = x_true.ravel()
        t_now += 1.0 / model.fs
    return xs


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestEveryControllerAgreesWithItsReference(unittest.TestCase):

    def agree(self, make, n_steps=20, atol=1e-12, fit=False):
        native_ctrl, ref_ctrl = make(), reference(make())
        if fit:
            native_ctrl.fit()
            ref_ctrl.fit()
        u_cxx, u_ref = drive(native_ctrl, n_steps), drive(ref_ctrl, n_steps)
        self.assertIsNotNone(native_ctrl.__dict__.get("_native_state"), "the C++ controller ran")
        self.assertIsNone(ref_ctrl.__dict__.get("_native_state"))
        scale = max(1.0, np.abs(u_ref).max())
        self.assertLess(np.abs(u_cxx - u_ref).max() / scale, atol)
        np.testing.assert_allclose([c for _, c in native_ctrl.cost_log], [c for _, c in ref_ctrl.cost_log],
                                   rtol=1e-12, atol=1e-14)
        return native_ctrl, ref_ctrl, u_cxx

    def test_proportional(self):
        m = lane()
        self.agree(lambda: ProportionalKalmanController(m, Kp=np.array([[0.1, 0.7, 0.0]]), **cost_of(m)))

    def test_pid_keeps_its_integral_in_cxx(self):
        m = lane()
        make = lambda: PIDKalmanController(m, Kp=np.array([[0.1, 0.7, 0.0]]), Ki=np.array([[0.02, 0.0, 0.0]]),
                                           Kd=np.array([[0.0, 0.05, 0.0]]), **cost_of(m))
        cxx, ref, u = self.agree(make)
        self.assertGreater(np.abs(np.diff(u[:, 0])).max(), 0.0)
        # reset() clears the C++ integral too: the same run again gives the same commands.
        cxx.reset()
        ref.reset()
        np.testing.assert_allclose(drive(cxx), drive(ref), rtol=0, atol=1e-12)

    def test_pid_with_missing_gains_is_proportional(self):
        m = lane()
        self.agree(lambda: PIDKalmanController(m, Kp=np.array([[0.1, 0.7, 0.0]])))

    def test_lqr_gain_and_policy(self):
        m = lane()
        cxx, ref, _ = self.agree(lambda: LQRKalmanController(m, **cost_of(m)), fit=True, atol=1e-12)
        self.assertLess(np.abs(cxx.K - ref.K).max() / np.abs(ref.K).max(), 1e-12)

    def test_lqr_follows_a_gain_set_by_hand(self):
        m = lane()
        cxx = LQRKalmanController(m, **cost_of(m)).fit()
        cxx.K = np.array([[0.2, 0.5, 0.0]])
        x, P, t, nu = next(beliefs(1))
        np.testing.assert_allclose(cxx._policy(x, P, t, nu=nu), -cxx.K @ (x - cxx.x_ref), rtol=0, atol=1e-15)

    def test_lqr_before_fit_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Call fit"):
            LQRKalmanController(lane())(np.zeros((3, 1)), np.eye(3), 0.0)

    def test_mppi_makes_the_same_decisions(self):
        m = lane(x0=(3.5, 0.0, 15.0))
        cost = dict(Qcost=np.diag([1.0, 0.0, 0.0]), Rcost=np.array([[10.0]]), x_ref=m.control_cost()[2])
        make = lambda: MPPIKalmanController(m, n_rollouts=40, horizon=15, noise_sigma=0.05, n_iters=2, seed=7, **cost)
        cxx, ref, u = self.agree(make, n_steps=6, atol=1e-11)
        self.assertGreater(np.abs(u).max(), 1e-3)
        # The warm start lives in C++; reset() restarts it and the generator on both paths.
        cxx.reset()
        ref.reset()
        np.testing.assert_allclose(drive(cxx, 3), drive(ref, 3), rtol=0, atol=1e-11)

    def test_mppi_on_a_model_written_in_cxx_never_leaves_cxx(self):
        try:
            core.load_plugin(core.resolve_plugin("KalmanPlugin_example"))
        except FileNotFoundError:
            self.skipTest("the example plugin is not built (make examples)")
        from modules.framework.models.native import NativeModel
        m = NativeModel("pendulum")
        x = np.asarray(m.x0, dtype=float).reshape(-1, 1) + 0.1
        cxx = MPPIKalmanController(m, n_rollouts=20, horizon=10, seed=7)
        ref = reference(MPPIKalmanController(m, n_rollouts=20, horizon=10, seed=7))
        u_cxx = [cxx._policy(x, np.eye(2), 0.0) for _ in range(3)]
        u_ref = [ref._policy(x, np.eye(2), 0.0) for _ in range(3)]
        self.assertIs(cxx._native_state[2], m._handle)            # the model's own handle, no callback
        np.testing.assert_allclose(u_cxx, u_ref, rtol=0, atol=1e-13)

    def test_mppi_planner_call_without_a_covariance(self):
        """NeuralKalmanController.fit_mppi calls _policy(x, None, t)."""
        m = lane()
        cxx = MPPIKalmanController(m, n_rollouts=10, horizon=5, seed=1)
        ref = reference(MPPIKalmanController(m, n_rollouts=10, horizon=5, seed=1))
        x = np.array([[0.5], [0.0], [15.0]])
        np.testing.assert_allclose(cxx._policy(x, None, 0.0), ref._policy(x, None, 0.0), rtol=0, atol=1e-12)

    def test_boost_retunes_the_python_model(self):
        m_cxx, m_ref = lane(), lane()
        cxx = BoostKalmanController(m_cxx, boost_factor=5.0, threshold=0.5, window=3)
        ref = reference(BoostKalmanController(m_ref, boost_factor=5.0, threshold=0.5, window=3))
        drive(cxx, 30)
        drive(ref, 30)
        self.assertEqual([t for t, _ in cxx.scale_log], [t for t, _ in ref.scale_log])
        np.testing.assert_allclose([s for _, s in cxx.scale_log], [s for _, s in ref.scale_log], rtol=0, atol=0)
        self.assertIn(0.2, [s for _, s in cxx.scale_log])
        np.testing.assert_allclose(m_cxx._R, m_ref._R, rtol=0, atol=0)
        cxx.reset()
        np.testing.assert_allclose(m_cxx._R, lane()._R, rtol=0, atol=0)

    def test_adaptive_qr_retunes_the_python_model(self):
        m_cxx, m_ref = lane(), lane()
        cxx = AdaptiveQRKalmanController(m_cxx, window=5, q_min=0.5, q_max=10.0)
        ref = reference(AdaptiveQRKalmanController(m_ref, window=5, q_min=0.5, q_max=10.0))
        drive(cxx, 30)
        drive(ref, 30)
        self.assertEqual(len(cxx.scale_log), 26)
        np.testing.assert_allclose([s for _, s in cxx.scale_log], [s for _, s in ref.scale_log], rtol=1e-13)
        np.testing.assert_allclose(m_cxx._Q, m_ref._Q, rtol=1e-13)
        cxx.reset()
        np.testing.assert_allclose(m_cxx._Q, lane()._Q, rtol=0, atol=0)

    def test_inside_python_reference_no_cxx_controller_is_built(self):
        m = lane()
        c = ProportionalKalmanController(m, Kp=np.array([[0.1, 0.7, 0.0]]))
        with native.python_reference():
            drive(c, 3)
        self.assertIsNone(c.__dict__.get("_native_state"))

    def test_a_gain_of_another_shape_runs_the_reference(self):
        m = lane()
        c = ProportionalKalmanController(m, Kp=np.array([0.1, 0.7, 0.0]))     # flat: numpy broadcasts it
        u = drive(c, 3)
        self.assertIsNone(c.__dict__.get("_native_state"))
        self.assertEqual(u.shape, (3, 1))

    def test_a_changed_gain_reaches_cxx(self):
        m = lane()
        c = ProportionalKalmanController(m, Kp=np.array([[0.1, 0.7, 0.0]]))
        drive(c, 2)
        c.Kp = np.array([[1.0, 0.0, 0.0]])
        x, P, t, nu = next(beliefs(1))
        np.testing.assert_allclose(c._policy(x, P, t, nu=nu), -(x[0] - c.x_ref[0]).reshape(1, 1), atol=1e-15)

    def test_copies_and_pickles_build_their_own(self):
        m = lane()
        c = PIDKalmanController(m, Kp=np.array([[0.1, 0.7, 0.0]]), Ki=np.array([[0.02, 0.0, 0.0]]))
        drive(c, 2)
        for twin in (copy.deepcopy(c), pickle.loads(pickle.dumps(c))):
            self.assertIsNone(twin.__dict__.get("_native_state"))
            twin.reset()
            c.reset()
            np.testing.assert_allclose(drive(twin, 4), drive(c, 4), atol=1e-14)
            self.assertIsNotNone(twin.__dict__.get("_native_state"))


class _Unstable(ExtendedKalmanModel):
    """A driven model whose f raises after a few calls, and whose clip can raise."""

    def __init__(self, fail_f=False, fail_clip=False):
        self.n, self.m, self.l = 2, 1, 1
        self.dtype = np.float64
        self.calls, self.fail_f, self.fail_clip = 0, fail_f, fail_clip
        super().__init__("unstable", Q=np.eye(2) * 1e-4, R=np.eye(1) * 1e-2, L=np.eye(2), fs=10.0,
                         x0=np.zeros((2, 1)), P0=np.eye(2))

    def f(self, x, u=None, w=None, t=None):
        self.calls += 1
        if self.fail_f and self.calls > 3:
            raise ZeroDivisionError("f gave up")
        x = np.asarray(x, float).reshape(-1)
        torque = 0.0 if u is None else float(np.ravel(u)[0])
        return np.array([[x[0] + 0.1 * x[1]], [x[1] + 0.1 * torque]])

    def h(self, x, v=None, t=None):
        return np.asarray(x, float).reshape(-1)[:1].reshape(1, 1)

    def clip_u(self, u):
        if self.fail_clip:
            raise KeyError("no actuator")
        return np.clip(u, -0.5, 0.5)


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestErrors(unittest.TestCase):

    def test_an_exception_in_f_comes_back_as_itself(self):
        c = MPPIKalmanController(_Unstable(fail_f=True), n_rollouts=4, horizon=3, seed=0)
        with self.assertRaisesRegex(ZeroDivisionError, "f gave up"):
            c(np.ones((2, 1)), np.eye(2), 0.0)

    def test_an_exception_in_clip_comes_back_as_itself(self):
        c = MPPIKalmanController(_Unstable(fail_clip=True), n_rollouts=4, horizon=3, seed=0)
        with self.assertRaisesRegex(KeyError, "no actuator"):
            c._policy(np.ones((2, 1)), np.eye(2), 0.0)

    def test_mppi_respects_the_actuator_limit_from_python(self):
        c = MPPIKalmanController(_Unstable(), n_rollouts=50, horizon=10, noise_sigma=2.0, seed=3)
        u = c._policy(np.array([[5.0], [0.0]]), np.eye(2), 0.0)
        self.assertLessEqual(abs(float(u[0, 0])), 0.5 + 1e-12)
        self.assertIsNotNone(c.__dict__.get("_native_state"))

    def test_c_api_refuses_what_does_not_fit(self):
        m = _Unstable()
        handle, keep = native.callback_handle(m)
        p = core.create_controller("proportional", handle, 2, 1, Kp=np.ones((1, 2)))
        with self.assertRaisesRegex(RuntimeError, "not an LQR"):
            core.lqr_fit(p, np.eye(2), np.ones((2, 1)), 2, 1)
        mppi = core.create_controller("mppi", handle, 2, 1, n_rollouts=2, horizon=3, noise_sigma=0.1, lam=1.0,
                                      n_iters=1)
        with self.assertRaisesRegex(RuntimeError, "6 expected"):
            core.mppi_set_draws(mppi, np.zeros(5))
        with self.assertRaisesRegex(RuntimeError, "lambda must be positive"):
            core.create_controller("mppi", handle, 2, 1, n_rollouts=2, horizon=3, noise_sigma=0.1, lam=0.0,
                                   n_iters=1)
        with self.assertRaisesRegex(RuntimeError, "window must be positive"):
            core.create_controller("boost", handle, 2, 1, boost_factor=2.0, threshold=1.0, window=0)
        with self.assertRaisesRegex(RuntimeError, "null kalman_model_t"):
            core.create_controller("lqr", core.ffi.NULL, 2, 1)
        lqr = core.create_controller("lqr", handle, 2, 1)
        with self.assertRaisesRegex(RuntimeError, "call Fit"):
            core.controller_policy(lqr, np.zeros(2), None, 0.0, None, 2, 1, 1)
        with self.assertRaisesRegex(KeyError, "no C\\+\\+ controller"):
            core.create_controller("bang-bang", handle, 2, 1)

    def test_a_gain_of_the_wrong_shape_is_refused_by_cxx(self):
        from modules.apps.vehicles import CarLaneModel  # noqa: F401  (l = 1, n = 3)
        m = lane()
        handle, keep = native.callback_handle(m)
        bad = core.ffi.new("double[]", [1.0, 2.0, 3.0])
        h = core.lib.kalman_controller_create_pid(handle, bad, core.ffi.NULL, core.ffi.NULL,
                                                  core.ffi.NULL, core.ffi.NULL, core.ffi.NULL)
        self.assertNotEqual(h, core.ffi.NULL)                    # (1, 3) is the right shape
        core.lib.kalman_controller_destroy(h)
        h = core.lib.kalman_controller_create_proportional(core.ffi.NULL, bad, core.ffi.NULL, core.ffi.NULL,
                                                           core.ffi.NULL)
        self.assertEqual(h, core.ffi.NULL)
        self.assertIn("null kalman_model_t", core.last_error())

    def test_dare_of_an_unstabilisable_pair_is_refused(self):
        A = np.array([[1.2, 0.0], [0.0, 0.5]])
        B = np.array([[0.0], [1.0]])                             # the unstable mode is not reached
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            core.solve_dare(A, B, np.eye(2), np.eye(1))

    def test_command_logs_the_cost_on_the_clipped_command(self):
        m = _Unstable()
        handle, keep = native.callback_handle(m)
        c = ProportionalKalmanController(m, Kp=np.array([[4.0, 0.0]]))
        c._native_handle()                                        # builds the clip callback too
        h = c._native_state[1]
        u, cost = core.controller_command(h, np.array([1.0, 0.0]), np.eye(2), 0.0, None, 2, 1, 1)
        self.assertEqual(float(u[0, 0]), -4.0)                    # the raw ask
        self.assertAlmostEqual(cost, 1.0 + 0.25, places=14)       # e^T e + clip(u)^2


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestRiccati(unittest.TestCase):
    """Kalman::Algebra::SolveDARE against scipy.linalg.solve_discrete_are."""

    def cases(self):
        rng = np.random.default_rng(3)
        out = [(np.array([[1, 0.02], [0, 1]]), np.array([[2e-4], [0.02]]), np.eye(2), np.eye(1)),
               (np.array([[1, 0.1], [0, 1]]), np.array([[0.005], [0.1]]), np.diag([1.0, 0.0]), np.array([[100.0]])),
               (np.array([[0.999, 0.01], [0, 0.999]]), np.array([[0.0], [0.01]]), np.eye(2), np.array([[1e3]]))]
        m = lane()
        out.append((m.F(), m.B(), np.diag([1.0, 0.0, 0.0]), np.array([[100.0]])))
        for n, l in ((4, 2), (6, 1), (8, 3)):
            A = rng.normal(size=(n, n))
            A /= max(abs(np.linalg.eigvals(A))) / 1.05
            out.append((A, rng.normal(size=(n, l)), np.eye(n), np.eye(l)))
        return out

    def test_agrees_with_scipy(self):
        from scipy.linalg import solve_discrete_are
        for A, B, Q, R in self.cases():
            P_ref = solve_discrete_are(A, B, Q, R)
            K_ref = np.linalg.solve(R + B.T @ P_ref @ B, B.T @ P_ref @ A)
            P, K = core.solve_dare(A, B, Q, R)
            # Measured: 1.4e-11 for the pole at 0.999 (where scipy's own residual is the
            # larger one, 2.4e-14 against 1.9e-16), 4e-14 or better elsewhere.
            self.assertLess(np.abs(K - K_ref).max() / np.abs(K_ref).max(), 1e-10)
            self.assertLess(np.abs(P - P_ref).max() / np.abs(P_ref).max(), 1e-10)
            G = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            residual = np.abs(A.T @ P @ A - A.T @ P @ B @ G + Q - P).max() / np.abs(P).max()
            self.assertLess(residual, 1e-13)


@unittest.skipUnless(core.HAS_CXX_CORE, "libKalman not built")
class TestNotebook12(unittest.TestCase):
    """The notebook's controller cells, shortened, on both paths."""

    def test_lqr_and_mppi_in_closed_loop(self):
        cost = dict(Qcost=np.diag([1.0, 0.0, 0.0]), Rcost=np.array([[10.0]]), x_ref=lane().control_cost()[2])
        mppi_kw = dict(n_rollouts=30, horizon=20, noise_sigma=0.05, lam=1.0, n_iters=2, seed=7)
        for make in (lambda m: LQRKalmanController(m, **cost).fit(),
                     lambda m: MPPIKalmanController(m, **mppi_kw, **cost)):
            runs = []
            for ref in (False, True):
                c = make(lane(x0=(3.5, 0.0, 15.0)))
                if ref:
                    c = reference(c)
                    if isinstance(c, LQRKalmanController):
                        c._reference_fit()
                xs = closed_loop_run(c, 25, seed=2)
                runs.append((xs, sum(v for _, v in c.cost_log), np.array([u.ravel() for _, u in c.u_log])))
            (x1, c1, u1), (x2, c2, u2) = runs
            np.testing.assert_allclose(u1, u2, rtol=0, atol=1e-10)
            np.testing.assert_allclose(x1, x2, rtol=0, atol=1e-10)
            self.assertAlmostEqual(c1, c2, delta=1e-9 * abs(c2))

    def _filter(self, make_ctrl, zk, controller_reference, everything_reference=False):
        from modules.framework.filters import ExtendedKalmanFilter
        model = lane(x0=(0.0, 0.0, 15.0))
        ctrl = make_ctrl(model)
        if controller_reference:
            reference(ctrl)
        if everything_reference:
            with native.python_reference():
                rec = ExtendedKalmanFilter(model).forward(zk, uk=ctrl)
        else:
            rec = ExtendedKalmanFilter(model).forward(zk, uk=ctrl)
        return np.asarray(rec.xk), [s for _, s in ctrl.scale_log]

    def test_boost_on_a_lane_change(self):
        rng = np.random.default_rng(29)
        t = np.arange(300) / 50
        truth = np.where(t < 2, 0.0, np.where(t > 4, 3.5, 3.5 * (1 - np.cos(np.pi * (t - 2) / 2)) / 2))
        zk = (truth + rng.normal(0, LATERAL_NOISE_STD, 300)).reshape(-1, 1)
        make = lambda m: BoostKalmanController(m, boost_factor=5.0, threshold=2.0, window=5)
        x_cxx, s_cxx = self._filter(make, zk, False)
        x_ref, s_ref = self._filter(make, zk, True)
        x_py, s_py = self._filter(make, zk, True, everything_reference=True)
        self.assertEqual(s_cxx, s_ref)                           # the same boosts at the same steps
        self.assertIn(0.2, s_cxx)
        np.testing.assert_allclose(x_cxx, x_ref, rtol=0, atol=1e-12)
        # Against the all-Python run (Python filter loop too): the loops differ in rounding only.
        self.assertEqual(s_cxx, s_py)
        np.testing.assert_allclose(x_cxx, x_py, rtol=0, atol=1e-8)

    def test_adaptive_qr_on_a_mistuned_lane(self):
        rng = np.random.default_rng(11)
        y, zk = 0.0, np.zeros((300, 1))
        for k in range(300):
            zk[k, 0] = y + rng.normal(0, LATERAL_NOISE_STD)
            y += rng.normal(0, np.sqrt(30e-4))
        make = lambda m: AdaptiveQRKalmanController(m, window=20, q_min=0.5, q_max=10.0)
        x_cxx, s_cxx = self._filter(make, zk, False)
        x_ref, s_ref = self._filter(make, zk, True)
        x_py, s_py = self._filter(make, zk, True, everything_reference=True)
        self.assertEqual(len(s_cxx), len(s_ref))
        # Rounding (a trace summed in another order) is fed back through Q into the
        # filter for 280 steps: measured 2.5e-11 on the factor, then.
        np.testing.assert_allclose(s_cxx, s_ref, rtol=1e-9)
        np.testing.assert_allclose(x_cxx, x_ref, rtol=0, atol=1e-9)
        self.assertGreater(np.ptp(s_cxx), 0.5)                    # Q really was retuned
        np.testing.assert_allclose(s_cxx, s_py, rtol=1e-7)
        np.testing.assert_allclose(x_cxx, x_py, rtol=0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

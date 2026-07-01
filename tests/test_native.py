"""Unit tests for the optional C acceleration layer (modules/framework/native/).

Tests verify that every C kernel produces results numerically identical (within
floating-point tolerance) to the corresponding pure-Python reference path. All
tests are skipped gracefully when the extension is unavailable so that the suite
can run without a C compiler.
"""
import sys
import unittest
import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import HAS_C_ACCEL, ffi, lib, as_contiguous, dptr, iptr


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestKfPredict(unittest.TestCase):

    def _run(self, n, l, F, x, P, B, u, L, Q):
        x_c = np.empty(n, dtype=np.float64)
        P_c = np.empty((n, n), dtype=np.float64)
        lib.kf_predict(
            n, l,
            dptr(as_contiguous(F)), dptr(as_contiguous(x)), dptr(as_contiguous(P)),
            dptr(as_contiguous(B)) if B is not None else ffi.NULL,
            dptr(as_contiguous(u)) if u is not None else ffi.NULL,
            dptr(as_contiguous(L)) if L is not None else ffi.NULL,
            dptr(as_contiguous(Q)),
            dptr(x_c), dptr(P_c),
        )
        return x_c, P_c

    def _ref(self, n, F, x, P, B, u, L, Q):
        x_ref = F @ x
        if B is not None and u is not None:
            x_ref = x_ref + B @ u
        LQ = L @ Q @ L.T if L is not None else Q
        P_ref = F @ P @ F.T + LQ
        return x_ref, P_ref

    def test_no_control_no_L(self):
        n = 4
        F = np.eye(n) + 0.1 * np.random.randn(n, n)
        x = np.random.randn(n)
        P = np.eye(n) * 2.0
        Q = np.eye(n) * 0.1
        x_c, P_c = self._run(n, 0, F, x, P, None, None, None, Q)
        x_r, P_r = self._ref(n, F, x, P, None, None, None, Q)
        np.testing.assert_allclose(x_c, x_r, atol=1e-12)
        np.testing.assert_allclose(P_c, P_r, atol=1e-12)

    def test_with_control_and_L(self):
        n, l = 3, 2
        F = np.eye(n) + 0.05 * np.random.randn(n, n)
        x = np.random.randn(n)
        P = np.eye(n)
        B = np.random.randn(n, l)
        u = np.random.randn(l)
        L = np.random.randn(n, n)
        Q = np.eye(n) * 0.05
        x_c, P_c = self._run(n, l, F, x, P, B, u, L, Q)
        x_r, P_r = self._ref(n, F, x, P, B, u, L, Q)
        np.testing.assert_allclose(x_c, x_r, atol=1e-12)
        np.testing.assert_allclose(P_c, P_r, atol=1e-12)


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestKfUpdate(unittest.TestCase):

    def _joseph_ref(self, n, H, x, P, z, R):
        keep = ~np.isnan(z)
        if not keep.any():
            return x.copy(), P.copy()
        Hk, Rk, zk = H[keep, :], R[np.ix_(keep, keep)], z[keep]
        y = zk - Hk @ x
        S = Hk @ P @ Hk.T + Rk
        K = P @ Hk.T @ np.linalg.pinv(S)
        I = np.eye(n)
        IKH = I - K @ Hk
        x_out = x + K @ y
        P_out = IKH @ P @ IKH.T + K @ Rk @ K.T
        return x_out, P_out

    def test_full_update(self):
        n, m = 4, 2
        H = np.random.randn(m, n)
        x = np.random.randn(n)
        P = np.eye(n)
        z = np.random.randn(m)
        R = np.eye(m) * 0.1
        x_c, P_c = np.empty(n), np.empty((n, n))
        lib.kf_update(n, m, dptr(as_contiguous(H)), dptr(as_contiguous(x)),
                      dptr(as_contiguous(P)), dptr(as_contiguous(z)),
                      dptr(as_contiguous(R)), dptr(x_c), dptr(P_c))
        x_r, P_r = self._joseph_ref(n, H, x, P, z, R)
        np.testing.assert_allclose(x_c, x_r, atol=1e-10)
        np.testing.assert_allclose(P_c, P_r, atol=1e-10)

    def test_nan_masking(self):
        n, m = 4, 3
        H = np.eye(m, n)
        x = np.zeros(n)
        P = np.eye(n)
        z = np.array([1.0, np.nan, 2.0])
        R = np.eye(m) * 0.5
        x_c, P_c = np.empty(n), np.empty((n, n))
        lib.kf_update(n, m, dptr(as_contiguous(H)), dptr(as_contiguous(x)),
                      dptr(as_contiguous(P)), dptr(as_contiguous(z)),
                      dptr(as_contiguous(R)), dptr(x_c), dptr(P_c))
        x_r, P_r = self._joseph_ref(n, H, x, P, z, R)
        np.testing.assert_allclose(x_c, x_r, atol=1e-10)
        np.testing.assert_allclose(P_c, P_r, atol=1e-10)

    def test_all_nan_is_noop(self):
        n, m = 3, 2
        H = np.eye(m, n)
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(n) * 5.0
        z = np.array([np.nan, np.nan])
        R = np.eye(m)
        x_c, P_c = np.empty(n), np.empty((n, n))
        mv = lib.kf_update(n, m, dptr(as_contiguous(H)), dptr(as_contiguous(x)),
                           dptr(as_contiguous(P)), dptr(as_contiguous(z)),
                           dptr(as_contiguous(R)), dptr(x_c), dptr(P_c))
        self.assertEqual(mv, 0)
        np.testing.assert_allclose(x_c, x, atol=1e-14)
        np.testing.assert_allclose(P_c, P, atol=1e-14)


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestKfUpdateResidual(unittest.TestCase):

    def test_matches_kf_update_for_linear_h(self):
        n, m = 4, 2
        np.random.seed(7)
        H = np.random.randn(m, n)
        x = np.random.randn(n)
        P = np.eye(n)
        z = np.random.randn(m)
        R = np.eye(m) * 0.1
        y = z - H @ x

        x_upd, P_upd = np.empty(n), np.empty((n, n))
        lib.kf_update(n, m, dptr(as_contiguous(H)), dptr(as_contiguous(x)),
                      dptr(as_contiguous(P)), dptr(as_contiguous(z)),
                      dptr(as_contiguous(R)), dptr(x_upd), dptr(P_upd))

        x_res, P_res = np.empty(n), np.empty((n, n))
        lib.kf_update_residual(n, m, dptr(as_contiguous(H)), dptr(as_contiguous(x)),
                               dptr(as_contiguous(P)), dptr(as_contiguous(y)),
                               dptr(as_contiguous(R)), dptr(x_res), dptr(P_res))

        np.testing.assert_allclose(x_res, x_upd, atol=1e-12)
        np.testing.assert_allclose(P_res, P_upd, atol=1e-12)

    def test_nonlinear_residual_differs_from_kf_update(self):
        n, m = 2, 1
        H = np.array([[1.0, 0.0]])
        x = np.array([0.5, 0.0])
        P = np.eye(n)
        z = np.array([1.2])
        R = np.array([[0.1]])
        # kf_update computes y = z - H@x = 1.2 - 0.5 = 0.7; use a different value
        # to simulate a genuinely nonlinear h(x) residual
        y_nonlinear = np.array([0.3])

        x_lin, P_lin = np.empty(n), np.empty((n, n))
        lib.kf_update(n, m, dptr(as_contiguous(H)), dptr(as_contiguous(x)),
                      dptr(as_contiguous(P)), dptr(as_contiguous(z)),
                      dptr(as_contiguous(R)), dptr(x_lin), dptr(P_lin))

        x_nl, P_nl = np.empty(n), np.empty((n, n))
        lib.kf_update_residual(n, m, dptr(as_contiguous(H)), dptr(as_contiguous(x)),
                               dptr(as_contiguous(P)), dptr(as_contiguous(y_nonlinear)),
                               dptr(as_contiguous(R)), dptr(x_nl), dptr(P_nl))

        self.assertFalse(np.allclose(x_lin, x_nl))


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestKfForwardLinear(unittest.TestCase):
    """Bit-for-bit comparison of the fused C forward pass against the Python loop."""

    def _python_forward(self, n, m, l, N, F, H, Q, R, B, L, zk, uk, x0, P0):
        x, P = x0.copy(), P0.copy()
        x_prior = np.empty((N, n))
        P_prior = np.empty((N, n, n))
        x_post  = np.empty((N, n))
        P_post  = np.empty((N, n, n))
        inno    = np.empty((N, m))
        for k in range(N):
            xp = F @ x
            if B is not None and l > 0:
                xp = xp + B @ uk[k]
            LQ = L @ Q @ L.T if L is not None else Q
            Pp = F @ P @ F.T + LQ
            x_prior[k], P_prior[k] = xp, Pp

            z = zk[k]
            keep = ~np.isnan(z)
            Hk = H[keep, :]
            Rk = R[np.ix_(keep, keep)]
            y = z[keep] - Hk @ xp
            inno[k] = z - H @ xp

            S = Hk @ Pp @ Hk.T + Rk
            K = Pp @ Hk.T @ np.linalg.pinv(S)
            IKH = np.eye(n) - K @ Hk
            x = xp + K @ y
            P = IKH @ Pp @ IKH.T + K @ Rk @ K.T
            x_post[k], P_post[k] = x, P
        return x_prior, P_prior, x_post, P_post, inno

    def test_constant_velocity_vs_python(self):
        np.random.seed(42)
        n, m, l, N = 4, 2, 0, 200
        F = np.array([[1, 0.1, 0, 0], [0, 1, 0, 0],
                      [0, 0, 1, 0.1], [0, 0, 0, 1]], dtype=np.float64)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        Q = np.eye(n, dtype=np.float64) * 0.01
        R = np.eye(m, dtype=np.float64) * 0.5
        x0 = np.zeros(n)
        P0 = np.eye(n) * 10.0
        zk = np.random.randn(N, m)
        uk = np.zeros((N, max(l, 1)))

        xpr_r, Ppr_r, xpo_r, Ppo_r, inno_r = self._python_forward(
            n, m, l, N, F, H, Q, R, None, None, zk, uk, x0, P0)

        xpr_c = np.empty((N, n)); Ppr_c = np.empty((N, n, n))
        xpo_c = np.empty((N, n)); Ppo_c = np.empty((N, n, n))
        inno_c = np.empty((N, m))
        lib.kf_forward_linear(
            n, m, l, N,
            dptr(as_contiguous(F)), dptr(as_contiguous(H)),
            dptr(as_contiguous(Q)), dptr(as_contiguous(R)),
            ffi.NULL, ffi.NULL,
            dptr(as_contiguous(zk)), dptr(as_contiguous(uk)),
            dptr(as_contiguous(x0)), dptr(as_contiguous(P0)),
            dptr(xpr_c), dptr(Ppr_c), dptr(xpo_c), dptr(Ppo_c), dptr(inno_c),
        )
        np.testing.assert_allclose(xpr_c, xpr_r, atol=1e-10, err_msg="x_prior mismatch")
        np.testing.assert_allclose(xpo_c, xpo_r, atol=1e-10, err_msg="x_post mismatch")
        np.testing.assert_allclose(Ppr_c, Ppr_r, atol=1e-10, err_msg="P_prior mismatch")
        np.testing.assert_allclose(Ppo_c, Ppo_r, atol=1e-10, err_msg="P_post mismatch")

    def test_nan_masking_in_forward(self):
        np.random.seed(11)
        n, m, l, N = 2, 2, 0, 50
        F = np.eye(n)
        H = np.eye(m, n)
        Q = np.eye(n) * 0.1
        R = np.eye(m) * 0.5
        x0, P0 = np.zeros(n), np.eye(n)
        zk = np.random.randn(N, m)
        zk[::3, 1] = np.nan
        uk = np.zeros((N, 1))

        xpr_r, _, xpo_r, Ppo_r, _ = self._python_forward(
            n, m, l, N, F, H, Q, R, None, None, zk, uk, x0, P0)

        xpr_c = np.empty((N, n)); Ppr_c = np.empty((N, n, n))
        xpo_c = np.empty((N, n)); Ppo_c = np.empty((N, n, n)); inno_c = np.empty((N, m))
        lib.kf_forward_linear(
            n, m, l, N,
            dptr(as_contiguous(F)), dptr(as_contiguous(H)),
            dptr(as_contiguous(Q)), dptr(as_contiguous(R)),
            ffi.NULL, ffi.NULL,
            dptr(as_contiguous(zk)), dptr(as_contiguous(uk)),
            dptr(as_contiguous(x0)), dptr(as_contiguous(P0)),
            dptr(xpr_c), dptr(Ppr_c), dptr(xpo_c), dptr(Ppo_c), dptr(inno_c),
        )
        np.testing.assert_allclose(xpo_c, xpo_r, atol=1e-10)
        np.testing.assert_allclose(Ppo_c, Ppo_r, atol=1e-10)


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestKfBackwardRts(unittest.TestCase):

    def _python_rts(self, n, N, x_post, P_post, x_prior, P_prior, F):
        xs = np.empty((N, n)); Ps = np.empty((N, n, n))
        xs[-1], Ps[-1] = x_post[-1], P_post[-1]
        for k in reversed(range(N - 1)):
            Pp = P_prior[k + 1]
            C = P_post[k] @ F.T @ np.linalg.pinv(Pp)
            xs[k] = x_post[k] + C @ (xs[k + 1] - x_prior[k + 1])
            Ps[k] = P_post[k] + C @ (Ps[k + 1] - Pp) @ C.T
        return xs, Ps

    def test_rts_vs_python(self):
        np.random.seed(3)
        n, N = 4, 100
        F = np.array([[1, 0.1, 0, 0], [0, 1, 0, 0],
                      [0, 0, 1, 0.1], [0, 0, 0, 1]])
        x_post  = np.random.randn(N, n)
        P_post  = np.array([np.eye(n) + 0.1 * i for i in range(N)])
        x_prior = np.random.randn(N, n)
        P_prior = np.array([np.eye(n) + 0.2 * i for i in range(N)])

        xs_r, Ps_r = self._python_rts(n, N, x_post, P_post, x_prior, P_prior, F)

        xs_c = np.empty((N, n)); Ps_c = np.empty((N, n, n))
        lib.kf_backward_rts(
            n, N,
            dptr(as_contiguous(x_post)), dptr(as_contiguous(P_post)),
            dptr(as_contiguous(x_prior)), dptr(as_contiguous(P_prior)),
            dptr(as_contiguous(F)),
            dptr(xs_c), dptr(Ps_c),
        )
        np.testing.assert_allclose(xs_c, xs_r, atol=1e-10)
        np.testing.assert_allclose(Ps_c, Ps_r, atol=1e-10)


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestPfHelpers(unittest.TestCase):

    def test_weight_update_vs_python(self):
        np.random.seed(5)
        n_particles, mv = 200, 3
        z   = np.random.randn(mv)
        zp  = np.random.randn(n_particles, mv)
        R   = np.eye(mv) * 0.5
        Rinv = np.linalg.pinv(R)
        w   = np.full(n_particles, 1.0 / n_particles)
        logw_r = np.log(w + 1e-300)
        for i in range(n_particles):
            dz = z - zp[i]
            logw_r[i] += -0.5 * dz @ Rinv @ dz

        logw_c = np.log(w + 1e-300)
        lib.pf_weight_update(
            n_particles, mv,
            dptr(as_contiguous(z)), dptr(as_contiguous(zp)),
            dptr(as_contiguous(Rinv)), dptr(logw_c),
        )
        np.testing.assert_allclose(logw_c, logw_r, atol=1e-12)

    def test_systematic_resample_length(self):
        N = 300
        w = np.random.dirichlet(np.ones(N))
        idx = np.empty(N, dtype=np.intc)
        lib.pf_systematic_resample(N, dptr(as_contiguous(w)), float(np.random.uniform()), iptr(idx))
        self.assertEqual(len(idx), N)
        self.assertTrue(np.all((idx >= 0) & (idx < N)))

    def test_systematic_resample_preserves_distribution(self):
        N = 1000
        w = np.zeros(N)
        w[N // 2] = 1.0
        idx = np.empty(N, dtype=np.intc)
        lib.pf_systematic_resample(N, dptr(as_contiguous(w)), 0.5 / N, iptr(idx))
        np.testing.assert_array_equal(idx, np.full(N, N // 2))


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestUkfHelpers(unittest.TestCase):

    def test_weighted_outer_sum_vs_python(self):
        np.random.seed(9)
        n_sigma, n = 9, 4
        Wc   = np.random.dirichlet(np.ones(n_sigma))
        Y    = np.random.randn(n_sigma, n)
        mean = np.einsum("i,ij->j", Wc, Y)
        ref  = sum(Wc[i] * np.outer(Y[i] - mean, Y[i] - mean) for i in range(n_sigma))

        P_out = np.empty((n, n))
        lib.ukf_weighted_outer_sum(
            n_sigma, n,
            dptr(as_contiguous(Wc)), dptr(as_contiguous(Y)), dptr(as_contiguous(mean)),
            dptr(P_out),
        )
        np.testing.assert_allclose(P_out, ref, atol=1e-12)

    def test_weighted_cross_sum_vs_python(self):
        np.random.seed(13)
        n_sigma, na, nb = 9, 4, 2
        Wc = np.random.dirichlet(np.ones(n_sigma))
        A  = np.random.randn(n_sigma, na)
        B  = np.random.randn(n_sigma, nb)
        ref = sum(Wc[i] * np.outer(A[i], B[i]) for i in range(n_sigma))

        C_out = np.empty((na, nb))
        lib.ukf_weighted_cross_sum(
            n_sigma, na, nb,
            dptr(as_contiguous(Wc)), dptr(as_contiguous(A)), dptr(as_contiguous(B)),
            dptr(C_out),
        )
        np.testing.assert_allclose(C_out, ref, atol=1e-12)


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestDiagBatch(unittest.TestCase):

    def test_nis_vs_python(self):
        np.random.seed(17)
        N, m = 50, 2
        nu = np.random.randn(N, m)
        S  = np.array([np.eye(m) + 0.1 * np.random.randn(m, m) for _ in range(N)])
        S  = np.array([(s + s.T) / 2 + np.eye(m) for s in S])
        ref = np.array([float(nu[k] @ np.linalg.solve(S[k], nu[k])) for k in range(N)])

        nis = np.empty(N)
        lib.diag_nis_batch(N, m, dptr(as_contiguous(nu)), dptr(as_contiguous(S)), dptr(nis))
        np.testing.assert_allclose(nis, ref, atol=1e-10)

    def test_nees_vs_python(self):
        np.random.seed(19)
        N, n = 50, 3
        e = np.random.randn(N, n)
        P = np.array([np.eye(n) + 0.05 * (i + 1) for i in range(N)])
        ref = np.array([float(e[k] @ np.linalg.pinv(P[k]) @ e[k]) for k in range(N)])

        nees = np.empty(N)
        lib.diag_nees_batch(N, n, dptr(as_contiguous(e)), dptr(as_contiguous(P)), dptr(nees))
        np.testing.assert_allclose(nees, ref, atol=1e-9)


@unittest.skipUnless(HAS_C_ACCEL, "C extension not built — skipping native tests")
class TestFilterIntegration(unittest.TestCase):
    """End-to-end: verify the native fast-path and Python fallback give identical records."""

    def _make_data(self):
        from modules.apps import BallisticLinearModel
        m = BallisticLinearModel("test", fs=50)
        np.random.seed(99)
        sample = m.sample(4.0)
        return np.asarray(sample.zk)

    def test_forward_native_vs_python(self):
        import modules.framework.native as nat_mod
        from modules.apps import BallisticLinearModel

        zk = self._make_data()

        m_c = BallisticLinearModel("test", fs=50)
        rec_c = m_c.filter(zk)
        xk_c = np.asarray(rec_c.xk)

        orig = nat_mod.HAS_C_ACCEL
        try:
            nat_mod.HAS_C_ACCEL = False
            m_p = BallisticLinearModel("test", fs=50)
            rec_p = m_p.filter(zk)
            xk_p = np.asarray(rec_p.xk)
        finally:
            nat_mod.HAS_C_ACCEL = orig

        np.testing.assert_allclose(xk_c, xk_p, atol=1e-10,
                                   err_msg="native vs Python forward mismatch")

    def test_backward_native_vs_python(self):
        import modules.framework.native as nat_mod
        from modules.apps import BallisticLinearModel

        zk = self._make_data()

        m_c = BallisticLinearModel("test", fs=50)
        rec_c = m_c.filter(zk, smooth=True)
        xk_c = np.asarray(rec_c.xk)

        orig = nat_mod.HAS_C_ACCEL
        try:
            nat_mod.HAS_C_ACCEL = False
            m_p = BallisticLinearModel("test", fs=50)
            rec_p = m_p.filter(zk, smooth=True)
            xk_p = np.asarray(rec_p.xk)
        finally:
            nat_mod.HAS_C_ACCEL = orig

        np.testing.assert_allclose(xk_c, xk_p, atol=1e-10,
                                   err_msg="native vs Python backward mismatch")

    def test_fallback_runs_without_extension(self):
        import modules.framework.native as nat_mod
        from modules.apps import BallisticLinearModel

        zk = self._make_data()
        orig = nat_mod.HAS_C_ACCEL
        try:
            nat_mod.HAS_C_ACCEL = False
            m = BallisticLinearModel("test", fs=50)
            rec = m.filter(zk, smooth=True)
            self.assertGreater(len(rec.xk), 0)
        finally:
            nat_mod.HAS_C_ACCEL = orig


if __name__ == "__main__":
    unittest.main()

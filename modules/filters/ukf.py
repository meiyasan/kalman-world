import inspect
import numpy as np

# Expect your codebase provides these:
# - KalmanStateModel-like API on each model: n, m, x, P, Q, R, optional L, M, and callables f, h
# - KalmanCompositeModel(*models) -> merged model exposing same API
from ..models.composite import KalmanCompositeModel

class UnscentedKalmanFilter:
    """
    UKF wrapper operating on a provided model (or a KalmanCompositeModel of several).
    Auto-detects non-additive noise via f(..., w) / h(..., v).
    """

    NAME = "Unscented Kalman Model"
    DESCRIPTION = "UKF with automatic augmented-state for non-additive noise."
    VERSION = "2.0"

    def __init__(self, *models, model=None, alpha=1e-3, beta=2.0, kappa=0.0):
        # ---- resolve model / composite ----
        if model is None:
            if len(models) == 0:
                raise ValueError("UnscentedKalmanFilter requires at least one model or model=...")
            model = models[0] if len(models) == 1 else KalmanCompositeModel(*models)
        self.model = model

        # ---- UT parameters ----
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.kappa = float(kappa)

        # ---- fast aliases to model state ----
        self.n = int(self.model.n)
        self.m = int(self.model.m)

        # local filter state (kept in sync with model)
        # Models store initial state as x0/P0, not x/P (which live on the engine)
        x0 = getattr(self.model, "x0", None)
        if x0 is None:
            x0 = getattr(self.model, "x", None)
        P0 = getattr(self.model, "P0", None)
        if P0 is None:
            P0 = getattr(self.model, "P", None)
        if x0 is None:
            raise ValueError("Model has no initial state (x0 is None); call model.initial_state() first.")
        if P0 is None:
            raise ValueError("Model has no initial covariance (P0 is None); call model.initial_state() first.")
        self.x = np.asarray(x0, dtype=np.float64).reshape(-1)
        self.P = np.asarray(P0, dtype=np.float64)

        # Non-additive detection: f(..., w) / h(..., v)
        self._nonadd_proc = self._accepts_param(self.model.f, "w")
        self._nonadd_meas = self._accepts_param(self.model.h, "v")

        # base (non-augmented) UT weights
        self._lambda, self._gamma, self.Wm, self.Wc = self._ut_weights(self.n)

    # ---------- utility ----------
    @staticmethod
    def _to_vec(a):
        a = np.asarray(a, dtype=np.float64)
        return a.reshape(-1) if a.ndim > 1 else a

    @staticmethod
    def _sym_psd(S):
        S = 0.5 * (S + S.T)
        eps = 1e-12 * np.eye(S.shape[0], dtype=S.dtype)
        return S + eps

    @staticmethod
    def _chol(S):
        S = UnscentedKalmanFilter._sym_psd(S)
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            d = S.shape[0]
            for k in range(6):
                jitter = (10.0**k) * 1e-12
                try:
                    return np.linalg.cholesky(S + jitter * np.eye(d))
                except np.linalg.LinAlgError:
                    pass
            raise

    def _ut_weights(self, n_eff):
        lam = self.alpha**2 * (n_eff + self.kappa) - n_eff
        gam = np.sqrt(n_eff + lam)
        Wm  = np.full(2 * n_eff + 1, 0.5 / (n_eff + lam), dtype=np.float64)
        Wc  = Wm.copy()
        Wm[0] = lam / (n_eff + lam)
        Wc[0] = lam / (n_eff + lam) + (1.0 - self.alpha**2 + self.beta)
        return lam, gam, Wm, Wc

    @staticmethod
    def _accepts_param(fn, name: str) -> bool:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        for p in sig.parameters.values():
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name == name:
                return True
        return False

    # ---------- model adapters ----------
    def _Qw(self, x=None, u=None, t=None):
        Q = self.model.Q
        # Q is always a bound method on KalmanModel subclasses; call with t only
        return Q(t=t) if callable(Q) else np.asarray(Q, dtype=np.float64)

    def _Rv(self, x=None, t=None):
        R = self.model.R
        # R is always a bound method on KalmanModel subclasses; call with t only
        return R(t=t) if callable(R) else np.asarray(R, dtype=np.float64)

    def _Lx(self, x=None, u=None, t=None):
        L = getattr(self.model, "L", None)
        if callable(L):
            # KalmanModel.L(x, u, w, t) returns None when w=None → treat as identity
            result = L(x=x, u=u, t=t)
            return result if result is not None else np.eye(self.n, dtype=np.float64)
        if L is None:
            return np.eye(self.n, dtype=np.float64)
        return np.asarray(L, dtype=np.float64)

    def _Mx(self, x=None, t=None):
        M = getattr(self.model, "M", None)
        if callable(M):
            # KalmanModel.M(x, v, t) returns None when v=None → treat as identity
            result = M(x=x, t=t)
            return result if result is not None else np.eye(self.m, dtype=np.float64)
        if M is None:
            return np.eye(self.m, dtype=np.float64)
        return np.asarray(M, dtype=np.float64)

    # ---------- sigma points ----------
    def _sigmas(self, mean, cov, gamma):
        mean = self._to_vec(mean)
        S = self._chol(cov)
        n = mean.shape[0]
        sig = np.zeros((2 * n + 1, n), dtype=np.float64)
        sig[0] = mean
        for i in range(n):
            col = gamma * S[:, i]
            sig[i + 1]      = mean + col
            sig[n + i + 1]  = mean - col
        return sig

    # ---------- function wrappers ----------
    def _f_call(self, x, u=None, t=None, w=None):
        # Reshape w to column vector so BallisticModel.f (and similar) handle it correctly
        if w is not None:
            w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
        try:
            return self._to_vec(self.model.f(x, u=u, w=w, t=t))
        except TypeError:
            return self._to_vec(self.model.f(x, u=u, t=t))

    def _h_call(self, x, t=None, v=None):
        try:
            return self._to_vec(self.model.h(x, v=v, t=t))
        except TypeError:
            return self._to_vec(self.model.h(x, t=t))

    # ---------- UKF steps ----------
    def __predict(self, u=None, t=None):
        if self._nonadd_proc:
            # Augment with process noise w ~ N(0, Qw)
            Qw = np.asarray(self._Qw(self.x, u, t), dtype=np.float64)
            n_w = Qw.shape[0]
            n_a = self.n + n_w
            _, gamma, Wm, Wc = self._ut_weights(n_a)

            mean_a = np.zeros(n_a, dtype=np.float64)
            mean_a[:self.n] = self.x

            P_a = np.zeros((n_a, n_a), dtype=np.float64)
            P_a[:self.n, :self.n] = self.P
            P_a[self.n:, self.n:] = Qw

            sig_a = self._sigmas(mean_a, P_a, gamma)
            Y = np.zeros((sig_a.shape[0], self.n), dtype=np.float64)
            for i, sp in enumerate(sig_a):
                x_i = sp[:self.n]
                w_i = sp[self.n:]
                Y[i] = self._f_call(x_i, u=u, t=t, w=w_i)

            x_pred = np.einsum('i,ij->j', Wm, Y)
            P_pred = np.zeros((self.n, self.n), dtype=np.float64)
            for i in range(Y.shape[0]):
                dy = Y[i] - x_pred
                P_pred += Wc[i] * np.outer(dy, dy)

            self.x = x_pred
            self.P = self._sym_psd(P_pred)

        else:
            # Additive process: standard UT + Σx = L Qw Lᵀ
            _, gamma, Wm, Wc = self._ut_weights(self.n)
            sig = self._sigmas(self.x, self.P, gamma)
            Y = np.array([self._f_call(sp, u=u, t=t) for sp in sig])
            x_pred = np.einsum('i,ij->j', Wm, Y)
            P_pred = np.zeros((self.n, self.n), dtype=np.float64)
            for i in range(Y.shape[0]):
                dy = Y[i] - x_pred
                P_pred += Wc[i] * np.outer(dy, dy)
            Qw = np.asarray(self._Qw(self.x, u, t), dtype=np.float64)
            L  = self._Lx(self.x, u, t)
            P_pred += self._sym_psd(L @ Qw @ L.T)
            self.x = x_pred
            self.P = self._sym_psd(P_pred)

        # keep underlying model in sync
        self.model.x = self.x.copy()
        self.model.P = self.P.copy()
        return self.x, self.P

    def __update(self, z, t=None):
        z = self._to_vec(z)

        if self._nonadd_meas:
            # Augment with measurement noise v ~ N(0, Rv)
            Rv = np.asarray(self._Rv(self.x, t), dtype=np.float64)
            n_v = Rv.shape[0]
            n_a = self.n + n_v
            _, gamma, Wm, Wc = self._ut_weights(n_a)

            mean_a = np.zeros(n_a, dtype=np.float64)
            mean_a[:self.n] = self.x

            P_a = np.zeros((n_a, n_a), dtype=np.float64)
            P_a[:self.n, :self.n] = self.P
            P_a[self.n:, self.n:] = Rv

            sig_a = self._sigmas(mean_a, P_a, gamma)
            Z = np.zeros((sig_a.shape[0], self.m), dtype=np.float64)
            for i, sp in enumerate(sig_a):
                x_i = sp[:self.n]
                v_i = sp[self.n:]
                Z[i] = self._h_call(x_i, t=t, v=v_i)

            z_pred = np.einsum('i,ij->j', Wm, Z)
            S   = np.zeros((self.m, self.m), dtype=np.float64)
            Pxz = np.zeros((self.n, self.m), dtype=np.float64)
            for i in range(Z.shape[0]):
                dz = Z[i] - z_pred
                dx = sig_a[i][:self.n] - self.x
                S  += Wc[i] * np.outer(dz, dz)
                Pxz += Wc[i] * np.outer(dx, dz)

            S = self._sym_psd(S)
            K = Pxz @ np.linalg.inv(S)
            self.x = self.x + K @ (z - z_pred)
            self.P = self._sym_psd(self.P - K @ S @ K.T)

        else:
            # Additive measurement: standard UT + Σz = M Rv Mᵀ
            _, gamma, Wm, Wc = self._ut_weights(self.n)
            sig = self._sigmas(self.x, self.P, gamma)
            Z = np.array([self._h_call(sp, t=t) for sp in sig])
            z_pred = np.einsum('i,ij->j', Wm, Z)

            S   = np.zeros((self.m, self.m), dtype=np.float64)
            Pxz = np.zeros((self.n, self.m), dtype=np.float64)
            for i in range(Z.shape[0]):
                dz = Z[i] - z_pred
                dx = sig[i] - self.x
                S  += Wc[i] * np.outer(dz, dz)
                Pxz += Wc[i] * np.outer(dx, dz)

            Rv = np.asarray(self._Rv(self.x, t), dtype=np.float64)
            M  = self._Mx(self.x, t)
            S += self._sym_psd(M @ Rv @ M.T)

            S = self._sym_psd(S)
            K = Pxz @ np.linalg.inv(S)
            self.x = self.x + K @ (z - z_pred)
            self.P = self._sym_psd(self.P - K @ S @ K.T)

        # keep model in sync
        self.model.x = self.x.copy()
        self.model.P = self.P.copy()
        return self.x, self.P

    # ---------- public API ----------
    def reset(self):
        """Reset filter state to model initial conditions."""
        x0 = getattr(self.model, "x0", None)
        if x0 is None:
            x0 = getattr(self.model, "x", None)
        P0 = getattr(self.model, "P0", None)
        if P0 is None:
            P0 = getattr(self.model, "P", None)
        self.x = np.asarray(x0, dtype=np.float64).reshape(-1)
        self.P = np.asarray(P0, dtype=np.float64)

    def filter(self, zk, uk=None, *, reset=True):
        """Run the UKF over a batch of measurements.

        Parameters
        ----------
        zk : array_like, shape (N, m)
            Measurement sequence. NaN rows trigger prediction-only steps.
        uk : array_like, shape (N, l) or None
            Control input sequence.
        reset : bool
            Reset state to model.x0 / model.P0 before filtering.

        Returns
        -------
        xk : ndarray, shape (N, n)
            Filtered state sequence.
        Pk : ndarray, shape (N, n, n)
            Filtered covariance sequence.
        """
        zk = np.asarray(zk, dtype=np.float64)
        if zk.ndim == 1:
            zk = zk.reshape(-1, 1)
        N = len(zk)

        if reset:
            self.reset()

        fs = getattr(self.model, "fs", None) or 1.0
        t0 = getattr(self.model, "t0", 0.0) or 0.0
        dt = 1.0 / float(fs)

        l = getattr(self.model, "l", 0) or 0
        if uk is None:
            uk = np.zeros((N, l))
        uk = np.asarray(uk, dtype=np.float64)
        if uk.ndim == 1:
            uk = uk.reshape(-1, 1)

        xk_out = np.zeros((N, self.n), dtype=np.float64)
        Pk_out = np.zeros((N, self.n, self.n), dtype=np.float64)

        for k in range(N):
            t = t0 + k * dt
            u_k = uk[k] if l > 0 else None
            self.__predict(u=u_k, t=t)
            z_k = zk[k]
            if not np.any(np.isnan(z_k)):
                self.__update(z_k, t=t)
            xk_out[k] = self.x
            Pk_out[k] = self.P

        return xk_out, Pk_out
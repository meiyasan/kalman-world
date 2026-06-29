"""Unscented Kalman engine (UKF) — sigma-point predict/update, stateless."""
from __future__ import annotations
import numpy as np

from ..engine import KalmanEngine
from ..constants import KalmanEngineKind


class UnscentedKalmanEngine(KalmanEngine):
    """
    Sigma-point engine. Uses ``model.f`` / ``model.h`` directly (no Jacobians).

    Noise handling is automatic per call:
      - non-additive (``model.f(...,w)`` / ``model.h(...,v)``) → augmented-state sigma points;
      - otherwise additive with ``L Q Lᵀ`` / ``M R Mᵀ`` coupling.

    Returns an identity transition for the generic RTS step (``_smooth`` inherited).
    The measurement update is missing-measurement aware (NaN channels dropped).
    """

    NAME = "Unscented Kalman Engine"
    DESCRIPTION = "UKF engine with automatic augmented-state for non-additive noise."
    VERSION = "3.0"
    KIND = KalmanEngineKind.UNSCENTED

    def __init__(self, alpha=1e-3, beta=2.0, kappa=0.0):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

    # ==========================================================
    # Utilities
    # ==========================================================
    @staticmethod
    def _to_vec(a):
        a = np.asarray(a, dtype=np.float64)
        return a.reshape(-1) if a.ndim > 1 else a

    @staticmethod
    def _sym_psd(S):
        S = 0.5 * (S + S.T)
        return S + 1e-12 * np.eye(S.shape[0], dtype=S.dtype)

    @staticmethod
    def _chol(S):
        S = UnscentedKalmanEngine._sym_psd(S)
        try:
            return np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            d = S.shape[0]
            for k in range(6):
                try:
                    return np.linalg.cholesky(S + (10.0 ** k) * 1e-12 * np.eye(d))
                except np.linalg.LinAlgError:
                    pass
            raise

    def _ut_weights(self, n_eff):
        lam = self.alpha ** 2 * (n_eff + self.kappa) - n_eff
        gamma = np.sqrt(n_eff + lam)
        Wm = np.full(2 * n_eff + 1, 0.5 / (n_eff + lam), dtype=np.float64)
        Wc = Wm.copy()
        Wm[0] = lam / (n_eff + lam)
        Wc[0] = lam / (n_eff + lam) + (1.0 - self.alpha ** 2 + self.beta)
        return gamma, Wm, Wc

    def _sigmas(self, mean, cov, gamma):
        mean = self._to_vec(mean)
        S = self._chol(cov)
        n = mean.shape[0]
        sig = np.zeros((2 * n + 1, n), dtype=np.float64)
        sig[0] = mean
        for i in range(n):
            col = gamma * S[:, i]
            sig[i + 1] = mean + col
            sig[n + i + 1] = mean - col
        return sig

    # ---- model adapters (model passed in) ----
    @staticmethod
    def _Q(model, t=None):
        return np.asarray(model.Q(t=t), dtype=np.float64)

    @staticmethod
    def _R(model, t=None):
        return np.asarray(model.R(t=t), dtype=np.float64)

    @staticmethod
    def _L(model, x=None, u=None, t=None):
        L = getattr(model, "L", None)
        if callable(L):
            try:
                res = L(x=x, u=u, w=None, t=t)
            except TypeError:
                res = L(t=t)
            return res if res is not None else np.eye(model.n, dtype=np.float64)
        return np.eye(model.n, dtype=np.float64)

    @staticmethod
    def _M(model, x=None, t=None):
        M = getattr(model, "M", None)
        if callable(M):
            try:
                res = M(x=x, v=None, t=t)
            except TypeError:
                res = M(t=t)
            return res if res is not None else np.eye(model.m, dtype=np.float64)
        return np.eye(model.m, dtype=np.float64)

    def _f_call(self, model, x, u=None, t=None, w=None):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        if w is not None:
            w = np.asarray(w, dtype=np.float64).reshape(-1, 1)
        try:
            return self._to_vec(model.f(x, u=u, w=w, t=t))
        except TypeError:
            return self._to_vec(model.f(x, u=u, t=t))

    def _h_call(self, model, x, t=None, v=None):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
        if v is not None:
            v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
        try:
            return self._to_vec(model.h(x, v=v, t=t))
        except TypeError:
            return self._to_vec(model.h(x, t=t))

    # ==========================================================
    # Prediction and update (stateless: (model, x, P, ...))
    # ==========================================================
    def _predict(self, model, x, P, u=None, t=None, **kw):
        if u is not None:
            u = np.asarray(u).reshape(-1, 1)
        u = u if model.l > 0 else None

        n = model.n
        x_vec = self._to_vec(x)
        P = np.asarray(P, dtype=np.float64)

        if self._accepts_param(model.f, "w"):
            Qw = self._Q(model, t)
            n_a = n + Qw.shape[0]
            gamma, Wm, Wc = self._ut_weights(n_a)
            mean_a = np.zeros(n_a); mean_a[:n] = x_vec
            P_a = np.zeros((n_a, n_a)); P_a[:n, :n] = P; P_a[n:, n:] = Qw
            sig_a = self._sigmas(mean_a, P_a, gamma)
            Y = np.array([self._f_call(model, sp[:n], u=u, t=t, w=sp[n:]) for sp in sig_a])
            x_pred = np.einsum("i,ij->j", Wm, Y)
            P_pred = sum(Wc[i] * np.outer(Y[i] - x_pred, Y[i] - x_pred) for i in range(Y.shape[0]))
            P_pred = self._sym_psd(P_pred)
        else:
            gamma, Wm, Wc = self._ut_weights(n)
            sig = self._sigmas(x_vec, P, gamma)
            Y = np.array([self._f_call(model, sp, u=u, t=t) for sp in sig])
            x_pred = np.einsum("i,ij->j", Wm, Y)
            P_pred = sum(Wc[i] * np.outer(Y[i] - x_pred, Y[i] - x_pred) for i in range(Y.shape[0]))
            L = self._L(model, x_vec.reshape(-1, 1), u, t)
            P_pred = self._sym_psd(P_pred + self._sym_psd(L @ self._Q(model, t) @ L.T))

        return x_pred.reshape(-1, 1), P_pred, np.eye(n)

    def _update(self, model, x, P, z, t=None, **kw):
        n = model.n
        m = model.m
        x_pred = self._to_vec(x)
        P_pred = np.asarray(P, dtype=np.float64)
        z = self._to_vec(z)
        keep = ~np.isnan(z)
        if not keep.any():
            return np.asarray(x).reshape(-1, 1), P_pred

        if self._accepts_param(model.h, "v"):
            Rv = self._R(model, t)
            n_a = n + Rv.shape[0]
            gamma, Wm, Wc = self._ut_weights(n_a)
            mean_a = np.zeros(n_a); mean_a[:n] = x_pred
            P_a = np.zeros((n_a, n_a)); P_a[:n, :n] = P_pred; P_a[n:, n:] = Rv
            sig_a = self._sigmas(mean_a, P_a, gamma)
            Z = np.array([self._h_call(model, sp[:n], t=t, v=sp[n:]) for sp in sig_a])
            z_pred = np.einsum("i,ij->j", Wm, Z)
            S = np.zeros((m, m)); Pxz = np.zeros((n, m))
            for i in range(Z.shape[0]):
                dz = Z[i] - z_pred
                dx = sig_a[i][:n] - x_pred
                S += Wc[i] * np.outer(dz, dz)
                Pxz += Wc[i] * np.outer(dx, dz)
            S = self._sym_psd(S)
        else:
            gamma, Wm, Wc = self._ut_weights(n)
            sig = self._sigmas(x_pred, P_pred, gamma)
            Z = np.array([self._h_call(model, sp, t=t) for sp in sig])
            z_pred = np.einsum("i,ij->j", Wm, Z)
            S = np.zeros((m, m)); Pxz = np.zeros((n, m))
            for i in range(Z.shape[0]):
                dz = Z[i] - z_pred
                dx = sig[i] - x_pred
                S += Wc[i] * np.outer(dz, dz)
                Pxz += Wc[i] * np.outer(dx, dz)
            M = self._M(model, x_pred.reshape(-1, 1), t)
            S = self._sym_psd(S + self._sym_psd(M @ self._R(model, t) @ M.T))

        # Multi-rate: restrict the update to the available measurement channels.
        if not keep.all():
            idx = np.where(keep)[0]
            S = S[np.ix_(idx, idx)]
            Pxz = Pxz[:, idx]
            z = z[idx]
            z_pred = z_pred[idx]

        K = Pxz @ np.linalg.pinv(S)
        x_new = (x_pred + K @ (z - z_pred)).reshape(-1, 1)
        P_new = self._sym_psd(P_pred - K @ S @ K.T)
        return x_new, P_new

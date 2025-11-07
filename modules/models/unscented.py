from __future__ import annotations
import numpy as np

from ..framework.filter import KalmanFilter

class KalmanUnscentedFilter(KalmanFilter):
    """
    Unscented Kalman Filter (UKF) engine.
    Derived from KalmanFilter but:
      - ignores Jacobians (F, H)
      - uses sigma-point propagation
      - reuses forward()/backward() infrastructure
    """

    def __init__(self, *models, alpha=1e-3, beta=2.0, kappa=0.0):
        # Initialize parent EKF engine (keeps hooks and logging)
        super().__init__(*models)

        # UKF sigma-point scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        n = self.model.n
        self.lambda_ = self.alpha**2 * (n + self.kappa) - n
        self.gamma = np.sqrt(n + self.lambda_)

        # Weights for mean and covariance
        self.Wm = np.full(2 * n + 1, 1 / (2 * (n + self.lambda_)))
        self.Wc = np.copy(self.Wm)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

    # ==========================================================
    # Disable Jacobian checks (not relevant to UKF)
    # ==========================================================
    def _check_shapes(self, *_, **__):
        return True

    # ==========================================================
    # Sigma-point generation and transform
    # ==========================================================
    def _sigma_points(self, x, P):
        """Generate sigma points around mean state."""
        x = x.reshape(-1, 1)
        n = x.shape[0]
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:
            S = np.linalg.cholesky(P + np.eye(n) * 1e-9)

        sigma = np.zeros((n, 2 * n + 1))
        sigma[:, 0] = x.ravel()
        for i in range(n):
            sigma[:, i + 1] = (x + self.gamma * S[:, [i]]).ravel()
            sigma[:, n + i + 1] = (x - self.gamma * S[:, [i]]).ravel()
        return sigma

    def _unscented_transform(self, X, noise_cov):
        """Recover mean and covariance from sigma points."""
        n = X.shape[0]
        mean = np.sum(self.Wm * X, axis=1, keepdims=True)
        X_diff = X - mean
        P = np.zeros((n, n))
        for i in range(2 * self.model.n + 1):
            P += self.Wc[i] * np.outer(X_diff[:, i], X_diff[:, i])
        P += noise_cov
        return mean, P

    # ==========================================================
    # Prediction and Update (override EKF steps)
    # ==========================================================
    def _predict(self, x, P, **kwargs):
        """Unscented prediction (no F, no B)."""
        u = kwargs.get("u", np.zeros((self.model.l, 1)))
        t = kwargs.get("t", None)

        sigmas = self._sigma_points(x, P)
        n_sig = sigmas.shape[1]
        X_pred = np.zeros_like(sigmas)

        for i in range(n_sig):
            X_pred[:, i] = self.model.f(sigmas[:, [i]], u=u, w=None, t=t).ravel()

        Q = self.model.Q(t)
        x_pred, P_pred = self._unscented_transform(X_pred, Q)
        return x_pred, P_pred, X_pred

    def _update(self, z, x_pred=None, P_pred=None, X_pred=None, t=None):
        """Unscented measurement update (no H)."""
        n = self.model.n
        m = self.model.m

        if x_pred is None or P_pred is None or X_pred is None:
            x_pred, P_pred, X_pred = self._predict(self.x, self.P, t=t)

        n_sig = X_pred.shape[1]
        Z_sigma = np.zeros((m, n_sig))
        for i in range(n_sig):
            Z_sigma[:, i] = self.model.h(X_pred[:, [i]], v=None, t=t).ravel()

        R = self.model.R(t)
        z_pred, Pz = self._unscented_transform(Z_sigma, R)

        # Cross covariance Pxz
        Pxz = np.zeros((n, m))
        for i in range(2 * n + 1):
            dx = (X_pred[:, [i]] - x_pred)
            dz = (Z_sigma[:, [i]] - z_pred)
            Pxz += self.Wc[i] * dx @ dz.T

        # Kalman gain and update
        K = Pxz @ np.linalg.pinv(Pz)
        x_upd = x_pred + K @ (z - z_pred)
        P_upd = P_pred - K @ Pz @ K.T
        return x_upd, P_upd
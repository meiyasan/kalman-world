import numpy as np

# @TODO: To be tested and documented

from . import KalmanStateModel
class KalmanUnscentedModel(KalmanStateModel):
    """
    Unscented Kalman Filter Model for nonlinear systems.
    Uses Unscented Transform to propagate uncertainty without linearization.
    """

    NAME = "Unscented Kalman Model"
    DESCRIPTION = "Nonlinear model using Unscented Transform for state and observation updates."
    VERSION = "1.0"

    def __init__(self, f, h, alpha=1e-3, beta=2, kappa=0, **kwargs):
        """
        f: callable, state transition function f(x, u)
        h: callable, observation function h(x)
        alpha: Spread of sigma points (typically small, ~1e-3)
        beta: Incorporates prior knowledge of distribution (2 is optimal for Gaussian)
        kappa: Secondary scaling parameter (usually 0)
        """
        self.f = f
        self.h = h
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        super().__init__(**kwargs)

        # Precompute scaling parameters
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)

        # Weights for sigma points
        self.Wm = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambda_))
        self.Wc = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambda_))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - self.alpha**2 + self.beta)

    def _generate_sigma_points(self, x, P):
        """
        Generates sigma points for UKF.
        """
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x
        sqrt_P = np.linalg.cholesky(P)

        for i in range(self.n):
            offset = self.gamma * sqrt_P[:, i]
            sigma_points[i + 1] = x + offset
            sigma_points[self.n + i + 1] = x - offset

        return sigma_points

    def _predict(self, x, P, u=None, **kwargs):
        """
        UKF prediction step.
        """
        sigma_points = self._generate_sigma_points(x, P)
        propagated_points = np.array([self.f(sp, u) for sp in sigma_points])

        # Predicted mean and covariance
        x_pred = np.sum(self.Wm[:, None] * propagated_points, axis=0)
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            y = propagated_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)

        return x_pred, P_pred

    def _update(self, z):
        """
        UKF update step with measurement z.
        """
        sigma_points = self._generate_sigma_points(self.x, self.P)
        predicted_measurements = np.array([self.h(sp) for sp in sigma_points])

        # Predicted observation mean
        z_pred = np.sum(self.Wm[:, None] * predicted_measurements, axis=0)

        # Innovation covariance
        S = self.R.copy()
        for i in range(2 * self.n + 1):
            dz = predicted_measurements[i] - z_pred
            S += self.Wc[i] * np.outer(dz, dz)

        # Cross-covariance
        Pxz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            dx = sigma_points[i] - self.x
            dz = predicted_measurements[i] - z_pred
            Pxz += self.Wc[i] * np.outer(dx, dz)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ S @ K.T

        return self.x, self.P

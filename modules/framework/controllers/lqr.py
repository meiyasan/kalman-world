"""LQRKalmanController — optimal linear-quadratic regulator via DARE."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class LQRKalmanController(KalmanController):
    """Linear Quadratic Regulator: ``u = −K · (x̂ − x_ref)``.

    The gain ``K`` is computed from the Discrete Algebraic Riccati Equation
    (scipy) — the analytically optimal solution for linear systems.  Call
    ``fit()`` before use.
    """

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.K = None

    def fit(self):
        from scipy.linalg import solve_discrete_are
        A = self.model.F()
        B = self.model.B()
        if B is None:
            raise ValueError("LQR requires a model with B matrix (l > 0).")
        P = solve_discrete_are(A, B, self.Qcost, self.Rcost)
        self.K = np.linalg.solve(self.Rcost + B.T @ P @ B, B.T @ P @ A)
        return self

    def _policy(self, x_hat, P, t, *, nu=None):
        if self.K is None:
            raise RuntimeError("Call fit() before using LQRKalmanController.")
        return -self.K @ (x_hat - self.x_ref)

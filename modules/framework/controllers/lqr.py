"""LQRKalmanController — optimal linear-quadratic regulator via DARE."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class LQRKalmanController(KalmanController):
    """Linear Quadratic Regulator: ``u = −K · (x̂ − x_ref)``.

    The gain ``K`` is computed from the Discrete Algebraic Riccati Equation —
    the analytically optimal solution for linear systems — for the model's
    ``F()`` and ``B()``.  Call ``fit()`` before use.

    With libKalman installed the equation is solved and the policy computed by
    ``Kalman::Controllers::LQR`` (``Kalman::Algebra::SolveDARE``, which agrees
    with scipy's ``solve_discrete_are`` to rounding); scipy is the reference.
    """

    NATIVE = "lqr"

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.K = None

    def fit(self):
        A = self.model.F()
        B = self.model.B()
        if B is None:
            raise ValueError("LQR requires a model with B matrix (l > 0).")
        handle = self._native_handle()
        n, L = int(self.model.n), max(int(self.model.l), 1)
        if (handle is not None and np.shape(A) == (n, n) and np.shape(B) == (n, L)
                and np.isrealobj(A) and np.isrealobj(B)):
            from ..native import core
            self.K = self._native_call(core.lqr_fit, handle, A, B, n, L)
            self._native_gain = self.K.tobytes()
            return self
        return self._reference_fit(A, B)

    def _reference_fit(self, A=None, B=None):
        from scipy.linalg import solve_discrete_are
        A = self.model.F() if A is None else A
        B = self.model.B() if B is None else B
        if B is None:
            raise ValueError("LQR requires a model with B matrix (l > 0).")
        P = solve_discrete_are(A, B, self.Qcost, self.Rcost)
        self.K = np.linalg.solve(self.Rcost + B.T @ P @ B, B.T @ P @ A)
        return self

    def _policy(self, x_hat, P, t, *, nu=None):
        if self.K is None:
            raise RuntimeError("Call fit() before using LQRKalmanController.")
        handle = self._native_handle()
        if handle is not None and self._gains_key(self.K) is not None:
            # The gain may have been fitted by scipy, or set by hand: the C++ one follows it.
            gain = np.asarray(self.K, dtype=np.float64).tobytes()
            if self.__dict__.get("_native_gain") != gain:
                from ..native import core
                core.lqr_set_gain(handle, self.K, int(self.model.n), max(int(self.model.l), 1))
                self._native_gain = gain
            u = self._native_policy(x_hat, P, t, nu)
            if u is not None:
                return u
        return self._reference_policy(x_hat, P, t, nu=nu)

    def _reference_policy(self, x_hat, P, t, *, nu=None):
        return -self.K @ (x_hat - self.x_ref)

    def _native_key(self):
        return self._cost_key()

    def _native_create(self, model_handle):
        from ..native import core
        self._native_gain = None                       # a new C++ LQR has no gain yet
        return core.create_controller("lqr", model_handle, int(self.model.n), max(int(self.model.l), 1),
                                      **self._cost_args())

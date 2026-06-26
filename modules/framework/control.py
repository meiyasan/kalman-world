"""KalmanController — base contract for closed-loop control over a Kalman filter."""
from __future__ import annotations
import numpy as np


class KalmanController:
    """Callable controller: ``u = ctrl(x̂, P, t, nu=ν)``.

    The KalmanFilter receives it via ``uk=`` and treats it like any other
    control sequence.  The controller may update ``model._B`` as a side
    effect before returning ``u``.

    Subclasses override ``_policy`` (required) and ``_update`` (optional).
    """

    def __init__(self, model, x_ref=None, Qcost=None, Rcost=None):
        self.model = model
        self.x_ref = (np.zeros((model.n, 1)) if x_ref is None
                      else np.asarray(x_ref, dtype=float).reshape(-1, 1))
        n, l = model.n, max(model.l, 1)
        self.Qcost = np.eye(n) if Qcost is None else np.asarray(Qcost, dtype=float)
        self.Rcost = np.eye(l) if Rcost is None else np.asarray(Rcost, dtype=float)
        self.reset()

    def __call__(self, x_hat, P, t, *, nu=None):
        x_hat = np.asarray(x_hat, dtype=float).reshape(-1, 1)
        u = np.asarray(self._policy(x_hat, P, t, nu=nu), dtype=float).reshape(-1, 1)

        B_new = self._update(x_hat, P, t, nu=nu)
        if B_new is not None:
            self.model._B = np.asarray(B_new, dtype=float)
            self.B_log.append((t, B_new.copy()))

        e = x_hat - self.x_ref
        cost = float((e.T @ self.Qcost @ e + u.T @ self.Rcost @ u).item())
        self.cost_log.append((t, cost))
        self.u_log.append((t, u.copy()))
        return u

    def _policy(self, x_hat, P, t, *, nu=None):
        raise NotImplementedError

    def _update(self, x_hat, P, t, *, nu=None):
        return None

    def reset(self):
        self.cost_log = []
        self.u_log = []
        self.B_log = []

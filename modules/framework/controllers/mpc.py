"""MPCKalmanController — model predictive control: a plan under limits, remade at every step."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class MPCKalmanController(KalmanController):
    """Plans ``horizon`` commands ahead at every step, honouring limits on the
    command while it plans, applies the first one, and plans again.

    The plan minimises ``Σ eᵀ Qcost e + uᵀ Rcost u`` over the horizon, with a
    terminal cost (the Riccati solution, so that with no limit in the way the
    command is the LQR command), subject to ``u_min ≤ u ≤ u_max`` and, when
    given, ``|u_k − u_{k−1}| ≤ du_max``. The prediction is the model's own ``f``:
    exact for a model of matrices, linearised along the previous plan otherwise
    (``n_iterations`` passes per step).

    Beside the others: LQR is one gain and knows no limit; MPPI replans by
    rolling random command sequences through ``f``. This one replans
    deterministically, and the limits are constraints of the plan.

    The planner is ``Kalman::Controllers::MPC`` (``include/Kalman/Controllers/MPC.h``);
    this class forwards to it and has no implementation of its own, so it needs
    libKalman.

    Parameters
    ----------
    horizon : steps planned ahead.
    u_min, u_max : limits on the command (a number, or one per command); ``None`` for none.
    du_max : limit on its change per step; ``None`` for none.
    n_iterations : linearisation passes per step (1 is enough once the plan is warm).
    """

    NATIVE = "mpc"

    def __init__(self, model, horizon=20, u_min=None, u_max=None, du_max=None, n_iterations=1, **kw):
        self.horizon = int(horizon)
        self.n_iterations = int(n_iterations)
        self._l = max(int(model.l), 1)
        self.u_min, self.u_max, self.du_max = (self._limit(u_min), self._limit(u_max), self._limit(du_max))
        super().__init__(model, **kw)

    def _limit(self, value):
        if value is None:
            return None
        limit = np.asarray(value, dtype=np.float64).reshape(-1)
        return np.full(self._l, float(limit[0])) if limit.size == 1 else limit

    def _policy(self, x_hat, P, t, *, nu=None):
        handle = self._native_handle()
        u = self._native_policy(x_hat, P, t, nu) if handle is not None else None
        if u is None:
            raise RuntimeError("MPCKalmanController needs libKalman (the planner is Kalman::Controllers::MPC), "
                               "a model in double precision, and limits with one entry per command.")
        return u

    def plan(self):
        """The commands planned at the last step, ``(horizon, l)``, first one first."""
        from ..native import core
        handle = self._native_handle()
        if handle is None:
            raise RuntimeError("MPCKalmanController needs libKalman.")
        return core.mpc_plan(handle, self.horizon, self._l)

    def _native_key(self):
        costs = self._cost_key()
        limits = (self.u_min, self.u_max, self.du_max)
        if costs is None or self.horizon <= 0 or any(v is not None and v.size != self._l for v in limits):
            return None
        return (self.horizon, self.n_iterations) + tuple(None if v is None else v.tobytes() for v in limits) + costs

    def _native_create(self, model_handle):
        from ..native import core
        return core.create_controller("mpc", model_handle, int(self.model.n), self._l,
                                      horizon=self.horizon, u_min=self.u_min, u_max=self.u_max,
                                      du_max=self.du_max, n_iterations=self.n_iterations, **self._cost_args())

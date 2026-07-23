"""MPPIKalmanController — online receding-horizon Model Predictive Path Integral control."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class MPPIKalmanController(KalmanController):
    """Direct (online) MPPI: replans a sampled receding-horizon control sequence at *every* step.

    Unlike :class:`LQRKalmanController` — a single fixed gain computed once from one
    linearisation of ``model.f`` — MPPI resolves
    ``u* = argmin E[ sum_k eᵀ Qcost e + uᵀ Rcost u ]`` from scratch at every control tick by
    Monte-Carlo rollout of the **true nonlinear** ``model.f()``, including whatever actuator
    limits the model enforces internally (e.g. ``CarLaneModel``'s ``steer_limit`` clip).

    That is the structural advantage over LQR under hard constraints: LQR computes its gain
    once, offline, against the *unconstrained* linear-quadratic optimum, and the clip is
    applied after the fact, blind to the fact that saturation is coming. MPPI's rollouts hit
    the same clip during planning, so the chosen action already accounts for it — no
    look-ahead is wasted commanding steering the actuator can never deliver.

    The price is cost: MPPI resolves an ``n_rollouts x horizon`` Monte-Carlo optimisation on
    **every** call — orders of magnitude more expensive per step than reading a stored gain
    matrix, and generally too slow for a tight real-time loop. That gap is exactly what
    :class:`NeuralKalmanController` is for: train an MLP offline to imitate MPPI's own
    decisions (``fit_mppi()``), then deploy the single cheap forward pass instead of
    resolving MPPI online at every tick.

    Warm-started receding horizon: the nominal sequence found at step *k* is shifted by one
    and reused as the starting point at step *k+1* (with the vacated last slot zeroed), so
    each step only has to refine an already-good guess rather than search from scratch.
    """

    def __init__(self, model, n_rollouts=200, horizon=30, noise_sigma=0.2, lam=1.0,
                 n_iters=3, seed=None, **kw):
        # Set before super().__init__() since that calls reset(), which needs these.
        self.n_rollouts = int(n_rollouts)
        self.horizon = int(horizon)
        self.noise_sigma = float(noise_sigma)
        self.lam = float(lam)
        self.n_iters = int(n_iters)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._l = max(model.l, 1)
        self._u_nom = np.zeros((self.horizon, self._l))
        super().__init__(model, **kw)

    def _policy(self, x_hat, P, t, *, nu=None):
        model = self.model
        dt = 1.0 / model.fs if model.fs else 1.0
        u_nom = self._u_nom

        for _ in range(self.n_iters):
            du = self._rng.normal(0, self.noise_sigma, (self.n_rollouts, self.horizon, self._l))
            u_try = u_nom[None, :, :] + du
            costs = np.zeros(self.n_rollouts)
            for i in range(self.n_rollouts):
                xi = x_hat.copy()
                cost = 0.0
                for k in range(self.horizon):
                    u_k = u_try[i, k].reshape(-1, 1)
                    xi = np.asarray(model.f(xi, u=u_k, t=t + k * dt), dtype=float).reshape(-1, 1)
                    e = xi - self.x_ref
                    cost += float((e.T @ self.Qcost @ e + u_k.T @ self.Rcost @ u_k).item())
                costs[i] = cost
            weights = np.exp(-(costs - costs.min()) / self.lam)
            weights /= weights.sum() + 1e-12
            u_nom = np.einsum("i,ihl->hl", weights, u_try)

        u0 = u_nom[0].reshape(-1, 1)
        # shift the warm-started nominal sequence for the next call's receding horizon
        self._u_nom = np.vstack([u_nom[1:], np.zeros((1, self._l))])
        return u0

    def reset(self):
        super().reset()
        self._u_nom = np.zeros((self.horizon, self._l))
        self._rng = np.random.default_rng(self._seed)

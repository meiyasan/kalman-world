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

    With libKalman installed the rollouts run in ``Kalman::Controllers::MPPI``, which then
    holds the warm-started sequence: ``model.f`` and ``model.clip_u`` are called back from
    C++ for a model written in Python (and not at all for a model written in C++). The
    perturbations are still drawn here, from the same seeded numpy generator in the same
    order, so a seeded controller makes the same decisions on either path.
    :meth:`_reference_policy` is the Python reference.
    """

    NATIVE = "mppi"

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
        handle = self._native_handle()
        if handle is not None and self._native_belief(x_hat, P, nu) is not None:
            from ..native import core
            # The draws of the reference, one (n_rollouts, horizon, l) block per iteration.
            draws = [self._rng.normal(0, self.noise_sigma, (self.n_rollouts, self.horizon, self._l))
                     for _ in range(self.n_iters)]
            core.mppi_set_draws(handle, np.concatenate([d.reshape(-1) for d in draws]) if draws else np.empty(0))
            return self._native_policy(x_hat, P, t, nu)
        return self._reference_policy(x_hat, P, t, nu=nu)

    def _reference_policy(self, x_hat, P, t, *, nu=None):
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
                    # Clip the *candidate* action before rolling it out or costing it, not
                    # after -- otherwise dynamics/cost go flat past the limit (any raw value
                    # out there looks identical), and the cost-weighted average of many
                    # different "however far past the limit" samples can land anywhere out
                    # there with no pull back toward what's deliverable. Clipping here keeps
                    # every candidate, and so the weighted-average nominal, inside the box
                    # the actuator can actually reach -- matching real MPC practice, and
                    # keeping u_try consistent with what u_nom will be averaged from.
                    u_k = np.asarray(model.clip_u(u_try[i, k].reshape(-1, 1)), dtype=float).reshape(-1)
                    u_try[i, k] = u_k
                    u_k = u_k.reshape(-1, 1)
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

    def _native_key(self):
        if self.n_rollouts <= 0 or self.horizon <= 0 or self.n_iters < 0 or not self.lam > 0:
            return None
        costs = self._cost_key()
        if costs is None or self._l != max(int(self.model.l), 1):
            return None
        return (self.n_rollouts, self.horizon, self.noise_sigma, self.lam, self.n_iters) + costs

    def _native_create(self, model_handle):
        from ..native import core
        # The C++ generator is never used (the draws come from self._rng); the seed is a formality.
        return core.create_controller("mppi", model_handle, int(self.model.n), self._l,
                                      n_rollouts=self.n_rollouts, horizon=self.horizon,
                                      noise_sigma=self.noise_sigma, lam=self.lam, n_iters=self.n_iters,
                                      seed=0, **self._cost_args())

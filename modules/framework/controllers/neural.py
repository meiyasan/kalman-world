"""NeuralKalmanController — MLP policy trained via MPPI rollouts of model.f()."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class NeuralKalmanController(KalmanController):
    """Neural controller trained offline to imitate :class:`MPPIKalmanController`.

    ``fit_mppi()`` runs the same warm-started, receding-horizon MPPI planner online at a
    grid of simulated states, records every ``(state, u*)`` decision, and fits a small MLP
    to reproduce it — turning an expensive online Monte-Carlo optimisation into a single
    cheap forward pass at deployment time.

    Parameters
    ----------
    features : tuple of str
        Which inputs the MLP sees.  Any subset of ``("error", "P", "nu")``:

        * ``"error"`` — state error  ``x̂ − x_ref``  (shape n)
        * ``"P"``     — navigation uncertainty  ``√diag(P)``  (shape n)
        * ``"nu"``    — latest innovation  ``z − H x̂⁻``  (shape m)

        Default is ``("error",)``.  MPPI's own decision ``u*`` never depends on ``P`` or
        ``ν`` — it plans from a state, not a belief — so for a *full-state-feedback* MPPI
        teacher (the common case) those two channels carry no signal, only noise, unless the
        training rollouts are given genuinely representative ``P``/``ν`` statistics to match
        what the controller will see at deployment (``fit_mppi`` does this — see below).
        Feeding the network a channel that is constant or always zero during training but
        varies at test time is a real train/test mismatch, not a free extra feature: adding
        ``"P"`` or ``"nu"`` without matched statistics measurably *hurts* closed-loop cost.
        Only opt in if your controller's optimal action genuinely depends on uncertainty or
        innovation (e.g. an uncertainty-aware or predictive policy you designed on purpose).
    """

    def __init__(self, model, hidden=64, features=("error",), **kw):
        super().__init__(model, **kw)
        self._hidden = hidden
        self._features = tuple(features)
        n, m = model.n, model.m
        self._input_dim = (
            (n if "error" in self._features else 0) +
            (n if "P"     in self._features else 0) +
            (m if "nu"    in self._features else 0)
        )
        self._x_mean = np.zeros(self._input_dim)
        self._x_std  = np.ones(self._input_dim)

    # ------------------------------------------------------------------
    def _build_input(self, x_hat, P, nu):
        parts = []
        if "error" in self._features:
            parts.append((x_hat - self.x_ref).ravel())
        if "P" in self._features:
            parts.append(np.sqrt(np.abs(np.diag(P))))
        if "nu" in self._features:
            parts.append(np.zeros(self.model.m) if nu is None
                         else np.asarray(nu).ravel())
        return np.concatenate(parts) if parts else np.zeros(1)

    def _policy(self, x_hat, P, t, *, nu=None):
        if self._weights is None:
            raise RuntimeError("Call fit_mppi() or load() before using NeuralKalmanController.")
        inp = self._build_input(x_hat, P, nu)
        return self._mlp_forward((inp - self._x_mean) / self._x_std)

    def _mlp_forward(self, x):
        W1, b1 = self._weights["W1"], self._weights["b1"]
        W2, b2 = self._weights["W2"], self._weights["b2"]
        W3, b3 = self._weights["W3"], self._weights["b3"]
        h = np.tanh(W1 @ x + b1)
        h = np.tanh(W2 @ h + b2)
        return W3 @ h + b3

    def _init_weights(self, seed=None):
        rng = np.random.default_rng(seed)
        d_in, d_h, d_out = self._input_dim, self._hidden, max(self.model.l, 1)
        scale1 = np.sqrt(2.0 / d_in)
        scale2 = np.sqrt(2.0 / d_h)
        self._weights = {
            "W1": rng.normal(0, scale1, (d_h, d_in)),
            "b1": np.zeros(d_h),
            "W2": rng.normal(0, scale2, (d_h, d_h)),
            "b2": np.zeros(d_h),
            "W3": rng.normal(0, scale2, (d_out, d_h)),
            "b3": np.zeros(d_out),
        }

    def fit_mppi(self, n_rollouts=100, horizon=15, noise_sigma=0.1, lam=1.0,
                 epochs=200, lr=1e-3, n_episodes=8, rollout_steps=30, x0_std=1.0, seed=None,
                 n_iters=3):
        """Train by **imitating** a warm-started, receding-horizon MPPI planner on ``model.f()``.

        This runs the same planner as :class:`MPPIKalmanController` — see its docstring for
        why warm-starting matters — as the *teacher*, and distils its decisions into the MLP:

        For every step of ``n_episodes`` simulated episodes:

        1. Starting from the nominal sequence carried over from the previous step (zero on
           the first step of an episode), refine it over ``n_iters`` rounds: sample
           ``n_rollouts`` perturbations around the nominal, roll each through
           ``model.f(x, u+δu, t)`` (full nonlinear physics, so any actuator clip the model
           enforces internally is respected during planning), weight by
           ``w_i = exp(−J_i / λ)`` on cost ``J = Σ eᵀ Qcost e + uᵀ Rcost u``, and replace the
           nominal with the weighted average.
        2. Record ``(state, u*_0)`` — the *first* action of the refined nominal sequence.
        3. Apply that action, advance the true state, shift the nominal by one step
           (receding horizon), and repeat.

        Refining a warm-started nominal (rather than resampling raw noise from zero every
        step, as one-shot random shooting does) is what lets MPPI actually converge toward
        its cost-optimal sequence within a practical rollout budget — without it, the
        collected ``u*`` targets are far noisier and the imitating MLP inherits that noise
        as a training-time bias.

        Replanning from the actual current state at every step — rather than committing to
        one open-loop sequence per episode — is what makes the collected dataset genuinely
        state-conditioned, so the trained MLP generalises as a feedback policy instead of
        just memorising a single nominal trajectory.

        Episodes start near ``model.x0`` (the system's typical operating point — not
        ``x_ref``, which may sit at an unreachable corner of state space, e.g. zero speed)
        with spread ``x0_std``.
        """
        from .mppi import MPPIKalmanController

        rng = np.random.default_rng(seed)
        model = self.model
        n, l, m = model.n, max(model.l, 1), model.m
        dt = 1.0 / model.fs if model.fs else 1.0

        # Realistic P(t)/S(t) schedule for the "P"/"nu" features, if requested — the
        # covariance recursion (unlike the state trajectory) doesn't depend on which control
        # is applied, so it can be precomputed once from the model's own F/H/Q/R rather than
        # left as the constant placeholder that would silently mismatch what a real filter
        # produces at deployment.
        need_P = "P" in self._features or "nu" in self._features
        if need_P:
            F0, H0, Q0, R0 = model.F(), model.H(), model.Q(), model.R()
            P0 = model.P0.copy() if hasattr(model, "P0") else np.eye(n)
            P_sched, S_sched = [], []
            Pc = P0
            for _ in range(rollout_steps + 1):
                Pm = F0 @ Pc @ F0.T + Q0
                S = H0 @ Pm @ H0.T + R0
                K = Pm @ H0.T @ np.linalg.inv(S)
                Pc = (np.eye(n) - K @ H0) @ Pm
                P_sched.append(Pm)
                S_sched.append(S)
        else:
            P_dummy = np.eye(n)

        planner = MPPIKalmanController(
            model, n_rollouts=n_rollouts, horizon=horizon, noise_sigma=noise_sigma,
            lam=lam, n_iters=n_iters, seed=seed, Qcost=self.Qcost, Rcost=self.Rcost,
            x_ref=self.x_ref,
        )

        dataset_x, dataset_u = [], []

        for _ in range(n_episodes):
            x = (model.x0.ravel() + rng.normal(0, x0_std, n)).reshape(-1, 1)
            planner.reset()

            for step in range(rollout_steps):
                t0 = step * dt
                u0 = planner._policy(x, None, t0)

                if need_P:
                    P_here, S_here = P_sched[step], S_sched[step]
                    nu_here = (rng.multivariate_normal(np.zeros(m), S_here).reshape(-1, 1)
                               if "nu" in self._features else None)
                else:
                    P_here, nu_here = P_dummy, None

                inp = self._build_input(x, P_here, nu_here)
                dataset_x.append(inp)
                dataset_u.append(u0.ravel())

                x = np.asarray(model.f(x, u=u0, t=t0), dtype=float).reshape(-1, 1)

        X = np.array(dataset_x)
        Y = np.array(dataset_u)

        # Raw input features span wildly different scales (e.g. a lateral offset ~O(1)
        # next to a cruise speed ~O(15)) — normalising is what lets gradient descent see
        # the small-scale features at all, instead of the largest feature dominating.
        self._x_mean = X.mean(axis=0)
        self._x_std  = X.std(axis=0)
        self._x_std[self._x_std < 1e-8] = 1.0
        Xn = (X - self._x_mean) / self._x_std

        self._init_weights(seed=seed)
        self._train_mlp(Xn, Y, epochs=epochs, lr=lr)
        return self

    def _train_mlp(self, X, Y, epochs=100, lr=1e-3):
        """Adam-optimised gradient descent — plain SGD converges too unevenly across
        parameters of very different gradient scale (first layer vs output layer)."""
        params = self._weights
        m_t = {k: np.zeros_like(v) for k, v in params.items()}
        v_t = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        N = len(X)
        for step in range(1, epochs + 1):
            W1, b1 = params["W1"], params["b1"]
            W2, b2 = params["W2"], params["b2"]
            W3, b3 = params["W3"], params["b3"]

            h1 = np.tanh(X @ W1.T + b1)
            h2 = np.tanh(h1 @ W2.T + b2)
            y_pred = h2 @ W3.T + b3

            err = y_pred - Y

            grads = {}
            grads["W3"] = (2.0 / N) * err.T @ h2
            grads["b3"] = (2.0 / N) * err.sum(axis=0)
            dh2 = (2.0 / N) * err @ W3 * (1 - h2 ** 2)
            grads["W2"] = dh2.T @ h1
            grads["b2"] = dh2.sum(axis=0)
            dh1 = dh2 @ W2 * (1 - h1 ** 2)
            grads["W1"] = dh1.T @ X
            grads["b1"] = dh1.sum(axis=0)

            for k in params:
                m_t[k] = beta1 * m_t[k] + (1 - beta1) * grads[k]
                v_t[k] = beta2 * v_t[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m_t[k] / (1 - beta1 ** step)
                v_hat = v_t[k] / (1 - beta2 ** step)
                params[k] = params[k] - lr * m_hat / (np.sqrt(v_hat) + eps)

        self._weights = params

    def save(self, path):
        np.savez(path, x_mean=self._x_mean, x_std=self._x_std, **self._weights)

    def load(self, path):
        data = dict(np.load(path))
        self._x_mean = data.pop("x_mean")
        self._x_std  = data.pop("x_std")
        self._weights = data
        return self

"""NeuralKalmanController — MLP policy trained via MPPI rollouts of model.f()."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


class NeuralKalmanController(KalmanController):
    """Neural controller trained offline via MPPI on ``model.f()``.

    The MLP sees ``[x̂ − x_ref, √diag(P), ν]`` — never F, H, or raw z.
    """

    def __init__(self, model, hidden=64, **kw):
        super().__init__(model, **kw)
        self._hidden = hidden
        self._weights = None
        self._input_dim = model.n + model.n + model.m
        self._x_mean = np.zeros(self._input_dim)
        self._x_std = np.ones(self._input_dim)

    def _policy(self, x_hat, P, t, *, nu=None):
        if self._weights is None:
            raise RuntimeError("Call fit_mppi() or load() before using NeuralKalmanController.")
        inp = np.concatenate([
            (x_hat - self.x_ref).ravel(),
            np.sqrt(np.abs(np.diag(P))),
            np.zeros(self.model.m) if nu is None else np.asarray(nu).ravel(),
        ])
        return self._mlp_forward((inp - self._x_mean) / self._x_std)

    def _mlp_forward(self, x):
        W1, b1 = self._weights["W1"], self._weights["b1"]
        W2, b2 = self._weights["W2"], self._weights["b2"]
        W3, b3 = self._weights["W3"], self._weights["b3"]
        h = np.tanh(W1 @ x + b1)
        h = np.tanh(W2 @ h + b2)
        return W3 @ h + b3

    def _init_weights(self):
        rng = np.random.default_rng()
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
                 epochs=200, lr=1e-3, n_episodes=8, rollout_steps=30, x0_std=1.0, seed=None):
        """Train via **receding-horizon** Model Predictive Path Integral on ``model.f()``.

        At every step of ``n_episodes`` simulated episodes:

        1. Sample ``n_rollouts`` random control sequences of length ``horizon``
        2. Roll each through ``model.f(x, u+δu, t)`` from the *current* state (full
           nonlinear physics), accumulating cost ``J = Σ eᵀ Qcost e + uᵀ Rcost u``
        3. Weight ``w_i = exp(−J_i / λ)``, normalised; average to a control sequence
        4. Apply only the **first** action (receding horizon, replanned every step) and
           record the ``(state, u)`` pair
        5. Advance the true state by that one action and repeat

        Replanning from the actual current state at every step — rather than committing
        to one open-loop sequence per episode — is what makes the collected dataset
        genuinely state-conditioned, so the trained MLP generalises as a feedback policy
        instead of just memorising a single nominal trajectory.

        Episodes start near ``model.x0`` (the system's typical operating point — not
        ``x_ref``, which may sit at an unreachable corner of state space, e.g. zero
        speed) with spread ``x0_std``.
        """
        rng = np.random.default_rng(seed)
        model = self.model
        n, l, m = model.n, max(model.l, 1), model.m
        dt = 1.0 / model.fs if model.fs else 1.0

        dataset_x, dataset_u = [], []
        P_dummy = np.eye(n)

        for _ in range(n_episodes):
            x = (model.x0.ravel() + rng.normal(0, x0_std, n)).reshape(-1, 1)

            for step in range(rollout_steps):
                t0 = step * dt
                du = rng.normal(0, noise_sigma, (n_rollouts, horizon, l))
                costs = np.zeros(n_rollouts)

                for i in range(n_rollouts):
                    xi = x.copy()
                    cost = 0.0
                    for k in range(horizon):
                        u_k = du[i, k].reshape(-1, 1)
                        xi = np.asarray(model.f(xi, u=u_k, t=t0 + k * dt), dtype=float).reshape(-1, 1)
                        e = xi - self.x_ref
                        cost += float((e.T @ self.Qcost @ e + u_k.T @ self.Rcost @ u_k).item())
                    costs[i] = cost

                weights = np.exp(-(costs - costs.min()) / lam)
                weights /= weights.sum() + 1e-12
                u0 = np.einsum("i,il->l", weights, du[:, 0, :]).reshape(-1, 1)

                inp = np.concatenate([
                    (x - self.x_ref).ravel(),
                    np.sqrt(np.abs(np.diag(P_dummy))),
                    np.zeros(m),
                ])
                dataset_x.append(inp)
                dataset_u.append(u0.ravel())

                x = np.asarray(model.f(x, u=u0, t=t0), dtype=float).reshape(-1, 1)

        X = np.array(dataset_x)
        Y = np.array(dataset_u)

        # Raw input features span wildly different scales (e.g. a lateral offset ~O(1)
        # next to a cruise speed ~O(15)) — normalising is what lets gradient descent see
        # the small-scale features at all, instead of the largest feature dominating.
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std < 1e-8] = 1.0
        Xn = (X - self._x_mean) / self._x_std

        self._init_weights()
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
        self._x_std = data.pop("x_std")
        self._weights = data
        return self

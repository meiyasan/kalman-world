"""MLP — pure-numpy 3-layer tanh MLP for use with NeuralRouter.

Usage::

    from kalman.filters.routers import NeuralRouter
    from kalman.filters.routers.nn import MLP

    router = NeuralRouter(model=MLP(hidden=32), window=5)
    ikf.fit(training_zk_list, labels)

Layer layout
------------
+-------+-----+---------------------+------------------------------+
| Layer | Key | Shape               | Role                         |
+=======+=====+=====================+==============================+
|     1 | W1  | (hidden, M×window)  | NIS window → hidden₁  (tanh) |
+-------+-----+---------------------+------------------------------+
|     2 | W2  | (hidden, hidden)    | hidden₁ → hidden₂  (tanh)   |
+-------+-----+---------------------+------------------------------+
|     3 | W3  | (M, hidden)         | hidden₂ → M logits (softmax) |
+-------+-----+---------------------+------------------------------+

He initialisation (σ = √(2/fan_in)) for tanh activations.
Adam optimiser (β₁=0.9, β₂=0.999, ε=1e-8) for cross-entropy training.
"""
from __future__ import annotations
import numpy as np


class MLP:
    """Pure-numpy 3-layer tanh MLP, callable as ``probs = mlp(features)``.

    Parameters
    ----------
    hidden : int
        Width of both hidden layers.
    """

    def __init__(self, hidden: int = 16):
        self.hidden   = int(hidden)
        self._weights: dict | None = None
        self._m_adam:  dict | None = None
        self._v_adam:  dict | None = None

    def build(self, d_in: int, d_out: int, seed=None) -> "MLP":
        """Initialise parameters. Called by NeuralRouter.fit().

        Parameters
        ----------
        d_in : int  — M × window (NIS window feature dimension)
        d_out : int — M (number of regimes / output classes)
        seed : int, optional
        """
        rng = np.random.default_rng(seed)
        d_h = self.hidden
        s1  = np.sqrt(2.0 / d_in)
        s2  = np.sqrt(2.0 / d_h)
        self._weights = {
            "W1": rng.normal(0, s1, (d_h, d_in)),
            "b1": np.zeros(d_h),
            "W2": rng.normal(0, s2, (d_h, d_h)),
            "b2": np.zeros(d_h),
            "W3": rng.normal(0, s2, (d_out, d_h)),
            "b3": np.zeros(d_out),
        }
        self._m_adam = {k: np.zeros_like(v) for k, v in self._weights.items()}
        self._v_adam = {k: np.zeros_like(v) for k, v in self._weights.items()}
        return self

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 500, lr: float = 2e-3, **_) -> "MLP":
        """Train with batch cross-entropy and Adam.

        Parameters
        ----------
        X : ndarray (N, d_in) — normalised feature matrix
        y : int ndarray (N,)  — class labels 0 … M−1
        """
        if self._weights is None:
            raise RuntimeError("Call build() before fit().")
        p  = self._weights
        mt = self._m_adam
        vt = self._v_adam
        N, M = len(X), p["b3"].shape[0]
        b1, b2, eps = 0.9, 0.999, 1e-8
        Y = np.zeros((N, M)); Y[np.arange(N), y] = 1.0

        for step in range(1, epochs + 1):
            h1 = np.tanh(X  @ p["W1"].T + p["b1"])
            h2 = np.tanh(h1 @ p["W2"].T + p["b2"])
            lg = h2 @ p["W3"].T + p["b3"]
            e  = np.exp(lg - lg.max(1, keepdims=True))
            pr = e / e.sum(1, keepdims=True)

            dl = (pr - Y) / N
            gr: dict = {"W3": dl.T @ h2,  "b3": dl.sum(0)}
            dh2 = dl @ p["W3"] * (1 - h2 ** 2)
            gr["W2"] = dh2.T @ h1; gr["b2"] = dh2.sum(0)
            dh1 = dh2 @ p["W2"] * (1 - h1 ** 2)
            gr["W1"] = dh1.T @ X;  gr["b1"] = dh1.sum(0)

            c1 = 1 - b1 ** step
            c2 = 1 - b2 ** step
            for k in p:
                mt[k] = b1 * mt[k] + (1 - b1) * gr[k]
                vt[k] = b2 * vt[k] + (1 - b2) * gr[k] ** 2
                p[k] -= lr * (mt[k] / c1) / (np.sqrt(vt[k] / c2) + eps)
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (d_in,) → probs: (M,) probability vector."""
        if self._weights is None:
            raise RuntimeError("MLP not built — call build() first.")
        p  = self._weights
        h1 = np.tanh(p["W1"] @ x + p["b1"])
        h2 = np.tanh(p["W2"] @ h1 + p["b2"])
        lg = p["W3"] @ h2 + p["b3"]
        e  = np.exp(lg - lg.max())
        return e / e.sum()

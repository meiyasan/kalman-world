"""KalmanRouter — base contract for IMM model routing.

A router maps the per-step IMM quantities to updated mode probabilities::

    mu = router(nis_arr, c, likelihoods)

is equivalent to::

    mu = router.route(nis_arr, c, likelihoods)

Arguments
---------
nis_arr : ndarray (k+1, M)
    NIS history up to the current step.

    NIS_j = ν_j^T S_j^{-1} ν_j

    where ν_j = z − h(x̂_j^−) (innovation) and S_j = H P_j^− H^T + R
    (predicted innovation covariance).  Under the correct model,
    NIS_j ~ χ²(m) with E[NIS_j] = m.  Under a wrong model NIS_j >> m.

c : ndarray (M,)
    Predicted mode probabilities from the Markov transition:
    c_j = Σ_i Π[i,j] μ_i.

likelihoods : ndarray (M,)
    Gaussian innovation likelihood L_j = N(ν_j; 0, S_j) per model.

Returns
-------
mu : ndarray (M,)
    Updated mode probabilities.  Non-negative and summing to 1.

Anatomy of one IMM step::

    ┌─────────────────────────────────────────────────────┐
    │ 1. Markov prediction:  c_j = Σ_i Π[i,j] μ_i        │
    │ 2. State mixing                                      │
    │ 3. Predict + update per model → ν_j, S_j, NIS_j, L_j│
    │ 4. Router:  μ = router.route(nis_arr, c, likelihoods)│  ← pluggable
    │ 5. Combined estimate: x̂ = Σ μ_j x̂_j               │
    └─────────────────────────────────────────────────────┘

The filter passes these three quantities and never sees inside the router.
Any callable ``(nis_arr, c, likelihoods) → mu`` also works directly.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class KalmanRouter(ABC):
    """Abstract base for IMM model routers.

    Subclass contract:

    * ``route(nis_arr, c, likelihoods) → mu``  — required
    * ``fit(nis_arrays, labels, **kw)``         — default no-op

    ``__call__`` delegates to ``route`` so any KalmanRouter is callable.

    Example::

        class MyRouter(KalmanRouter):
            def route(self, nis_arr, c, likelihoods):
                return self._normalise(c * likelihoods)
    """

    @abstractmethod
    def route(self,
              nis_arr: np.ndarray,
              c: np.ndarray,
              likelihoods: np.ndarray) -> np.ndarray:
        """Return updated mode probabilities.

        Parameters
        ----------
        nis_arr : ndarray (k+1, M)
        c : ndarray (M,) — Markov predicted mode probabilities
        likelihoods : ndarray (M,) — Gaussian L_j per model

        Returns
        -------
        mu : ndarray (M,), non-negative, sums to 1
        """

    def fit(self,
            nis_arrays: list[np.ndarray],
            labels_arrays: list[np.ndarray],
            **kwargs) -> "KalmanRouter":
        """Train the router on labelled NIS histories.

        Called by ``InteractiveKalmanFilter.fit()``.
        Default is a no-op (stateless routers need no training).

        Parameters
        ----------
        nis_arrays : list of ndarray (T_i, M)
            NIS history per training scenario (Bayesian-IMM run).
        labels_arrays : list of int ndarray (T_i,)
            True regime index 0 … M−1 at every step.
        """
        return self

    def __call__(self, nis_arr, c, likelihoods) -> np.ndarray:
        return self.route(nis_arr, c, likelihoods)

    @staticmethod
    def _normalise(mu: np.ndarray) -> np.ndarray:
        """Clip to (ε, ∞) and normalise to a probability vector."""
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
        return mu / mu.sum()

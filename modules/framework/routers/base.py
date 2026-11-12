"""KalmanRouter — base contract for IKF model routing.

A router maps the per-step IKF quantities to updated mode probabilities::

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

Anatomy of one IKF step::

    ┌─────────────────────────────────────────────────────┐
    │ 1. Markov prediction:  c_j = Σ_i Π[i,j] μ_i        │
    │ 2. State mixing                                      │
    │ 3. Predict + update per model → ν_j, S_j, NIS_j, L_j│
    │ 4. Router:  μ = router.route(nis_arr, c, likelihoods)│  ← pluggable
    │ 5. Combined estimate: x̂ = Σ μ_j x̂_j               │
    └─────────────────────────────────────────────────────┘

The filter passes these three quantities and never sees inside the router.
Any callable ``(nis_arr, c, likelihoods) → mu`` also works directly.

The rule-based routers (Bayesian, Threshold, Hybrid) decide in C++
(``Kalman::Routers``) when libKalman is installed; their ``_reference_route`` is
the Python reference, which runs inside ``native.python_reference()`` and
without the library.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class KalmanRouter(ABC):
    """Abstract base for IKF model routers.

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
            NIS history per training scenario (Bayesian-IKF run).
        labels_arrays : list of int ndarray (T_i,)
            True regime index 0 … M−1 at every step.
        """
        return self

    def __call__(self, nis_arr, c, likelihoods) -> np.ndarray:
        return self.route(nis_arr, c, likelihoods)

    # ------------------------------------------------------------------
    # The C++ router behind the class
    # ------------------------------------------------------------------
    #: The kind of C++ router this class forwards to (core.create_router), or
    #: ``None`` for a router that decides in Python.
    NATIVE: str | None = None

    def __getstate__(self):
        # The C++ handle does not pickle or copy; a copy builds its own on first use.
        state = self.__dict__.copy()
        state.pop("_native_state", None)
        return state

    def _native_owner(self):
        """The class whose ``NATIVE`` this instance has."""
        for cls in type(self).__mro__:
            if "NATIVE" in cls.__dict__:
                return cls
        return KalmanRouter

    def _native_key(self):
        """What the C++ router was built from; a change rebuilds it. ``None`` when
        this configuration has no C++ counterpart."""
        return ()

    def _native_create(self):
        """``(handle, keepalive)`` of the C++ router."""
        raise NotImplementedError

    def _native_rows(self):
        """How many of the latest NIS rows the C++ router reads (``None``: all)."""
        return None

    def _native_router(self):
        """``(handle, errors)`` of this router's C++ counterpart, or ``None``: no
        ``NATIVE`` kind, ``route`` overridden by a subclass, the reference path,
        no library, or a configuration the C++ class does not take."""
        owner = self._native_owner()
        if self.NATIVE is None or type(self).route is not owner.route:
            return None
        from ..models import native
        if not native.native_loop_enabled():
            return None
        key = self._native_key()
        if key is None:
            return None
        state = self.__dict__.get("_native_state")
        if state is None or state[0] != key:
            handle, keep, errors = self._native_create()
            self._native_state = state = (key, handle, keep, errors)
        return state[1], state[3]

    def _native_route(self, nis_arr, c, likelihoods):
        """The C++ router's ``mu``, or ``None`` when the reference must run."""
        c = np.asarray(c)
        likelihoods = np.asarray(likelihoods)
        nis_arr = np.asarray(nis_arr)
        if any(a.dtype.kind not in "fiub" for a in (c, likelihoods, nis_arr)):
            return None
        M = c.size
        if c.ndim != 1 or likelihoods.shape != (M,) or M == 0 or nis_arr.ndim != 2 or nis_arr.shape[1] != M:
            return None
        found = self._native_router()
        if found is None:
            return None
        rows = self._native_rows()
        if rows is not None:
            nis_arr = nis_arr[nis_arr.shape[0] - min(rows, nis_arr.shape[0]):]
        from ..native import core
        handle, errors = found
        errors.clear()
        try:
            return core.route(handle, nis_arr, c, likelihoods)
        except RuntimeError:
            if errors:
                raise errors[0]
            raise

    @staticmethod
    def _normalise(mu: np.ndarray) -> np.ndarray:
        """Clip to (ε, ∞) and normalise to a probability vector."""
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
        return mu / mu.sum()

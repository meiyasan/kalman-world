"""KalmanRouter — base contract for the decision layer of a routed model (``route``).

A router maps the per-step quantities of the interacting engine to updated mode probabilities::

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

Anatomy of one step::

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
    """Abstract base for model routers.

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

        Called by ``Router.fit()`` (the model made by ``route``).
        Default is a no-op (stateless routers need no training).

        Parameters
        ----------
        nis_arrays : list of ndarray (T_i, M)
            NIS history per training scenario (from a run with the Bayesian decision).
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

    #: Whether route() may use a C++ call prepared once per shape (the buffers and the
    #: C pointers into them made once). A router whose decision depends on other
    #: routers (Hybrid) says what in :meth:`_route_token`.
    PREPARED_ROUTE: bool = False

    def __setattr__(self, name, value):
        # A change to the router drops its prepared call, and counts as a new version for
        # the routers that use this one (a Hybrid's prepared call checks it).
        d = self.__dict__
        d.pop("_route_call", None)
        if name != "_route_call":
            d["_version"] = d.get("_version", 0) + 1
        object.__setattr__(self, name, value)

    def _route_token(self):
        """What a prepared call depends on besides this router's own settings, checked at
        every call; ``None`` for nothing."""
        return None

    def __getstate__(self):
        # The C++ handle does not pickle or copy; a copy builds its own on first use.
        state = self.__dict__.copy()
        state.pop("_native_state", None)
        state.pop("_route_call", None)
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
        prepared = self.__dict__.get("_route_call")
        if prepared is not None:
            mu = self._prepared_route(prepared, nis_arr, c, likelihoods)
            if mu is not None:
                return mu
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
            mu = core.route(handle, nis_arr, c, likelihoods)
        except RuntimeError:
            if errors:
                raise errors[0]
            raise
        if self.PREPARED_ROUTE and rows is not None:
            self._route_call = self._prepare_route(handle, errors, M, rows)
        return mu

    def _prepare_route(self, handle, errors, M, rows):
        """The call for ``M`` modes and the latest ``rows`` NIS rows: buffers and the
        C pointers into them, made once."""
        from ..models import native
        from ..native import core
        c, lk, mu, nis = np.empty(M), np.empty(M), np.empty(M), np.empty((max(rows, 1), M))
        ptr = core.ffi.from_buffer
        return (native, M, rows, core.lib.kalman_router_route, handle, errors, c, lk, mu, nis,
                ptr("double[]", c), ptr("double[]", lk), ptr("double[]", mu), ptr("double[]", nis), core.ffi.NULL, core,
                self._route_token())

    def _prepared_route(self, prepared, nis_arr, c, likelihoods):
        """``mu`` from the prepared call, or ``None`` when the inputs are not its shape
        or type (the general path then decides)."""
        (native, M, rows, call, handle, errors, cb, lb, mub, nb, cp, lp, mup, np_, null, core, token) = prepared
        if not native.USE_NATIVE_LOOP:
            return None
        if token is not None and self._route_token() != token:
            self.__dict__.pop("_route_call", None)      # a router it uses was changed
            return None
        if type(c) is not np.ndarray or type(likelihoods) is not np.ndarray or type(nis_arr) is not np.ndarray:
            return None
        if c.shape != (M,) or likelihoods.shape != (M,) or nis_arr.ndim != 2 or nis_arr.shape[1] != M:
            return None
        try:
            np.copyto(cb, c)
            np.copyto(lb, likelihoods)
            k = min(rows, nis_arr.shape[0])
            if k:
                np.copyto(nb[:k], nis_arr[nis_arr.shape[0] - k:])
        except (TypeError, ValueError):                 # complex, object: not for the C API
            return None
        errors.clear()
        if call(handle, k, M, np_ if k else null, cp, lp, mup) < 0:
            if errors:
                raise errors[0]
            core.check(-1, "kalman_router_route")
        return mub.copy()

    @staticmethod
    def _normalise(mu: np.ndarray) -> np.ndarray:
        """Clip to (ε, ∞) and normalise to a probability vector."""
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-12)
        return mu / mu.sum()

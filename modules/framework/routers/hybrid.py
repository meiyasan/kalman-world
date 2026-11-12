"""HybridRouter — geometric blend of two arbitrary routers."""
from __future__ import annotations
import numpy as np
from .base import KalmanRouter


class HybridRouter(KalmanRouter):
    """Geometric blend (log-linear opinion pool) of two routers.

        μ = router_a(...)^(1−α) · router_b(...)^α  / Z

    At α = 0 → pure router_a.  At α = 1 → pure router_b.

    Common uses::

        # Bayesian anchor + learned refinement
        HybridRouter(BayesianRouter(), NeuralRouter(MLP()), alpha=0.3)

        # Learned primary + rule-based guard
        HybridRouter(NeuralRouter(MLP()), ThresholdRouter(soft=True), alpha=0.5)

    ``fit()`` is forwarded to both children.

    Parameters
    ----------
    router_a, router_b : KalmanRouter
    alpha : float in [0, 1]
        Blend weight toward router_b.

    Blended by ``Kalman::Routers::Hybrid`` when libKalman is installed. A child
    with a C++ counterpart (Bayesian, Threshold, Hybrid) is evaluated there; any
    other child (a NeuralRouter, a plain callable) is called back from C++.
    :meth:`_reference_route` is the Python reference.
    """

    NATIVE = "hybrid"

    def __init__(self,
                 router_a: KalmanRouter,
                 router_b: KalmanRouter,
                 alpha: float = 0.5):
        self.router_a = router_a
        self.router_b = router_b
        self.alpha    = float(alpha)

    def route(self, nis_arr, c, likelihoods):
        mu = self._native_route(nis_arr, c, likelihoods)
        return mu if mu is not None else self._reference_route(nis_arr, c, likelihoods)

    def _reference_route(self, nis_arr, c, likelihoods):
        mu_a = np.maximum(np.asarray(self.router_a(nis_arr, c, likelihoods), float), 1e-12)
        mu_b = np.maximum(np.asarray(self.router_b(nis_arr, c, likelihoods), float), 1e-12)
        return self._normalise(mu_a ** (1 - self.alpha) * mu_b ** self.alpha)

    # --- the C++ blend ---------------------------------------------------------
    @staticmethod
    def _child(router):
        """``(handle, keepalive, errors, rows)`` of a child: its own C++ router, or a
        callback router that calls it."""
        if isinstance(router, KalmanRouter):
            found = router._native_router()
            if found is not None:
                return found[0], router, found[1], router._native_rows()
        from ..native import core
        ffi, errors = core.ffi, []

        @ffi.callback("int(void *, int, int, const double *, const double *, const double *, double *)")
        def call(_user, rows, M, nis, c, likelihoods, mu):
            try:
                def read(ptr, size):
                    return np.frombuffer(ffi.buffer(ptr, size * 8), dtype=np.float64).copy() if size else np.empty(0)
                out = np.asarray(router(read(nis, rows * M).reshape(rows, M), read(c, M), read(likelihoods, M)),
                                 dtype=np.float64).reshape(-1)
                if out.size != M:
                    raise ValueError(f"the router returned {out.size} probabilities for {M} models.")
                ffi.memmove(mu, np.ascontiguousarray(out), M * 8)
                return 0
            except BaseException as exc:            # must not unwind through C++
                errors.append(exc)
                return 1

        return core.create_router("callback", route=call), (router, call), errors, None

    def _native_key(self):
        if not 0.0 <= self.alpha <= 1.0:
            return None
        # A child's C++ handle is part of the key: when it is rebuilt, so is the blend.
        return (self.alpha,) + tuple(self._child_key(r) for r in (self.router_a, self.router_b))

    @staticmethod
    def _child_key(router):
        if isinstance(router, KalmanRouter):
            found = router._native_router()
            if found is not None:
                return found[0]
        return id(router)

    def _native_create(self):
        from ..native import core
        a, b = self._child(self.router_a), self._child(self.router_b)
        errors = _Errors(a[2], b[2])
        handle = core.create_router("hybrid", a=a[0], b=b[0], alpha=self.alpha)
        self._rows = None if a[3] is None or b[3] is None else max(a[3], b[3])
        return handle, (a, b), errors

    def _native_rows(self):
        self._native_router()                           # sets _rows for the current children
        return self.__dict__.get("_rows")

    def fit(self, nis_arrays, labels_arrays, **kwargs) -> "HybridRouter":
        self.router_a.fit(nis_arrays, labels_arrays, **kwargs)
        self.router_b.fit(nis_arrays, labels_arrays, **kwargs)
        return self


class _Errors:
    """The exceptions of both children's callbacks, read and cleared as one list."""

    def __init__(self, *lists):
        self._lists = lists

    def clear(self):
        for errors in self._lists:
            errors.clear()

    def __bool__(self):
        return any(self._lists)

    def __getitem__(self, i):
        return [e for errors in self._lists for e in errors][i]

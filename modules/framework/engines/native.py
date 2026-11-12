"""
The C++ estimation step, driven by the Python filter.

:class:`NativeEngine` is a :class:`~kalman.framework.engine.KalmanEngine` whose
``_predict`` / ``_update`` / ``_smooth`` run in ``libKalman`` — the same
``Engines::Linear`` / ``Extended`` / ``Unscented`` / ``Particle`` / ``Ensemble`` a
C++ or ROOT+ program uses — through the C API. The Python
:class:`~kalman.framework.filter.KalmanFilter` keeps the workflow, the records,
the diagnostics and the plots; only the arithmetic crosses.

It needs a :class:`~kalman.framework.models.native.NativeModel`: the C++ engine
evaluates the model's ``f``/``h``/``F``/``Q`` on the C++ side, and there is no
route from there back into a Python-defined model — by design, since a
per-step callback into Python is what the measurements in ``docs/15`` rule out.
A Python model keeps its Python engines.

Not registered in ``ENGINE_BY_KIND``: it is a *backend* for an existing kind,
not a kind. ``KalmanFilter(model, engine=NativeEngine("extended"))`` is the
per-step form; :meth:`NativeModel.filter_native` is the fused one.
"""
from __future__ import annotations
import numpy as np

from ..engine import KalmanEngine
from ..constants import KalmanEngineKind
from ..native import core

KIND_BY_NAME = {k.name.lower(): k for k in KalmanEngineKind}


class NativeEngine(KalmanEngine):
    """A libKalman engine behind the Python :class:`KalmanFilter`."""

    NAME = "Native Engine"
    DESCRIPTION = "A libKalman (C++) estimation step driven by the Python filter."
    VERSION = "1.0"

    def __init__(self, kind="extended", **params):
        """``kind`` is an engine name (``"linear"``, ``"extended"``, ``"unscented"``,
        ``"particle"``, ``"ensemble"``) or a :class:`KalmanEngineKind`; ``params`` are the
        C++ constructor's (``alpha/beta/kappa``, ``n_particles/threshold/seed``,
        ``n_members/inflation/seed``)."""
        name = kind.name.lower() if isinstance(kind, KalmanEngineKind) else str(kind).lower()
        if name not in core.ENGINE_KINDS:
            raise ValueError(f"unknown native engine kind {kind!r}; choose from {sorted(core.ENGINE_KINDS)}.")
        self.kind_name = name
        self.KIND = KIND_BY_NAME[name]
        self.params = dict(params)
        self._handle = core.create_engine(name, **params)
        self._model_id = None

    # ------------------------------------------------------------------
    @property
    def handle(self):
        return self._handle

    @property
    def has_smoother(self) -> bool:
        return core.check(core.lib.kalman_engine_has_smoother(self._handle), "has_smoother") == 1

    @staticmethod
    def _native(model):
        h = getattr(model, "_handle", None)
        if h is None:
            raise TypeError(f"NativeEngine needs a NativeModel (a C++ model); {type(model).__name__} "
                            "is a Python model, which keeps its Python engines.")
        return h

    def reset(self, model):
        core.check(core.lib.kalman_engine_reset(self._handle, self._native(model)), "kalman_engine_reset")
        self._model_id = id(model)
        return self

    def _ensure_reset(self, model):
        if self._model_id != id(model):
            self.reset(model)

    @staticmethod
    def _t(t):
        return 0.0 if t is None else float(t)

    # ------------------------------------------------------------------
    def _predict(self, model, x, P, u=None, t=None, **kw):
        mh = self._native(model)
        self._ensure_reset(model)
        n = model.n
        xf = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
        Pf = np.ascontiguousarray(np.asarray(P, dtype=np.float64).reshape(n, n))
        uf = None
        if u is not None and model.l > 0:
            uf = np.ascontiguousarray(np.asarray(u, dtype=np.float64).reshape(-1))
        xo, Po, Fo = np.empty(n), np.empty((n, n)), np.empty((n, n))
        core.check(core.lib.kalman_engine_predict(
            self._handle, mh, core.cdptr(xf), core.cdptr(Pf),
            core.cdptr(uf) if uf is not None else core.ffi.NULL, self._t(t),
            core.dptr(xo), core.dptr(Po), core.dptr(Fo)), "kalman_engine_predict")
        return xo.reshape(-1, 1), Po, Fo

    def _update(self, model, x, P, z, t=None, **kw):
        mh = self._native(model)
        self._ensure_reset(model)
        n, m = model.n, model.m
        xf = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
        Pf = np.ascontiguousarray(np.asarray(P, dtype=np.float64).reshape(n, n))
        zf = np.ascontiguousarray(np.asarray(z, dtype=np.float64).reshape(-1))
        if zf.size != m:
            raise ValueError(f"z has {zf.size} channels, expected {m}.")
        xo, Po = np.empty(n), np.empty((n, n))
        core.check(core.lib.kalman_engine_update(
            self._handle, mh, core.cdptr(xf), core.cdptr(Pf), core.cdptr(zf), self._t(t),
            core.dptr(xo), core.dptr(Po)), "kalman_engine_update")
        return xo.reshape(-1, 1), Po

    def _smooth(self, model, post_k, prior_next, smooth_next, F_k):
        n = model.n
        c = lambda a, shape: np.ascontiguousarray(np.asarray(a, dtype=np.float64).reshape(shape))
        x, P = post_k
        xp, Pp = prior_next
        xs, Ps = smooth_next
        xo, Po = np.empty(n), np.empty((n, n))
        core.check(core.lib.kalman_engine_smooth(
            self._handle, n, core.cdptr(c(x, -1)), core.cdptr(c(P, (n, n))),
            core.cdptr(c(xp, -1)), core.cdptr(c(Pp, (n, n))),
            core.cdptr(c(xs, -1)), core.cdptr(c(Ps, (n, n))), core.cdptr(c(F_k, (n, n))),
            core.dptr(xo), core.dptr(Po)), "kalman_engine_smooth")
        return xo.reshape(-1, 1), Po

    def __repr__(self):
        return f"NativeEngine({self.kind_name!r}{', ' + repr(self.params) if self.params else ''})"

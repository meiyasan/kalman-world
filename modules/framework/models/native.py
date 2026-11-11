"""
A libKalmanFramework model as a :class:`~kalman.framework.model.KalmanModel`.

:class:`NativeModel` holds an instance built by the C++ factory and forwards the
model contract — ``f/h``, ``F/H/B/L``, ``Q/R``, the initial state — to it through
the C API (:mod:`kalman.framework.native.core`). Everything above the contract is
the ordinary Python framework: the engines, the filters, the sampler, the record,
``plot``, the API verbs and the MCP tools all see just another ``KalmanModel``.

That is the one crossing between the two implementations for *models*: the C++
side owns the physics, the Python side owns everything a notebook does with it.
A crossing costs one C call per ``f``/``h``/coefficient evaluation, which is what
the measured numbers in ``docs/15_Cxx-Library.md`` say is affordable in this
direction (a C call is cheap; it is a *Python* callback from C++ that is not).

:func:`native_model_classes` turns every registered name into a class — ``NAME``,
``DESCRIPTION``, ``VERSION`` and ``DEFAULT_ENGINE`` filled from the C++ metadata —
so the catalogue, ``describe()`` and ``make_model()`` treat a plugin model exactly
like a class in ``apps/``: ``{"base": "constant-velocity", "sensor_id": "cv"}``.
"""
from __future__ import annotations
import numpy as np

from ..model import KalmanModel
from ..constants import KalmanEngineKind, ALL_ENGINES
from ..native import core

# Kalman::EngineKind (Constants.h) -> the Python kind, by the shared integer.
ENGINE_BY_INT = {0: KalmanEngineKind.LINEAR, 1: KalmanEngineKind.EXTENDED,
                 2: KalmanEngineKind.UNSCENTED, 3: KalmanEngineKind.PARTICLE,
                 4: KalmanEngineKind.ENSEMBLE, 5: KalmanEngineKind.COMBINATORIAL}


class NativeModel(KalmanModel):
    """A C++ model from the factory, driven by the Python engines.

    ``NativeModel("constant-velocity", "sensor-1")`` constructs the registered
    C++ model and adopts its dimensions, sample rate, engine binding and initial
    state. ``fs=``, ``x0=``, ``P0=`` and ``t0=`` override those and are pushed
    back into the C++ instance, so both sides agree on every step length.
    """

    NAME = "Native Model"
    DESCRIPTION = "A libKalmanFramework model (C++) exposed through the C API."
    VERSION = "1.0"

    NATIVE_NAME: str | None = None      # set on the classes native_model_classes() builds
    DEFAULT_ENGINE = KalmanEngineKind.EXTENDED
    ALLOWED_ENGINES = ALL_ENGINES

    def __init__(self, name: str | None = None, sensor_id=None, **kwargs):
        name = name or self.NATIVE_NAME
        if not name:
            raise ValueError("NativeModel needs the registered name of a C++ model.")
        L = core.require()
        handle = L.kalman_model_create(name.encode())
        if handle == core.ffi.NULL:
            raise KeyError(f"NativeModel: {core.last_error()}")
        self._handle = core.ffi.gc(handle, L.kalman_model_destroy)
        self.native_name = name

        n, m, l = core.ffi.new("int *"), core.ffi.new("int *"), core.ffi.new("int *")
        core.check(L.kalman_model_dims(self._handle, n, m, l), "kalman_model_dims")
        self.n, self.m, self.l = int(n[0]), int(m[0]), int(l[0])

        # Time base: an explicit fs wins and is pushed down; otherwise the C++ one is adopted.
        fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None)
        native_fs = float(L.kalman_model_sample_rate(self._handle))
        if fs is not None:
            core.check(L.kalman_model_set_sample_rate(self._handle, float(fs)), "set_sample_rate")
        else:
            fs = native_fs if native_fs > 0 else None
        if "t0" in kwargs and kwargs["t0"] is not None:
            core.check(L.kalman_model_set_t0(self._handle, float(kwargs["t0"])), "set_t0")
        else:
            kwargs["t0"] = float(L.kalman_model_t0(self._handle))

        # Initial state: same rule. A C++ P0 that was never set is all zeros,
        # which the Python side reads as "infer from Q", as it does for any model.
        x0 = kwargs.pop("x0", None)
        if x0 is not None:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
            core.check(L.kalman_model_set_x0(self._handle, core.cdptr(np.ascontiguousarray(x0))), "set_x0")
        else:
            x0 = self._vec(L.kalman_model_x0, "x0")
        P0 = kwargs.pop("P0", None)
        if P0 is not None:
            P0 = np.ascontiguousarray(np.asarray(P0, dtype=float).reshape(self.n, self.n))
            core.check(L.kalman_model_set_P0(self._handle, core.cdptr(P0)), "set_P0")
        else:
            P0 = self._mat(L.kalman_model_P0, (self.n, self.n), "P0")
            if not np.any(P0):
                P0 = None

        self.DEFAULT_ENGINE = ENGINE_BY_INT.get(
            core.check(L.kalman_model_engine(self._handle), "engine"), KalmanEngineKind.EXTENDED)
        self._analytic = core.check(L.kalman_model_analytic_jacobian(self._handle), "analytic") == 1

        super().__init__(sensor_id if sensor_id is not None else name,
                         fs=fs, x0=x0.reshape(-1, 1), P0=P0, **kwargs)

    # --- plumbing ---------------------------------------------------------------
    def _vec(self, fn, what, *args):
        out = np.empty(self.n if what != "z" else self.m, dtype=np.float64)
        core.check(fn(self._handle, *args, core.dptr(out)), what)
        return out

    def _mat(self, fn, shape, what, *args):
        out = np.empty(shape, dtype=np.float64)
        core.check(fn(self._handle, *args, core.dptr(out)), what)
        return out

    def _mat_optional(self, fn, shape, what, t):
        out = np.empty(shape, dtype=np.float64)
        return out if core.check(fn(self._handle, t, core.dptr(out)), what) == 1 else None

    @staticmethod
    def _t(t):
        return 0.0 if t is None else float(t)

    def _flat(self, v, size, what):
        a = np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1))
        if a.size != size:
            raise ValueError(f"{self.native_name}: {what} has {a.size} elements, expected {size}.")
        return a

    def _u(self, u):
        if u is None or self.l == 0:
            return core.ffi.NULL
        return core.cdptr(self._flat(u, self.l, "u"))

    @staticmethod
    def _like(out, x):
        # Column in, column out; flat in, flat out — the engines use both.
        return out.reshape(-1, 1) if np.ndim(x) == 2 else out

    # --- the contract: coefficients --------------------------------------------
    def F(self, x=None, u=None, t=None):
        L = core.lib
        if self._analytic:
            return self._mat(L.kalman_model_F, (self.n, self.n), "F", self._t(t))
        xf = self._flat(self.x0 if x is None else x, self.n, "x")
        return self._mat(L.kalman_model_jacobian, (self.n, self.n), "df/dx",
                         core.cdptr(xf), self._u(u), self._t(t))

    def H(self, x=None, t=None):
        L = core.lib
        if self._analytic:
            return self._mat(L.kalman_model_H, (self.m, self.n), "H", self._t(t))
        xf = self._flat(self.x0 if x is None else x, self.n, "x")
        return self._mat(L.kalman_model_jacobian_h, (self.m, self.n), "dh/dx",
                         core.cdptr(xf), self._t(t))

    def B(self, x=None, u=None, t=None):
        if self.l == 0:
            return None
        return self._mat_optional(core.lib.kalman_model_B, (self.n, self.l), "B", self._t(t))

    def L(self, x=None, u=None, w=None, t=None):
        return self._mat_optional(core.lib.kalman_model_L, (self.n, self.n), "L", self._t(t))

    def M(self, x=None, v=None, t=None):
        return None                                  # additive measurement noise

    def Q(self, t=None):
        return self._mat(core.lib.kalman_model_Q, (self.n, self.n), "Q", self._t(t))

    def R(self, t=None):
        return self._mat(core.lib.kalman_model_R, (self.m, self.m), "R", self._t(t))

    # --- the contract: maps ----------------------------------------------------
    def f(self, x, u=None, w=None, t=None):
        xf = self._flat(x, self.n, "x")
        out = np.empty(self.n, dtype=np.float64)
        core.check(core.lib.kalman_model_f(self._handle, core.cdptr(xf), self._u(u), self._t(t),
                                           core.dptr(out)), "f")
        if w is not None:
            Lw = self.L(t=t)
            wf = self._flat(w, self.n, "w")
            out = out + (wf if Lw is None else Lw @ wf)
        return self._like(out, x)

    def h(self, x, v=None, t=None):
        xf = self._flat(x, self.n, "x")
        out = np.empty(self.m, dtype=np.float64)
        core.check(core.lib.kalman_model_h(self._handle, core.cdptr(xf), self._t(t), core.dptr(out)), "h")
        if v is not None:
            out = out + self._flat(v, self.m, "v")
        return self._like(out, x)

    # --- the fused C++ pass -----------------------------------------------------
    def run_native(self, zk, uk=None, *, engine="extended", smooth=False, **engine_params) -> dict:
        """Kalman::Filter over ``zk`` entirely in C++ with a native ``engine`` (name
        or kind, plus its constructor parameters). Returns the per-step arrays:
        ``x_prior/P_prior/x_post/P_post/innovations/transitions`` and, with
        ``smooth=True``, ``x_smooth/P_smooth``."""
        eng = core.create_engine(engine.name.lower() if hasattr(engine, "name") else str(engine).lower(),
                                 **engine_params)
        return core.run_filter(self._handle, eng, self.n, self.m, self.l, zk, uk, smooth=smooth)

    def filter_native(self, zk, uk=None, *, engine="extended", smooth=False, **engine_params):
        """:meth:`run_native` as a :class:`~kalman.framework.io.record.KalmanRecord` —
        posterior, or smoothed when ``smooth=True`` — so it plots, records and
        compares like any Python filter's output."""
        from ..io.record import KalmanRecord
        from ..constants import KalmanStateFlavor
        out = self.run_native(zk, uk, engine=engine, smooth=smooth, **engine_params)
        N = out["x_post"].shape[0]
        t0 = float(self.t0 or 0.0)
        time = t0 + np.arange(N, dtype=float) / (self.fs or 1.0)
        zk = np.asarray(zk, dtype=float).reshape(N, self.m)
        if smooth:
            return KalmanRecord(time=time, xk=out["x_smooth"], Pk=out["P_smooth"], zk=zk,
                                model=self, flavor=KalmanStateFlavor.SMOOTHED)
        return KalmanRecord(time=time, xk=out["x_post"], Pk=out["P_post"], zk=zk,
                            model=self, flavor=KalmanStateFlavor.POSTERIOR)

    # --- introspection ----------------------------------------------------------
    def describe_native(self) -> dict:
        """The C++ side's own account of this model (title, plugin, dims ...)."""
        d = core.describe(self.native_name)
        d["summary"] = core._s(core.lib.kalman_model_describe(self._handle))
        d["library"] = core.LIBRARY_PATH
        return d

    def __repr__(self):
        return f"NativeModel({self.native_name!r}, n={self.n}, m={self.m}, l={self.l}, fs={self.fs})"


# --- a Python model with a C++ twin ---------------------------------------------
APPS_LIBRARY = "KalmanApps"
#: Switch every twin off (``False``) to run the Python physics — how a before/after
#: comparison is made. ``KALMAN_NATIVE_TWINS=0`` in the environment does the same.
USE_NATIVE_TWINS = True
_apps_names: set[str] | None = None


def apps_library_names() -> set[str]:
    """Names registered by libKalmanApps, loading it on first use; empty when the
    C++ core or that library is not installed."""
    global _apps_names
    if _apps_names is None:
        _apps_names = set()
        if core.HAS_CXX_CORE:
            try:
                _apps_names = set(core.load_plugin(core.resolve_plugin(APPS_LIBRARY)))
            except (FileNotFoundError, PermissionError):
                pass                                  # not built: the Python physics runs
    return _apps_names


class NativeTwin:
    """Mixin for an application model whose physics also exists in libKalmanApps.

    The Python class stays the model: its constructor, its attributes, ``plot``
    and the notebooks are unchanged. What changes is who runs the filtering
    pass. When the configuration has a C++ counterpart, ``KalmanFilter.forward``
    hands the whole pass to the twin (``Kalman::Filter`` in C++, one call), and
    otherwise the Python loop runs as before.

    A model declares its twin with two methods:

    - :meth:`native_name` — the registered C++ name for *this* configuration, or
      ``None`` when it has none (a Python callback, a feature not ported);
    - :meth:`native_parameters` — the scalars to push, by the names the C++
      class declares.

    The sample rate, ``Q`` and ``R`` are pushed from the Python instance on every
    run, and the initial state by the filter, so nothing is defined twice and a
    change made to the Python model after construction reaches the twin.
    """

    #: What must be exactly the declaring class's own for the twin to stand in.
    _TWIN_CONTRACT = ("f", "h", "F", "H", "B", "L", "Q", "R")

    def native_name(self) -> str | None:
        return None

    def native_parameters(self) -> dict:
        return {}

    def _twin_owner(self) -> type:
        for cls in type(self).__mro__:
            if "native_name" in cls.__dict__ and cls is not NativeTwin:
                return cls
        return NativeTwin

    def native_handle(self):
        """The configured C++ twin, or ``None`` when the Python physics must run."""
        import os
        if not USE_NATIVE_TWINS or os.environ.get("KALMAN_NATIVE_TWINS", "1") == "0":
            return None
        if not core.HAS_CXX_CORE:
            return None
        name = self.native_name()
        if not name or name not in apps_library_names():
            return None
        # A subclass or an instance that redefines the contract is a different
        # model from the one that was ported.
        owner = self._twin_owner()
        for attr in self._TWIN_CONTRACT:
            if attr in self.__dict__ or getattr(type(self), attr, None) is not getattr(owner, attr, None):
                return None
        if self.l or np.dtype(getattr(self, "dtype", np.float64)) != np.float64:
            return None
        if self._F is not None or self._H is not None or callable(self._Q) or callable(self._R):
            return None
        L_ = self._L
        if L_ is not None and (callable(L_) or not np.array_equal(np.asarray(L_), np.eye(self.n))):
            return None

        lib = core.lib
        cached = self.__dict__.get("_native_twin")
        if cached is None or cached[0] != name:
            handle = lib.kalman_model_create(name.encode())
            if handle == core.ffi.NULL:
                raise KeyError(f"{type(self).__name__}: {core.last_error()}")
            handle = core.ffi.gc(handle, lib.kalman_model_destroy)
            n, m, l = core.ffi.new("int *"), core.ffi.new("int *"), core.ffi.new("int *")
            core.check(lib.kalman_model_dims(handle, n, m, l), "kalman_model_dims")
            if (int(n[0]), int(m[0]), int(l[0])) != (self.n, self.m, self.l):
                raise RuntimeError(
                    f"{type(self).__name__}: the C++ twin {name!r} has (n, m, l) = "
                    f"({n[0]}, {m[0]}, {l[0]}), the Python model ({self.n}, {self.m}, {self.l}).")
            self._native_twin = cached = (name, handle)
        handle = cached[1]

        if self.fs:
            core.check(lib.kalman_model_set_sample_rate(handle, float(self.fs)), "set_sample_rate")
        for key, value in self.native_parameters().items():
            core.check(lib.kalman_model_set_parameter(handle, key.encode(), float(value)),
                       f"set_parameter({key!r})")
        Q = np.ascontiguousarray(np.asarray(self._Q, dtype=np.float64).reshape(self.n, self.n))
        R = np.ascontiguousarray(np.asarray(self._R, dtype=np.float64).reshape(self.m, self.m))
        core.check(lib.kalman_model_set_Q(handle, core.cdptr(Q)), "set_Q")
        core.check(lib.kalman_model_set_R(handle, core.cdptr(R)), "set_R")
        return handle


# --- one class per registered name ---------------------------------------------
_CLASSES: dict[str, type] = {}


def _init(self, sensor_id=None, fs=None, **kwargs):
    # `fs` is named (not swallowed by **kwargs) so make_model() binds the spec's
    # sample rate to it the way it does for every other catalogue class; None
    # means "keep the C++ model's own".
    if fs is not None:
        kwargs["fs"] = fs
    NativeModel.__init__(self, self.NATIVE_NAME, sensor_id, **kwargs)


def native_model_classes() -> dict[str, type]:
    """``{registered name -> NativeModel subclass}`` for every model the C++
    factory currently knows, including those loaded from plugins since the last
    call. Empty when the library is not available."""
    if not core.HAS_CXX_CORE:
        return {}
    for name in core.names():
        # libKalmanApps' models are reached through their Python classes
        # (StormModel, BallisticModel ...), not listed a second time here.
        if name in _CLASSES or name in (_apps_names or ()):
            continue
        info = core.describe(name)
        _CLASSES[name] = type(name, (NativeModel,), {
            "__module__": __name__,
            "__doc__": f"{info['title']} — {info['description']} (C++, {info['plugin'] or 'built in'})",
            "__init__": _init,
            "NATIVE_NAME": name,
            "NAME": info["title"],
            "DESCRIPTION": info["description"],
            "VERSION": info["version"],
            "DEFAULT_ENGINE": ENGINE_BY_INT.get(info["engine"], KalmanEngineKind.EXTENDED),
        })
    return dict(_CLASSES)

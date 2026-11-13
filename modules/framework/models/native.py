"""
A libKalman model as a :class:`~kalman.framework.model.KalmanModel`.

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
    DESCRIPTION = "A libKalman model (C++) exposed through the C API."
    VERSION = "1.0"

    NATIVE_NAME: str | None = None      # set on the classes native_model_classes() builds
    DEFAULT_ENGINE = KalmanEngineKind.EXTENDED
    ALLOWED_ENGINES = ALL_ENGINES

    def __init__(self, name: str | None = None, sensor_id=None, **kwargs):
        L = core.require()
        # A model that exists already (read from a file) is wrapped as it is.
        handle = kwargs.pop("_handle", None)
        if handle is not None:
            name = name or core._s(L.kalman_model_name(handle))
        else:
            name = name or self.NATIVE_NAME
            if not name:
                raise ValueError("NativeModel needs the registered name of a C++ model.")
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

        # Noise and named parameters live in the C++ instance (Q(), R() and
        # parameters() read them from there), so what is passed here is pushed
        # down, not kept on the Python side where nothing would read it.
        for key, setter, size in (("Q", L.kalman_model_set_Q, self.n), ("R", L.kalman_model_set_R, self.m)):
            value = kwargs.pop(key, None)
            if value is not None:
                value = np.ascontiguousarray(np.asarray(value, dtype=float).reshape(size, size))
                core.check(setter(self._handle, core.cdptr(value)), "set_" + key)
                # A model that computes its noise from its own parameters does
                # not take a matrix: say so instead of running with another one.
                getter = L.kalman_model_Q if key == "Q" else L.kalman_model_R
                if not np.allclose(self._mat(getter, (size, size), key, 0.0), value, rtol=1e-12, atol=0.0):
                    raise ValueError(
                        f"the C++ model {name!r} computes its own {key} and does not take one; "
                        f"set its parameters instead ({', '.join(sorted(self.parameters())) or 'it declares none'}).")
        declared = self.parameters()
        for key in [k for k in kwargs if k in declared]:
            self.set_parameter(key, kwargs.pop(key))

        self.DEFAULT_ENGINE = ENGINE_BY_INT.get(
            core.check(L.kalman_model_engine(self._handle), "engine"), KalmanEngineKind.EXTENDED)
        self._analytic = core.check(L.kalman_model_analytic_jacobian(self._handle), "analytic") == 1

        super().__init__(sensor_id if sensor_id is not None else name,
                         fs=fs, x0=x0.reshape(-1, 1), P0=P0, **kwargs)

    def native_handle(self):
        """The C++ model itself, so that a filtering pass runs wholly in
        Kalman::Filter instead of asking for f, h and the matrices step by step.
        ``None`` for a subclass that rewrites part of the model in Python."""
        import os
        if not USE_NATIVE_TWINS or os.environ.get("KALMAN_NATIVE_TWINS", "1") == "0":
            return None
        for name in ("f", "h", "F", "H", "B", "L", "Q", "R"):
            if name in self.__dict__ or getattr(type(self), name) is not getattr(NativeModel, name):
                return None
        return self._handle

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

    # --- named parameters -------------------------------------------------------
    def parameters(self) -> dict:
        """The model's named parameters and their current values (``{}`` when it
        declares none) — the scalars its C++ class exposes with ``DeclareParameter``."""
        lib = core.lib
        value = core.ffi.new("double *")
        out = {}
        for i in range(core.check(lib.kalman_model_parameter_count(self._handle), "parameter_count")):
            name = core._s(lib.kalman_model_parameter_name(self._handle, i))
            core.check(lib.kalman_model_parameter(self._handle, name.encode(), value), name)
            out[name] = float(value[0])
        return out

    def set_parameter(self, name: str, value: float):
        """Set one named parameter of the C++ model. An unknown name raises
        ``KeyError`` listing the names the model declares."""
        if core.lib.kalman_model_set_parameter(self._handle, name.encode(), float(value)) < 0:
            raise KeyError(core.last_error())
        return self

    def arrays(self) -> dict:
        """The model's named arrays (``{}`` when it declares none): what its C++
        class exposes with ``DeclareArray``."""
        lib = core.lib
        out = {}
        for i in range(core.check(lib.kalman_model_array_count(self._handle), "array_count")):
            name = core._s(lib.kalman_model_array_name(self._handle, i))
            values = np.empty(core.check(lib.kalman_model_array_size(self._handle, name.encode()), name))
            if values.size:
                core.check(lib.kalman_model_array(self._handle, name.encode(), core.dptr(values)), name)
            out[name] = values
        return out

    def set_array(self, name: str, values):
        """Set one named array of the C++ model. The model may resize itself (one
        measurement per satellite, say): ``n``, ``m`` and ``l`` are read again."""
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1))
        if core.lib.kalman_model_set_array(self._handle, name.encode(), array.size, core.cdptr(array)) < 0:
            raise KeyError(core.last_error())
        n, m, l = core.ffi.new("int *"), core.ffi.new("int *"), core.ffi.new("int *")
        core.check(core.lib.kalman_model_dims(self._handle, n, m, l), "kalman_model_dims")
        self.n, self.m, self.l = int(n[0]), int(m[0]), int(l[0])
        return self

    # --- introspection ----------------------------------------------------------
    @property
    def structure(self):
        """What the C++ model declares it is made of."""
        from ..constants import KalmanStructureKind
        text = core._s(core.lib.kalman_model_structure(self._handle))
        return {"maps": KalmanStructureKind.MAPS, "matrices": KalmanStructureKind.MATRICES,
                "time-varying matrices": KalmanStructureKind.TIME_VARYING}[text]

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
#: Run the filter loop in C++ for a model written in Python (its f/h called back).
#: ``False``, or ``KALMAN_NATIVE_LOOP=0``, keeps the Python reference loop.
USE_NATIVE_LOOP = True


def native_loop_enabled() -> bool:
    import os
    return USE_NATIVE_LOOP and os.environ.get("KALMAN_NATIVE_LOOP", "1") != "0" and core.HAS_CXX_CORE


class python_reference:
    """Context manager: inside it every model runs the Python reference path
    (no C++ twin, no C++ loop). What the equivalence tests and the benchmarks
    compare against."""

    def __enter__(self):
        global USE_NATIVE_TWINS, USE_NATIVE_LOOP
        self._saved = (USE_NATIVE_TWINS, USE_NATIVE_LOOP)
        USE_NATIVE_TWINS = USE_NATIVE_LOOP = False
        return self

    def __exit__(self, *exc):
        global USE_NATIVE_TWINS, USE_NATIVE_LOOP
        USE_NATIVE_TWINS, USE_NATIVE_LOOP = self._saved
        return False


def callback_eligible(model) -> bool:
    """Whether the C++ extended engine can run ``model`` through callbacks: any
    model in double precision. What differs between models is only whether the
    noise is handed over once or asked for at every step (:func:`constant_noise`)."""
    return np.dtype(getattr(model, "dtype", np.float64)) == np.float64


def constant_noise(model) -> bool:
    """Whether ``model``'s noise is constant and additive: ``Q``, ``R`` and the
    noise coupling ``L`` are the base class's and hold matrices. Then they are
    pushed to C++ once. Otherwise (an overridden or callable ``Q``/``R``/``L``,
    a state-dependent ``Q``, a coupling left to be differenced from ``f``) the
    C++ filter asks the Python model for them at every step."""
    from ..model import KalmanModel
    cls = type(model)
    if any(getattr(cls, name) is not getattr(KalmanModel, name) for name in ("Q", "R", "L")):
        return False
    if any(name in model.__dict__ for name in ("Q", "R", "L")):
        return False
    return not (callable(model._Q) or callable(model._R) or model._L is None or callable(model._L))
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
    #: A continuous model's class keeps what it wrote as ``f`` under
    #: ``derivative`` (KalmanModel.__init_subclass__), so its ``f`` is the shared
    #: decoded step: the derivative, the closing step and the sub-stepping are
    #: what a subclass changes, and are checked too.
    _TWIN_CONTRACT = ("f", "h", "F", "H", "B", "L", "Q", "R", "derivative", "after_step", "substeps")

    def native_name(self) -> str | None:
        return None

    def native_parameters(self) -> dict:
        return {}

    def native_arrays(self) -> dict:
        """The named arrays to push (the C++ class's ``DeclareArray`` names). The
        twin may size itself from them."""
        return {}

    #: Stored matrices the twin computes itself from what is pushed (``("_F",)``
    #: for a model built with ``A=``): they do not keep the twin from standing in.
    _TWIN_DECODES = ()

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
        # A control input is fine: the twin's f takes u, and the filter hands
        # the twin the control sequence (a controller function keeps the Python
        # model, see KalmanFilter._native_twin).
        if np.dtype(getattr(self, "dtype", np.float64)) != np.float64:
            return None
        if any(getattr(self, attr) is not None for attr in ("_F", "_H") if attr not in self._TWIN_DECODES):
            return None
        if callable(self._Q) or callable(self._R):
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
            self._push_arrays(handle)             # a twin may size itself from them
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
        # The twin linearises with the scheme the Python model declares.
        scheme = getattr(getattr(self, "DIFFERENTIATOR", None), "name", "FORWARD").lower()
        if scheme not in ("forward", "central"):
            return None
        core.check(lib.kalman_model_set_differentiator(handle, core.DIFFERENTIATOR_KINDS[scheme], 0.0),
                   "set_differentiator")
        self._push_arrays(handle)
        for key, value in self.native_parameters().items():
            core.check(lib.kalman_model_set_parameter(handle, key.encode(), float(value)),
                       f"set_parameter({key!r})")
        Q = np.ascontiguousarray(np.asarray(self._Q, dtype=np.float64).reshape(self.n, self.n))
        R = np.ascontiguousarray(np.asarray(self._R, dtype=np.float64).reshape(self.m, self.m))
        core.check(lib.kalman_model_set_Q(handle, core.cdptr(Q)), "set_Q")
        core.check(lib.kalman_model_set_R(handle, core.cdptr(R)), "set_R")
        return handle

    def _push_arrays(self, handle):
        for key, value in self.native_arrays().items():
            array = np.ascontiguousarray(np.asarray(value, dtype=np.float64).reshape(-1))
            core.check(core.lib.kalman_model_set_array(handle, key.encode(), array.size, core.cdptr(array)),
                       f"set_array({key!r})")


#: Python integrator name -> Kalman::IntegratorKind, the "integrator" parameter of a twin.
NATIVE_INTEGRATORS = {"euler": 0, "rk4": 1, "dormand_prince": 2}


def native_integrator_kind(model):
    """The C++ ``IntegratorKind`` of ``model``'s integrator, for a continuous twin's
    ``integrator`` parameter; ``None`` when the C++ side has no such scheme (a
    custom integrator, a subclass of a built-in one, Dormand-Prince with other
    tolerances than the C++ defaults)."""
    from ..integrator import KalmanIntegrator, _BUILTIN_STEPS
    integrator = model.default_integrator()
    name = getattr(integrator, "NATIVE", None)
    kind = NATIVE_INTEGRATORS.get(name)
    if kind is None or type(integrator)._step is not _BUILTIN_STEPS.get(name) \
            or type(integrator).step is not KalmanIntegrator.step:
        return None
    if kind == 2 and (integrator.rtol, integrator.atol, integrator.max_substeps) != (1e-6, 1e-9, 10_000):
        return None
    return kind


# --- a Python-written model on the C++ engines ----------------------------------
def callback_handle(model, engine="extended", live_noise=False):
    """A C++ ``Kalman::Models::Callback`` whose ``f``, ``h`` and Jacobians call
    ``model``'s own Python methods, so the C++ filter can run a model written in
    Python. Returns ``(handle, keep)``: ``keep`` holds the callback objects and
    must stay referenced for as long as the handle is used.

    The Jacobians are the model's ``F(x, u, t)`` and ``H(x, t)``, analytic where
    the model defines them and differenced by the Python framework otherwise, so
    both sides linearise alike. ``Q`` and ``R`` are pushed as constants (``Q``
    with the noise coupling ``L`` applied); the sample rate too. A Python
    exception inside a callback stops the pass and is re-raised by the caller
    (see :func:`run_callback_filter`).
    """
    lib = core.require()
    ffi = core.ffi
    n, m, l = int(model.n), int(model.m), int(model.l)
    errors: list[BaseException] = []

    def col(ptr, size):
        return np.frombuffer(ffi.buffer(ptr, size * 8), dtype=np.float64).reshape(size, 1).copy()

    def guard(fn):
        def wrapped(*args):
            try:
                fn(*args)
                return 0
            except BaseException as exc:            # must not unwind through C++
                errors.append(exc)
                return 1
        return wrapped

    def write(out, values, size):
        ffi.memmove(out, np.ascontiguousarray(np.asarray(values, dtype=np.float64).reshape(-1)), size * 8)

    def control(u):
        return None if (u == ffi.NULL or l == 0) else col(u, l)

    @ffi.callback("int(void *, const double *, const double *, double, double *)")
    @guard
    def step(user, x, u, t, out):
        write(out, model.f(col(x, n), u=control(u), t=t), n)

    @ffi.callback("int(void *, const double *, double, double *)")
    @guard
    def measure(user, x, t, out):
        write(out, model.h(col(x, n), t=t), m)

    @ffi.callback("int(void *, const double *, const double *, double, double *)")
    @guard
    def step_jacobian(user, x, u, t, out):
        write(out, model.F(x=col(x, n), u=control(u), t=t), n * n)

    @ffi.callback("int(void *, const double *, double, double *)")
    @guard
    def measure_jacobian(user, x, t, out):
        try:
            H = model.H(x=col(x, n), t=t)
        except TypeError:
            # A model whose h takes no noise argument: the base class cannot
            # difference it (it passes v=). Only the sigma-point engines accept
            # such a model, and they use H for the reported innovation alone.
            from ...helpers import numerical_jacobian
            H = numerical_jacobian(lambda xx: model.h(xx, t=t), col(x, n))
        write(out, H, m * n)

    handle = lib.kalman_model_create_callback(type(model).__name__.encode(), n, m, l,
                                              step, measure, step_jacobian, measure_jacobian, ffi.NULL)
    if handle == ffi.NULL:
        raise RuntimeError(f"callback model: {core.last_error()}")
    handle = ffi.gc(handle, lib.kalman_model_destroy)

    if model.fs:
        core.check(lib.kalman_model_set_sample_rate(handle, float(model.fs)), "set_sample_rate")

    keep = [step, measure, step_jacobian, measure_jacobian]
    if engine in ("unscented", "particle"):
        keep += _unscented_noise(model, handle, guard, col, control, write)
    elif constant_noise(model) and not live_noise:
        Lm = np.asarray(model._L, dtype=np.float64)
        Q = Lm @ np.asarray(model.Q(), dtype=np.float64).reshape(n, n) @ Lm.T
        R = np.asarray(model.R(), dtype=np.float64).reshape(m, m)
        core.check(lib.kalman_model_set_Q(handle, core.cdptr(np.ascontiguousarray(Q))), "set_Q")
        core.check(lib.kalman_model_set_R(handle, core.cdptr(np.ascontiguousarray(R))), "set_R")
    else:
        # The noise as the Python engine works it out at each step: Q at the
        # state the step starts from when Q takes one, then the coupling L.
        from ..engine import KalmanEngine
        takes_state = KalmanEngine._accepts_param(model.Q, "x")

        @ffi.callback("int(void *, const double *, const double *, double, double *)")
        @guard
        def process_noise(user, x, u, t, out):
            xc, uc = col(x, n), control(u)
            Q = model.Q(t=t, x=xc) if takes_state else model.Q(t=t)
            Lm = KalmanEngine._safe_L(model, xc, uc, t)
            if Lm is not None:
                Q = Lm @ Q @ Lm.T
            write(out, Q, n * n)

        @ffi.callback("int(void *, double, double *)")
        @guard
        def measurement_noise(user, t, out):
            write(out, model.R(t=t), m * m)

        core.check(lib.kalman_model_callback_set_noise(handle, process_noise, measurement_noise), "set_noise")
        keep += [process_noise, measurement_noise]
    return handle, (*keep, errors)


def _unscented_noise(model, handle, guard, col, control, write):
    """The noise hooks of the unscented engine, as the Python engine applies
    them: where ``f`` takes ``w`` (or ``h`` takes ``v``) the noise goes through
    the map and the covariance is handed over as it is; otherwise it is added
    afterwards, coupled by ``L`` (or ``M``)."""
    from ..engine import KalmanEngine
    from ..engines.unscented import UnscentedKalmanEngine as U
    lib, ffi = core.lib, core.ffi
    n, m = int(model.n), int(model.m)
    through_f = KalmanEngine._noise_through_f(model)
    through_h = KalmanEngine._noise_through_h(model)

    @ffi.callback("int(void *, const double *, const double *, double, double *)")
    @guard
    def process_noise(user, x, u, t, out):
        Q = U._Q(model, t)
        if not through_f:
            Lm = U._L(model, col(x, n), control(u), t)
            Q = Lm @ Q @ Lm.T
        write(out, Q, n * n)

    @ffi.callback("int(void *, double, double *)")
    @guard
    def measurement_noise(user, t, out):
        R = U._R(model, t)
        if not through_h:
            Mm = U._M(model, np.asarray(model.x0, dtype=np.float64).reshape(-1, 1), t)
            R = Mm @ R @ Mm.T
        write(out, R, m * m)

    @ffi.callback("int(void *, const double *, const double *, const double *, double, double *)")
    @guard
    def noisy_step(user, x, u, w, t, out):
        write(out, model.f(col(x, n), u=control(u), w=col(w, n), t=t), n)

    @ffi.callback("int(void *, const double *, const double *, double, double *)")
    @guard
    def noisy_measure(user, x, v, t, out):
        write(out, model.h(col(x, n), v=col(v, m), t=t), m)

    core.check(lib.kalman_model_callback_set_noise(handle, process_noise, measurement_noise), "set_noise")
    core.check(lib.kalman_model_callback_set_noise_inputs(
        handle, noisy_step if through_f else ffi.NULL, noisy_measure if through_h else ffi.NULL), "set_noise_inputs")
    return [process_noise, measurement_noise, noisy_step, noisy_measure]


def run_callback_filter(model, zk, uk=None, *, x0=None, P0=None, smooth=False,
                        engine="extended", control=None, **engine_params) -> dict:
    """Filter ``zk`` with a C++ engine (``"extended"`` or ``"unscented"``, with
    its constructor parameters) on a Python-written ``model``. Returns the
    per-step arrays of :func:`core.run_filter`."""
    lib = core.require()
    # A controller may retune the model while it runs (adaptive Q and R), so in
    # closed loop the noise is asked for at every step, never handed over once.
    handle, keep = callback_handle(model, engine, live_noise=control is not None)
    n = int(model.n)
    x0 = np.ascontiguousarray(np.asarray(model.x0 if x0 is None else x0, dtype=np.float64).reshape(-1))
    P0 = np.ascontiguousarray(np.asarray(model.P0 if P0 is None else P0, dtype=np.float64).reshape(n, n))
    core.check(lib.kalman_model_set_x0(handle, core.cdptr(x0)), "set_x0")
    core.check(lib.kalman_model_set_P0(handle, core.cdptr(P0)), "set_P0")
    core.check(lib.kalman_model_set_t0(handle, float(model.t0) if model.t0 is not None else 0.0), "set_t0")
    try:
        return core.run_filter(handle, core.create_engine(engine, **engine_params), n, int(model.m),
                               int(model.l), zk, uk, smooth=smooth, control=control)
    except RuntimeError:
        if keep[-1]:
            raise keep[-1][0]                       # the Python exception, not its C++ shadow
        raise


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

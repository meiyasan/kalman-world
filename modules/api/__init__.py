"""
``kalman.api`` — one JSON-friendly facade over the framework, for humans, agents,
and the MCP/REST transports.

It *harvests* the existing metadata and ``*_BY_KIND`` registries rather than
inventing a parallel description layer, and exposes plain-dict/-list verbs (numpy
arrays ↔ nested lists) so any transport can carry them:

  - **introspect** — :func:`catalog`, :func:`describe`
  - **execute**   — :func:`make_filter`, :func:`sample`, :func:`run_filter`,
    :func:`forecast`, :func:`run_osse`, :func:`run_control`, :func:`diagnostics`,
    :func:`render_plot`
  - **author**    — :func:`register_model`, :func:`save_spec`, :func:`scaffold`
    (see ``spec.py``/``scaffold.py``), gated by :mod:`kalman.api.auth`
  - **share**     — :func:`models`, :func:`delete_model`, :func:`register_source`,
    :func:`push_measurements`, :func:`events`: the registry in :mod:`.store`,
    one SQLite file that every process on the same state directory sees, so a
    model registered through REST is in the MCP catalog and the other way round

A model is described by a :class:`KalmanSpec` dict::

    {"base": "matrices" | "continuous" | "<CatalogClassName>",
     "sensor_id": ..., "fs": ...,
     "dynamics": {"F":..,"H":..} | {"A":..,"Qc":..} | {"params": {...}},
     "noise": {"Q":.., "R":..}, "initial": {"x0":.., "P0":..},
     "labels": {"states":[...], "observables":[...]},
     "control": {"B":.., "cost": {"Qcost":.., "Rcost":.., "x_ref":..}},
     "bindings": {"engine":.., "integrator":.., "solver":..}}
"""
from __future__ import annotations
import inspect
import os
import re
import uuid
from collections import OrderedDict
from collections.abc import Mapping
import numpy as np

from ..framework.model import KalmanModel
from ..framework.filter import KalmanFilter
from ..framework.constants import (KalmanEngineKind, KalmanIntegratorKind,
                                   KalmanSolverKind)

from . import store
from .store import ConflictError

#: run_id -> {"record":.., "model":.., "spec":..}. Runs hold live model objects,
#: so they stay in the process that made them -- one process when the MCP server
#: talks to the REST API. The oldest go first past KALMAN_MAX_RUNS.
_RUNS: OrderedDict = OrderedDict()
_MAX_RUNS = int(os.environ.get("KALMAN_MAX_RUNS", "1000"))


class _Registered(Mapping):
    """Authored specs by name, read through to the shared store on every access
    -- another process may have registered one a moment ago."""

    def __getitem__(self, name):
        rec = store.get_model(name)
        if rec is None:
            raise KeyError(name)
        return rec["spec"]

    def __contains__(self, name):
        return isinstance(name, str) and store.get_model(name) is not None

    def __iter__(self):
        return iter(store.model_names())

    def __len__(self):
        return len(store.model_names())


_MODELS = _Registered()

#: Names a registry entry may take: no path separators, no spaces, no surprises.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _check_name(name, what: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"{what} name {name!r} must be 1-64 characters of letters, digits, "
                         "'_', '-' or '.', starting with a letter or digit.")
    return name


# ======================================================================
# JSON (de)serialization — numpy <-> nested lists, JSON-safe scalars
# ======================================================================
def jsonify(obj):
    """Recursively convert numpy arrays/scalars to JSON-native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    return obj


def _arr(v):
    return None if v is None else np.asarray(v, dtype=float)


# ======================================================================
# Catalogs (auto-harvested from the app namespace + registries)
# ======================================================================
def _model_catalog() -> dict:
    """{name -> class} for every KalmanModel subclass in ``kalman.apps`` and
    ``kalman.framework.models`` (plus session-authored specs' base classes)."""
    from .. import apps
    from ..framework import models as fmodels
    out = {}
    for mod in (fmodels, apps):
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and issubclass(obj, KalmanModel):
                out.setdefault(name, obj)
    out.setdefault("KalmanModel", KalmanModel)
    # Models compiled into libKalman (or a plugin it has loaded), one
    # class per registered name. They are keyed by that name — lowercase-hyphenated
    # by convention, so a clash with a CamelCase Python class does not happen; if
    # one did, the Python class keeps the name.
    for name, cls in _native_classes().items():
        out.setdefault(name, cls)
    return out


def _native_classes() -> dict:
    try:
        from ..framework.models.native import native_model_classes
        return native_model_classes()
    except Exception:            # no cffi, no library: the catalogue is just Python
        return {}


_PLUGIN_ERRORS: dict = {}   # plugin name -> why it could not be loaded again here


def _sync_plugins():
    """Load every plugin the registry records and this process has not loaded.

    A plugin loaded through any process (or before a restart) is part of the
    shared catalog; this is what makes that true here. It is resolved again by
    *name* inside this process's allow-listed directories, never by the
    recorded path, so a record cannot point a server at a file outside them."""
    try:
        recorded = store.list_plugins()
    except Exception:
        return
    if not recorded:
        return
    from ..framework.native import core
    if not core.HAS_CXX_CORE:
        return
    loaded = set(core.plugins())
    for rec in recorded:
        if rec["name"] in _PLUGIN_ERRORS:
            continue
        try:
            path = core.resolve_plugin(rec["name"])
            if path not in loaded:
                core.load_plugin(path)
        except Exception as e:          # missing here; report, do not break the catalog
            _PLUGIN_ERRORS[rec["name"]] = str(e)


def native() -> dict:
    """What the C++ core contributes: the library actually loaded (or why not),
    the models it registers, and the plugins loaded so far."""
    from ..framework.native import core
    if not core.HAS_CXX_CORE:
        return {"library": None, "error": core.LOAD_ERROR, "models": [], "plugins": [],
                "plugin_dirs": core.plugin_dirs()}
    out = {"library": core.LIBRARY_PATH, "version": core.version(),
           "models": core.names(), "plugins": core.plugins(),
           "plugin_dirs": core.plugin_dirs()}
    if _PLUGIN_ERRORS:
        out["unavailable_plugins"] = dict(_PLUGIN_ERRORS)
    return out


def _controller_catalog() -> dict:
    from ..framework import controllers as ctl
    from ..framework.control import KalmanController
    return {n: o for n, o in vars(ctl).items()
            if inspect.isclass(o) and issubclass(o, KalmanController)}


def _describe_class(cls) -> dict:
    try:
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters if p not in ("self",)]
    except (TypeError, ValueError):
        params = []
    return {"name": cls.__name__,
            "description": getattr(cls, "DESCRIPTION", ""),
            "version": getattr(cls, "VERSION", ""),
            "params": params}


def catalog() -> dict:
    """Everything discoverable: models, engines, integrators, solvers,
    controllers, filters, providers — names + one-line metadata."""
    from ..framework.engines import available_engines
    from ..framework.integrators import available_integrators
    from ..framework.solvers import available_solvers
    _sync_plugins()
    prov = []
    try:
        from ..providers import PROVIDER_BY_KIND
        prov = [k.name.lower() for k in PROVIDER_BY_KIND]
    except Exception:
        pass
    return {
        "models": sorted(_model_catalog()),
        "controllers": sorted(_controller_catalog()),
        "engines": available_engines(),
        "integrators": available_integrators(),      # shipped kinds + registered ones
        "solvers": available_solvers(),
        "providers": prov,
        "authored": sorted(_MODELS),
        "sources": [s["name"] for s in store.list_sources()],
        "native": native(),
    }


def describe(name: str) -> dict:
    """Full record for a model or controller ``name`` — params, dims (if
    resolvable), states/observables, and the DEFAULT/ALLOWED engine/integrator/
    solver bindings."""
    _sync_plugins()
    rec = store.get_model(name) if isinstance(name, str) else None
    if rec is not None:
        return {"name": name, "kind": "authored", "base": rec["spec"].get("base", "matrices"),
                "description": rec["spec"].get("description", ""),
                "n": rec.get("n"), "m": rec.get("m"), "l": rec.get("l"),
                "owner": rec["owner"], "spec": rec["spec"], "backend": "spec"}
    models, controllers = _model_catalog(), _controller_catalog()
    if name in models:
        cls = models[name]
        d = _describe_class(cls)
        d["kind"] = "model"
        for attr in ("DEFAULT_ENGINE", "DEFAULT_INTEGRATOR", "DEFAULT_SOLVER"):
            v = getattr(cls, attr, None)
            d[attr.lower()] = v.name.lower() if v is not None else None
        for attr in ("ALLOWED_ENGINES", "ALLOWED_INTEGRATORS", "ALLOWED_SOLVERS"):
            v = getattr(cls, attr, None)
            d[attr.lower()] = sorted(k.name.lower() for k in v) if v else None
        if getattr(cls, "NATIVE_NAME", None):
            from ..framework.native import core
            info = core.describe(cls.NATIVE_NAME)
            d["backend"] = "native"
            d["plugin"] = info["plugin"]
            d["library"] = core.LIBRARY_PATH
            d.update({k: info[k] for k in ("n", "m", "l", "fs", "analytic")})
        else:
            d["backend"] = "python"
        return d
    if name in controllers:
        d = _describe_class(controllers[name]); d["kind"] = "controller"; return d
    raise KeyError(f"unknown catalog name {name!r}; see catalog().")


# ======================================================================
# Spec -> objects
# ======================================================================
def _kind(enum_cls, value):
    """Spec binding → a Kind member, or the name itself when it is a custom
    (registered) kind — the resolvers accept both and raise on a typo."""
    if value is None or not isinstance(value, str):
        return value
    try:
        return enum_cls[value.upper()]
    except KeyError:
        return value


#: Where each KalmanSpec key lives.  A spec is nested, and the overwhelmingly
#: common authoring mistake — especially from an agent writing JSON by hand — is
#: to put the matrices at the top level.  Mapping inner key -> owning section is
#: what lets :func:`_check_spec` say "F belongs in dynamics" instead of letting a
#: dimensionless model fail deep in the framework.
_SPEC_SECTIONS = {
    "F": "dynamics", "H": "dynamics", "B": "dynamics", "L": "dynamics",
    "M": "dynamics", "A": "dynamics", "Qc": "dynamics", "Lc": "dynamics",
    "params": "dynamics",
    "Q": "noise", "R": "noise",
    "x0": "initial", "P0": "initial",
    "states": "labels", "observables": "labels",
    "engine": "bindings", "integrator": "bindings", "solver": "bindings",
    "cost": "control", "Qcost": "control", "Rcost": "control", "x_ref": "control",
}

_SPEC_TOP_LEVEL = {"base", "name", "sensor_id", "fs", "dynamics", "noise",
                   "initial", "labels", "control", "bindings", "description"}

#: Names callers reach for that the spec calls something else.  ``model`` is the
#: big one: every other verb in this facade takes a model *name*, so a spec
#: argument invites ``{"model": "SomeModel"}`` — which is a spec with no base
#: and no dynamics, i.e. the dimensionless-model failure again.
_SPEC_ALIASES = {"model": "base", "class": "base", "model_name": "base",
                 "state_names": "labels.states", "observable_names": "labels.observables",
                 "measurements": "(pass measurements to run_filter's data argument)",
                 "data": "(pass measurements to run_filter's data argument)"}


def _check_spec(spec: dict):
    """Reject a malformed spec while the caller can still see their own mistake.

    Without this a spec whose matrices sat at the top level simply produced a
    model with no dimensions, and the failure surfaced from deep inside the
    framework as ``array must be (d,1) or (d,), got 0`` — a message that names
    nothing the caller wrote, and that reads like a broken server rather than a
    misplaced key.  (It cost a downstream integration an afternoon.)  Checked
    here, in :func:`make_model`, so every verb that builds a model —
    ``run_filter``, ``run_osse``, ``forecast``, ``register_model`` — gets the
    same guidance.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dict; see catalog()/describe() for the shape.")

    problems = [f"{k!r} belongs in {_SPEC_SECTIONS[k]}[{k!r}]"
                for k in sorted(spec) if k in _SPEC_SECTIONS]
    problems += [f"{k!r} is named {_SPEC_ALIASES[k]!r} in a spec"
                 for k in sorted(spec) if k in _SPEC_ALIASES]
    problems += [f"{k!r} is not a spec key" for k in sorted(spec)
                 if k not in _SPEC_TOP_LEVEL and k not in _SPEC_SECTIONS
                 and k not in _SPEC_ALIASES]
    if problems:
        raise ValueError(
            "malformed KalmanSpec — " + "; ".join(problems) + ". A spec is nested: "
            '{"base": "matrices", "fs": 1.0, "dynamics": {"F": .., "H": ..}, '
            '"noise": {"Q": .., "R": ..}, "initial": {"x0": .., "P0": ..}}; for a '
            'catalog model use {"base": "<ClassName>", "dynamics": {"params": {..}}}.')

    base = spec.get("base", "matrices")
    if base in ("matrices", "continuous"):
        dyn = spec.get("dynamics", {}) or {}
        if not any(dyn.get(k) is not None
                   for k in ("F", "H", "A", "Qc", "B", "L", "M", "Lc")):
            raise ValueError(
                f"a {base!r} spec carries its own dynamics, but none were given. "
                'Provide dynamics.F and dynamics.H (base "matrices"), or '
                'dynamics.A and dynamics.Qc (base "continuous"). To use a '
                "catalog model instead, set base to its class name — see "
                "catalog()['models'].")


def make_model(spec: dict) -> KalmanModel:
    """Instantiate a :class:`KalmanModel` from a :class:`KalmanSpec` dict."""
    spec = dict(spec or {})
    _check_spec(spec)
    base = spec.get("base", "matrices")
    if base in _MODELS:                            # authored spec referenced by name
        return make_model(_MODELS[base])
    sensor = spec.get("sensor_id", "api")
    fs = spec.get("fs", 1.0)
    noise = spec.get("noise", {}) or {}
    initial = spec.get("initial", {}) or {}
    binds = spec.get("bindings", {}) or {}
    kw = {}
    if "engine" in binds:
        pass  # engine is chosen by the filter, not the model ctor
    if binds.get("integrator"):
        kw["integrator"] = _kind(KalmanIntegratorKind, binds["integrator"])
    if binds.get("solver"):
        kw["solver"] = _kind(KalmanSolverKind, binds["solver"])
    if "x0" in initial:
        kw["x0"] = _arr(initial["x0"])
    if "P0" in initial:
        kw["P0"] = _arr(initial["P0"])

    if base in ("matrices", "continuous"):
        dyn = spec.get("dynamics", {}) or {}
        for k in ("F", "H", "B", "L", "M", "A", "Qc", "Lc"):
            if dyn.get(k) is not None:
                kw[k] = _arr(dyn[k])
        for k in ("Q", "R"):
            if noise.get(k) is not None:
                kw[k] = _arr(noise[k])
        return KalmanModel(sensor, fs=fs, **kw)

    cls = _model_catalog().get(base)
    if cls is None:
        _sync_plugins()                            # a plugin another process loaded
        cls = _model_catalog().get(base)
    if cls is None:
        raise ValueError(f"unknown model base {base!r}; see catalog()['models'].")
    params = dict((spec.get("dynamics", {}) or {}).get("params", {}) or {})
    # bind params against the class signature; pass sensor_id/fs when accepted
    try:
        plist = list(inspect.signature(cls.__init__).parameters)   # ordered
    except (TypeError, ValueError):
        plist = []
    accepted = set(plist)
    # A class taking **kwargs accepts x0/P0 even though they are not named parameters --
    # KalmanModel's base __init__ reads them straight out of kwargs, which is exactly how the
    # matrices path passes them. Filtering on the NAMED signature alone silently dropped
    # initial.x0 for every catalog model, and silently is the problem: the filter then starts
    # at the origin with a default covariance and produces confident nonsense rather than an
    # error. A satellite spec carrying its true initial state diverged by 892 km this way.
    try:
        takes_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                           for p in inspect.signature(cls.__init__).parameters.values())
    except (TypeError, ValueError):
        takes_var_kw = False
    is_native = hasattr(cls, "NATIVE_NAME")
    if is_native:
        # A C++ model has its own sample rate, noise and parameters: a spec that
        # does not give fs keeps the model's, and its noise is pushed into the
        # C++ instance (the catalogue's Python classes define their noise through
        # their own parameters instead).
        if "fs" in spec:
            params["fs"] = fs
        for k in ("Q", "R"):
            if noise.get(k) is not None:
                params[k] = _arr(noise[k])
        try:
            return cls(sensor, **{**kw, **params})
        except KeyError as e:
            raise ValueError(f"model {base!r}: {e.args[0] if e.args else e}") from None
    if "fs" in accepted and "fs" not in params:
        params["fs"] = fs
    for k, v in list(kw.items()):
        if k in accepted or takes_var_kw:
            params.setdefault(k, v)
    ordered = [p for p in plist if p != "self"]
    if ordered and ordered[0] in ("sensor_id", "name"):
        return cls(sensor, **params)
    return cls(**params)


def make_filter(spec: dict) -> KalmanFilter:
    """Build a :class:`KalmanFilter` (model + resolved engine) from a spec."""
    model = make_model(spec)
    binds = spec.get("bindings", {}) or {}
    engine_kind = _kind(KalmanEngineKind, binds.get("engine"))
    if str(binds.get("backend", "python")).lower() == "native":
        # The C++ engine behind the Python filter. Only a native (C++) model can be
        # driven this way -- NativeEngine says so itself if the model is Python.
        from ..framework.engines import NativeEngine
        kind = engine_kind or getattr(model, "DEFAULT_ENGINE", KalmanEngineKind.EXTENDED)
        return KalmanFilter(model, engine=NativeEngine(kind, **(binds.get("engine_params", {}) or {})))
    if engine_kind is not None:
        from ..framework.engines import resolve_engine
        return KalmanFilter(model, engine=resolve_engine(engine_kind))
    return KalmanFilter(model)


# ======================================================================
# Execute
# ======================================================================
def _stash(record, model, spec, truth=None, measurements=None) -> str:
    """Store a run for later diagnostics/plotting.

    ``measurements`` is kept SEPARATELY from ``record`` on purpose: a filtered
    record's ``zk`` holds the filter's own output, not the data that went in, so
    a diagnostic that reaches for ``record.zk`` ends up re-filtering estimates
    (see :func:`diagnostics`)."""
    rid = f"run-{uuid.uuid4().hex[:12]}"   # unique across processes and restarts
    while len(_RUNS) >= max(_MAX_RUNS, 1):
        _RUNS.popitem(last=False)
    _RUNS[rid] = {"record": record, "model": model, "spec": spec, "truth": truth,
                  "measurements": None if measurements is None
                                  else np.asarray(measurements, dtype=float)}
    return rid


def sample(spec: dict, duration: float, **kw) -> dict:
    """Generate synthetic truth + measurements; returns ``{run_id, time, xk, zk}``."""
    model = make_model(spec)
    rec = model.sample(duration, **kw)
    rid = _stash(rec, model, spec, truth=rec)
    return {"run_id": rid, "time": jsonify(rec.time), "xk": jsonify(rec.xk), "zk": jsonify(rec.zk)}


def run_filter(spec: dict, data=None, *, smooth: bool = False, source: str | None = None,
               query: dict | None = None, credential: str | None = None, **kw) -> dict:
    """Filter ``data`` (measurements, list of lists) through a spec's filter.
    Returns ``{run_id, xk, Pk}`` (posterior mean + covariance).

    Give ``source`` (a registered source name) instead of ``data`` to measure
    and filter in one call: the measurements are fetched first, with ``query``,
    under the ``send_measurement`` scope."""
    if source is not None:
        if data is not None:
            raise ValueError("give data or source, not both.")
        data = fetch({"source": source}, query, credential=credential)["zk"]
    if data is None:
        raise ValueError("run_filter needs measurements: data=[[...], ...] or source=<name>.")
    filt = make_filter(spec)
    zk = np.asarray(data, dtype=float)
    if zk.size == 0:
        raise ValueError("no measurements to filter.")
    rec = filt.filter(zk, smooth=smooth, **kw)
    rid = _stash(rec, filt.model, spec, measurements=zk)
    return {"run_id": rid, "xk": jsonify(rec.xk), "Pk": jsonify(rec.Pk),
            "time": jsonify(rec.time)}


def run_osse(spec: dict, *, n_cycles: int = 40, n_members: int = 16,
             inflation: float = 1.06, **kw) -> dict:
    """Twin-experiment OSSE for a field/flow spec (EnKF vs free run).
    Returns the per-cycle RMSE/spread histories."""
    from ..apps.weather import TwinExperiment
    model = make_model(spec)
    if not getattr(model, "_clim", None):
        model.climatology()
    osse = TwinExperiment(model, n_cycles=n_cycles, n_members=n_members)
    osse.add_method("enkf", inflation=inflation)
    osse.add_free_run()
    hist = osse.run(verbose=False, **kw)
    return {"hist": jsonify(hist)}


def run_control(model_spec: dict, controller_spec: dict, duration: float) -> dict:
    """Close the loop: run ``controller`` over the model for ``duration``.
    Returns ``{run_id, xk, cost}`` (state trace + total control cost)."""
    model = make_model(model_spec)
    cs = dict(controller_spec or {})
    cls = _controller_catalog().get(_ctrl_name(cs.get("controller")))
    if cls is None:
        raise ValueError(f"unknown controller {cs.get('controller')!r}.")
    cost = cs.get("cost", {}) or {}
    ctrl = cls(model,
               Qcost=_arr(cost.get("Qcost")), Rcost=_arr(cost.get("Rcost")),
               x_ref=_arr(cost.get("x_ref")), **(cs.get("params", {}) or {}))
    if hasattr(ctrl, "fit"):          # LQR/Neural controllers train before use
        ctrl.fit()
    filt = KalmanFilter(model)
    truth = model.sample(duration)
    rec = filt.forward(truth.zk, uk=ctrl)
    rid = _stash(rec, model, model_spec)
    total = float(sum(c for _, c in ctrl.cost_log))
    return {"run_id": rid, "xk": jsonify(rec.xk), "cost": total,
            "cost_log": jsonify([c for _, c in ctrl.cost_log])}


def _ctrl_name(key):
    if key is None:
        return None
    for n in _controller_catalog():
        if n.lower().startswith(str(key).lower()):
            return n
    return key


def diagnostics(run_id: str) -> dict:
    """Filter-consistency diagnostics (NIS + innovation) for a stored run.

    NIS answers the only question a filter can ask without ground truth: are the
    innovations the size the claimed noise says they should be (≈ m on average,
    χ²(m))?  That makes it the measurement contract — and therefore a diagnostic
    that must be fed the MEASUREMENTS.

    It previously reached for ``record.zk``, which on a filtered run holds the
    filter's own output rather than its input; the pass then re-filtered
    estimates and reported NIS ≈ 0.1 instead of ≈ 1 — a filter flagged as
    over-conservative when it was in fact consistent, and a check that failed
    quietly in the direction of looking fine.  Stored measurements are used when
    present, and a run that has none says so rather than silently diagnosing
    the wrong array."""
    r = _RUNS.get(run_id)
    if r is None:
        raise KeyError(f"unknown run_id {run_id!r}.")

    zk = r.get("measurements")
    if zk is None:
        record = r["record"]
        zk = getattr(record, "zk", None)
        # A sample() run IS its own measurement set; a filtered run is not.
        if zk is None or r.get("truth") is not record:
            raise ValueError(
                f"run {run_id!r} carries no stored measurements to diagnose; "
                "re-run it through run_filter (which now keeps them) or pass a "
                "sample() run.")
    filt = KalmanFilter(r["model"])
    filt.forward(np.asarray(zk, dtype=float), record=False)
    nu, S = filt.innovations()
    return {"nis": jsonify(filt.nis()), "innovation": jsonify(nu),
            "innovation_covariance": jsonify(S)}


def render_plot(run_id: str) -> bytes:
    """Render a stored run's signature figure to PNG bytes (base64-encode for JSON).

    The default figure carries a measurement-error panel that needs a truth
    trajectory, which only a :func:`sample` run has; for a filtered run the panel
    is dropped (``error=False``) rather than raising.
    """
    import io, matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    r = _RUNS.get(run_id)
    if r is None:
        raise KeyError(f"unknown run_id {run_id!r}.")
    model, record, truth = r["model"], r["record"], r.get("truth")
    try:
        # A stored run carries no separate ground truth (except a sample() run,
        # where the record *is* the truth), so drop the measurement-error panel
        # rather than raise.
        fig = (model.plot(record, sample=truth) if truth is not None and truth is not record
               else model.plot(record, error=False))
    except TypeError:
        fig = model.plot(record)      # an app whose plot() takes neither keyword
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def forecast(spec: dict, x0, P0=None, steps: int = 1, **kw) -> dict:
    """Forecast ahead from ``x0``/``P0`` for ``steps``.

    Uses the model's ``forecast_cone`` when it defines one (the storm family
    returns a track cone: ``{lon, lat, I, I_sigma, ellipses}``); otherwise the
    state and its covariance are propagated with the model's own ``f``/``F``, so
    every model answers something rather than only the storms.
    """
    model = make_model(spec)
    x = _arr(x0).reshape(model.n, 1)
    P = _arr(P0) if P0 is not None else np.asarray(model.P0, dtype=float)
    steps = int(steps)

    cone = getattr(model, "forecast_cone", None)
    if callable(cone):
        return jsonify(cone(x, P, steps, **kw))

    Q = model.Q()
    xs, Ps = [], []
    for k in range(steps):
        F = model.F(x=x, t=None)
        x = np.asarray(model.f(x), dtype=float).reshape(model.n, 1)
        P = F @ P @ F.T + Q
        xs.append(x); Ps.append(P)
    return {"xk": jsonify(xs), "Pk": jsonify(Ps)}


#: Provider kinds whose configuration reaches outside the server's own data: a
#: file path on the server, or a URL the server calls with ITS outbound secrets.
#: Given raw in a request, they would let any caller read server files or send
#: the server's credentials to a host of the caller's choice. So they take the
#: ``admin`` scope raw, and everybody else uses a named source an admin set up.
_PRIVILEGED_KINDS = ("file", "remote", "sensor")


def fetch(provider_spec, query: dict = None, *, credential: str | None = None) -> dict:
    """Pull data and return its JSON payload; requires ``send_measurement``.

    ``provider_spec`` is a registered source -- ``{"source": "<name>"}`` or just
    the name -- or a raw provider ``{"kind": "synthetic"|..., "init": {...}}``.
    Raw ``file``/``remote``/``sensor`` providers also need ``admin`` (see
    ``_PRIVILEGED_KINDS``). The result feeds straight into
    :func:`run_filter`/:func:`run_osse`."""
    from .auth import authorize
    authorize(credential, "send_measurement")
    if isinstance(provider_spec, str):
        provider_spec = {"source": provider_spec}
    spec = dict(provider_spec or {})
    if spec.get("source") is not None:
        return _fetch_source(spec["source"], dict(query or {}))
    from ..providers import resolve_provider
    if str(spec.get("kind", "")).lower() in _PRIVILEGED_KINDS:
        authorize(credential, "admin")
    prov = resolve_provider(spec.get("kind"), **(spec.get("init", {}) or {}))
    return jsonify(prov.fetch(**(query or {})))


def _fetch_source(name: str, query: dict) -> dict:
    rec = store.get_source(name)
    if rec is None:
        raise KeyError(f"no source named {name!r}; see sources().")
    if rec["kind"] == "push":
        since = int(query.pop("since", 0) or 0)
        limit = int(query.pop("limit", 10000) or 10000)
        if query:
            raise ValueError(f"a push source takes only since= and limit=, not {sorted(query)}.")
        rows = store.read_measurements(name, since, limit)
        out = {"zk": [r["row"] for r in rows],
               "meta": {"provider": "push", "source": name,
                        "first": rows[0]["seq"] if rows else None,
                        "last": rows[-1]["seq"] if rows else None,
                        "next": rows[-1]["seq"] if rows else since}}
        if rows and all(r["t"] is not None for r in rows):
            out["time"] = [r["t"] for r in rows]
        return out
    # The source fixes where the data comes from; the query may only narrow it.
    fixed = set(rec.get("init") or {}) | {"path", "source"}
    if rec["kind"] == "remote":
        fixed.discard("path")                 # a sub-path under the fixed base_url
    clash = sorted(k for k in query if k in fixed)
    if clash:
        raise ValueError(f"source {name!r} fixes {clash}; a query cannot change them.")
    from ..providers import resolve_provider
    prov = resolve_provider(rec["kind"], **(rec.get("init") or {}))
    out = jsonify(prov.fetch(**query))
    if isinstance(out, dict):
        out.setdefault("meta", {})["source"] = name
    return out


def load_plugin(name: str, *, credential: str | None = None) -> dict:
    """Load a compiled model library into the C++ factory and return what it added.

    ``name`` is a plugin *name* resolved inside the allow-listed directories
    (``KALMAN_PLUGIN_PATH``, default: this checkout's ``var/install/lib`` and
    ``var/build/lib``); a path is accepted only when it already lies inside one
    of them.  Loading executes code, so this is gated by its own scope,
    ``load_plugins`` — never by ``write_models``, which only ever writes data.
    """
    from .auth import authorize, identity
    authorize(credential, "load_plugins")
    from ..framework.native import core
    core.require()
    path = core.resolve_plugin(name)
    added = core.load_plugin(path)
    # Record it by the name it was asked for, so every process and every restart
    # loads it again (see _sync_plugins) and subscribers hear "plugin.loaded".
    key = os.path.splitext(os.path.basename(name))[0]
    store.put_plugin(key, path, added, identity(credential))
    _PLUGIN_ERRORS.pop(key, None)
    return {"plugin": path, "models": added, "native": sorted(core.names())}


def plugins() -> dict:
    """Plugins recorded in the shared registry, and which loaded here."""
    _sync_plugins()
    return {"plugins": store.list_plugins(), "unavailable": dict(_PLUGIN_ERRORS)}


# ======================================================================
# The shared registry: models, sources, pushed measurements, events
# ======================================================================
def models() -> dict:
    """Every authored model in the registry, with its owner and dimensions."""
    return {"models": store.list_models()}


def delete_model(name: str, *, credential: str | None = None) -> dict:
    """Remove an authored model; ``write_models``, and its owner or an admin."""
    from .auth import authorize, identity, require_owner
    authorize(credential, "write_models")
    rec = store.delete_model(name, by=identity(credential),
                             may_delete=lambda r: require_owner(r, credential, f"model {name!r}"))
    return {"deleted": rec["name"]}


def register_source(name: str, provider: dict, *, replace: bool = False,
                    credential: str | None = None) -> dict:
    """Set up a named measurement source that everybody can fetch by name.

    ``provider`` is ``{"kind": "push", "m": <width>?, "description": ..}`` --
    a buffer that devices, simulations or game engines write rows into with
    :func:`push_measurements` -- or a provider ``{"kind": "<kind>", "init":
    {...}}``. A ``file``/``remote``/``sensor`` source reads server files or
    uses the server's outbound credentials, so it needs ``admin``; the others
    need ``write_models``."""
    from .auth import authorize, identity, require_owner
    _check_name(name, "source")
    prov = dict(provider or {})
    kind = str(prov.get("kind", "")).lower()
    if not kind:
        raise ValueError('a source needs a kind: {"kind": "push"} or a provider kind.')
    authorize(credential, "admin" if kind in _PRIVILEGED_KINDS else "write_models")
    if kind == "push":
        m = prov.get("m")
        if m is not None and (not isinstance(m, int) or m < 1):
            raise ValueError("a push source's m (the row width) must be a positive integer.")
        spec = {"kind": "push", "m": m, "description": str(prov.get("description", ""))}
    else:
        from ..providers import resolve_provider
        init = dict(prov.get("init") or {})
        resolve_provider(kind, **init)        # fails now, not on the first fetch
        spec = {"kind": kind, "init": init, "description": str(prov.get("description", ""))}
    rec, action = store.put_source(
        name, spec, identity(credential), replace=replace,
        may_replace=lambda r: require_owner(r, credential, f"source {name!r}"))
    return {**rec, "action": action}


def sources() -> dict:
    """Registered measurement sources. A remote source's ``init`` is shown,
    but it holds only the *name* of an outbound credential, never the secret."""
    return {"sources": store.list_sources()}


def describe_source(name: str) -> dict:
    for rec in store.list_sources():
        if rec["name"] == name:
            return rec
    raise KeyError(f"no source named {name!r}; see sources().")


def delete_source(name: str, *, credential: str | None = None) -> dict:
    """Remove a source and its buffered rows; owner or admin."""
    from .auth import authorize, identity, require_owner
    authorize(credential, "write_models")
    rec = store.delete_source(name, by=identity(credential),
                              may_delete=lambda r: require_owner(r, credential, f"source {name!r}"))
    return {"deleted": rec["name"]}


#: Rows one push may carry; a bigger batch goes in several requests.
MAX_PUSH_ROWS = 10000


def push_measurements(name: str, rows, time=None, *, credential: str | None = None) -> dict:
    """Append measurement rows to a ``push`` source; requires ``send_measurement``.

    ``rows`` is a list of rows (one measurement vector each) or one row;
    ``time`` an optional timestamp per row. Subscribers get a
    ``measurement.received`` event naming the new sequence numbers."""
    from .auth import authorize, identity
    authorize(credential, "send_measurement")
    rec = store.get_source(name)
    if rec is None:
        raise KeyError(f"no source named {name!r}; register_source first.")
    if rec["kind"] != "push":
        raise ValueError(f"source {name!r} is a {rec['kind']!r} source; only a push "
                         "source takes measurements.")
    try:
        z = np.asarray(rows, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("rows must be numbers: [[z1, z2, ...], ...].")
    if z.ndim == 1:
        z = z.reshape(1, -1)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] == 0:
        raise ValueError(f"rows must be a non-empty list of rows; got shape {z.shape}.")
    if z.shape[0] > MAX_PUSH_ROWS:
        raise ValueError(f"at most {MAX_PUSH_ROWS} rows per push; got {z.shape[0]}.")
    if not np.all(np.isfinite(z)):
        raise ValueError("rows must be finite (no NaN or infinity).")
    if rec.get("m") is not None and z.shape[1] != rec["m"]:
        raise ValueError(f"source {name!r} takes rows of width {rec['m']}; got {z.shape[1]}.")
    times = None
    if time is not None:
        times = [float(t) for t in np.asarray(time, dtype=float).reshape(-1)]
        if len(times) != z.shape[0]:
            raise ValueError(f"{len(times)} timestamps for {z.shape[0]} rows.")
    return store.append_measurements(name, z.tolist(), times, identity(credential))


def events(since: int = 0, limit: int = 100) -> dict:
    """Registry events after ``since``: ``{"events": [...], "last": <id>}``.
    Pass ``last`` back as ``since`` to read on; ``since=-1`` returns no events
    and the newest id, the place to start listening from.  ``GET
    /events/stream`` pushes the same events as Server-Sent Events."""
    limit = max(1, min(int(limit), 1000))
    if int(since) < 0:                      # "from now": the cursor a new subscriber starts at
        return {"events": [], "last": store.last_event()}
    evs = store.events(int(since), limit)
    return {"events": evs, "last": evs[-1]["id"] if evs else max(int(since), 0)}


def whoami(credential: str | None = None) -> dict:
    """Who a token says the caller is: ``{"auth", "user", "scopes", "exp"}``.
    Verifies the token, but checks no scope."""
    from .auth import claims, enabled
    c = claims(credential)
    if c is None:
        return {"auth": enabled(), "user": None, "scopes": ["admin"], "exp": None}
    return {"auth": True, "user": c.get("sub"), "scopes": c.get("scopes", []),
            "exp": c.get("exp")}


def providers() -> dict:
    """Every provider with what it can actually supply — the data half of
    :func:`catalog`.

    ``catalog()['providers']`` answers "which providers exist"; this answers
    "and what measurements can I get from them", which is the question a caller
    has to answer before ``fetch`` can be written. Models were already
    introspectable through :func:`describe`; sources now are too."""
    from ..providers import PROVIDER_BY_KIND
    out = {}
    for kind, cls in PROVIDER_BY_KIND.items():
        record = cls.describe()
        out[getattr(kind, "name", str(kind)).lower()] = record
    return {"providers": out}


def describe_provider(name: str) -> dict:
    """Full record for one provider: measurements, query keys, prerequisites.

    ``name`` is the provider kind as it appears in ``catalog()['providers']``
    (``synthetic``, ``era5``, ``turbulence``, ``file``, ``sensor``, ``remote``)."""
    from ..providers import PROVIDER_BY_KIND
    key = str(name).strip().lower()
    for kind, cls in PROVIDER_BY_KIND.items():
        if getattr(kind, "name", str(kind)).lower() == key:
            return cls.describe()
    known = sorted(getattr(k, "name", str(k)).lower() for k in PROVIDER_BY_KIND)
    raise ValueError(f"unknown provider {name!r}; known providers: {known}.")


# Authoring verbs live in spec.py / scaffold.py; re-exported here.
from .spec import register_model, save_spec, load_spec  # noqa: E402
from .scaffold import scaffold, write_scaffold          # noqa: E402

__all__ = [
    "catalog", "describe", "make_model", "make_filter",
    "sample", "run_filter", "forecast", "run_osse", "run_control",
    "diagnostics", "render_plot",
    "fetch", "providers", "describe_provider", "load_plugin", "plugins",
    "register_model", "save_spec", "load_spec", "scaffold", "write_scaffold",
    "models", "delete_model", "register_source", "sources", "describe_source",
    "delete_source", "push_measurements", "events", "whoami",
    "ConflictError", "jsonify",
]

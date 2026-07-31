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
import numpy as np

from ..framework.model import KalmanModel
from ..framework.filter import KalmanFilter
from ..framework.constants import (KalmanEngineKind, KalmanIntegratorKind,
                                   KalmanSolverKind)

_RUNS: dict = {}          # run_id -> {"record":.., "model":.., "spec":..}
_SESSION_MODELS: dict = {}  # authored specs registered this session (name -> spec)


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
    from ..framework.engines import ENGINE_BY_KIND
    from ..framework.integrators import INTEGRATOR_BY_KIND
    from ..framework.solvers import SOLVER_BY_KIND
    prov = []
    try:
        from ..providers import PROVIDER_BY_KIND
        prov = [k.name.lower() for k in PROVIDER_BY_KIND]
    except Exception:
        pass
    return {
        "models": sorted(_model_catalog()),
        "controllers": sorted(_controller_catalog()),
        "engines": [k.name.lower() for k in ENGINE_BY_KIND],
        "integrators": [k.name.lower() for k in INTEGRATOR_BY_KIND],
        "solvers": [k.name.lower() for k in SOLVER_BY_KIND],
        "providers": prov,
        "authored": sorted(_SESSION_MODELS),
    }


def describe(name: str) -> dict:
    """Full record for a model or controller ``name`` — params, dims (if
    resolvable), states/observables, and the DEFAULT/ALLOWED engine/integrator/
    solver bindings."""
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
        return d
    if name in controllers:
        d = _describe_class(controllers[name]); d["kind"] = "controller"; return d
    raise KeyError(f"unknown catalog name {name!r}; see catalog().")


# ======================================================================
# Spec -> objects
# ======================================================================
def _kind(enum_cls, value):
    if value is None:
        return None
    return enum_cls[value.upper()] if isinstance(value, str) else value


def make_model(spec: dict) -> KalmanModel:
    """Instantiate a :class:`KalmanModel` from a :class:`KalmanSpec` dict."""
    spec = dict(spec or {})
    base = spec.get("base", "matrices")
    if base in _SESSION_MODELS:                    # authored spec referenced by name
        return make_model(_SESSION_MODELS[base])
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
        raise ValueError(f"unknown model base {base!r}; see catalog()['models'].")
    params = dict((spec.get("dynamics", {}) or {}).get("params", {}) or {})
    # bind params against the class signature; pass sensor_id/fs when accepted
    try:
        plist = list(inspect.signature(cls.__init__).parameters)   # ordered
    except (TypeError, ValueError):
        plist = []
    accepted = set(plist)
    if "fs" in accepted and "fs" not in params:
        params["fs"] = fs
    for k, v in list(kw.items()):
        if k in accepted:
            params.setdefault(k, v)
    ordered = [p for p in plist if p != "self"]
    if ordered and ordered[0] in ("sensor_id", "name"):
        return cls(sensor, **params)
    return cls(**params)


def make_filter(spec: dict) -> KalmanFilter:
    """Build a :class:`KalmanFilter` (model + resolved engine) from a spec."""
    model = make_model(spec)
    engine_kind = _kind(KalmanEngineKind, (spec.get("bindings", {}) or {}).get("engine"))
    if engine_kind is not None:
        from ..framework.engines import resolve_engine
        return KalmanFilter(model, engine=resolve_engine(engine_kind))
    return KalmanFilter(model)


# ======================================================================
# Execute
# ======================================================================
def _stash(record, model, spec) -> str:
    rid = f"run{len(_RUNS)}"
    _RUNS[rid] = {"record": record, "model": model, "spec": spec}
    return rid


def sample(spec: dict, duration: float, **kw) -> dict:
    """Generate synthetic truth + measurements; returns ``{run_id, time, xk, zk}``."""
    model = make_model(spec)
    rec = model.sample(duration, **kw)
    rid = _stash(rec, model, spec)
    return {"run_id": rid, "time": jsonify(rec.time), "xk": jsonify(rec.xk), "zk": jsonify(rec.zk)}


def run_filter(spec: dict, data, *, smooth: bool = False, **kw) -> dict:
    """Filter ``data`` (measurements, list of lists) through a spec's filter.
    Returns ``{run_id, xk, Pk}`` (posterior mean + covariance)."""
    filt = make_filter(spec)
    zk = np.asarray(data, dtype=float)
    rec = filt.filter(zk, smooth=smooth, **kw)
    rid = _stash(rec, filt.model, spec)
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
    """Filter-consistency diagnostics (NIS + innovation) for a stored run."""
    r = _RUNS.get(run_id)
    if r is None:
        raise KeyError(f"unknown run_id {run_id!r}.")
    filt = KalmanFilter(r["model"])
    filt.forward(np.asarray(r["record"].zk, dtype=float), record=False)
    return {"nis": jsonify(filt.nis()), "innovation": jsonify(filt.innovations()[0])}


def render_plot(run_id: str) -> bytes:
    """Render a stored run's signature figure to PNG bytes (base64-encode for JSON)."""
    import io, matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    r = _RUNS.get(run_id)
    if r is None:
        raise KeyError(f"unknown run_id {run_id!r}.")
    fig = r["model"].plot(r["record"])
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fetch(provider_spec: dict, query: dict = None, *, credential: str | None = None) -> dict:
    """Pull data from a provider (``{"kind": "synthetic"|"file"|..., "init": {...}}``)
    and return its JSON payload.  Requires the ``send_measurement`` scope.  The
    result feeds straight into :func:`run_filter`/:func:`run_osse`."""
    from .auth import authorize
    authorize(credential, "send_measurement")
    from ..providers import resolve_provider
    spec = dict(provider_spec or {})
    prov = resolve_provider(spec.get("kind"), **(spec.get("init", {}) or {}))
    return jsonify(prov.fetch(**(query or {})))


# Authoring verbs live in spec.py / scaffold.py; re-exported here.
from .spec import register_model, save_spec, load_spec  # noqa: E402
from .scaffold import scaffold, write_scaffold          # noqa: E402

__all__ = [
    "catalog", "describe", "make_model", "make_filter",
    "sample", "run_filter", "run_osse", "run_control", "diagnostics", "render_plot",
    "fetch",
    "register_model", "save_spec", "load_spec", "scaffold", "write_scaffold",
    "jsonify",
]

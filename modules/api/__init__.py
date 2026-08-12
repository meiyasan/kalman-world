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
    from ..framework.engines import available_engines
    from ..framework.integrators import available_integrators
    from ..framework.solvers import available_solvers
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
    engine_kind = _kind(KalmanEngineKind, (spec.get("bindings", {}) or {}).get("engine"))
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
    rid = f"run{len(_RUNS)}"
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


def run_filter(spec: dict, data, *, smooth: bool = False, **kw) -> dict:
    """Filter ``data`` (measurements, list of lists) through a spec's filter.
    Returns ``{run_id, xk, Pk}`` (posterior mean + covariance)."""
    filt = make_filter(spec)
    zk = np.asarray(data, dtype=float)
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
    "fetch", "providers", "describe_provider",
    "register_model", "save_spec", "load_spec", "scaffold", "write_scaffold",
    "jsonify",
]

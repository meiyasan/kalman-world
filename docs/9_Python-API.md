# Python API — the `kalman.api` facade

`kalman.api` is one JSON-friendly surface over the whole framework, for humans,
scripts and agents. It *harvests* the existing metadata and registries rather
than maintaining a parallel description layer, and speaks plain dicts and lists
(NumPy arrays ↔ nested lists), so the MCP and REST transports are thin adapters
with no logic of their own.

Use it directly from Python; use [REST](10_REST-API.md) or
[MCP](11_MCP-Server.md) to reach the same verbs from elsewhere.

---

## The verbs

| Verb | Scope | Returns |
|---|---|---|
| `catalog()` | `read` | every discoverable name: models, engines, integrators, solvers, controllers, providers, authored specs |
| `describe(name)` | `read` | full record for one model or controller |
| `make_model(spec)` | — | a live `KalmanModel` |
| `make_filter(spec)` | — | a live `KalmanFilter` with the engine resolved |
| `sample(spec, duration)` | — | `{run_id, time, xk, zk}` synthetic truth + measurements |
| `run_filter(spec, data, smooth=False)` | `send_measurement` | `{run_id, xk, Pk, time}` |
| `forecast(spec, x0, P0, steps)` | `send_measurement` | the model forecast cone, else the propagated `{xk, Pk}` |
| `run_osse(spec, n_cycles, n_members, inflation)` | `send_measurement` | per-cycle RMSE/spread histories |
| `run_control(model_spec, controller_spec, duration)` | `send_measurement` | `{run_id, xk, cost, cost_log}` |
| `fetch(provider_spec, query, credential=…)` | `send_measurement` | a provider payload |
| `diagnostics(run_id)` | `read` | `{nis, innovation}` for a stored run |
| `render_plot(run_id)` | `read` | PNG bytes |
| `register_model(spec, name)` | `write_models` | validates, smoke-tests and registers an authored spec |
| `load_plugin(name)` | `load_plugins` | loads a compiled (C++) model library from the allow-listed plugin dirs; its models join the catalogue |
| `save_spec(name, directory)` / `load_spec(path)` | `write_models` | JSON persistence |
| `scaffold(spec, class_name)` / `write_scaffold(…)` | `write_models` | reviewable Python source |
| `jsonify(obj)` | — | NumPy → JSON-native, recursively |

Scopes are enforced only when authentication is enabled — in-process Python use
on your own machine is unrestricted. See
[Authentication](12_Authentication.md).

---

## `KalmanSpec` — describing a model as data

Everything in the facade is driven by one dict. No code is executed: only names
already in the catalog, or the two declarative bases, are accepted.

```python
{
  "base": "matrices" | "continuous" | "<CatalogClassName>",
  "name": "my_model",                  # optional, used by register_model
  "sensor_id": "api",
  "fs": 100.0,

  "dynamics": {"F": …, "H": …, "B": …, "L": …, "M": …}    # base="matrices"
            | {"A": …, "Qc": …, "Lc": …, "H": …}          # base="continuous"
            | {"params": {…}},                            # base=<CatalogClassName>

  "noise":   {"Q": …, "R": …},
  "initial": {"x0": …, "P0": …},
  "labels":  {"states": [...], "observables": [...]},
  "control": {"B": …, "cost": {"Qcost": …, "Rcost": …, "x_ref": …}},
  "bindings": {"engine": "linear", "integrator": "rk4", "solver": "spectral"}
}
```

All matrices are nested lists (or NumPy arrays, when calling from Python).

### The three bases

**`matrices`** — discrete matrices, the most direct form:

```python
dt = 0.01
spec = {
    "base": "matrices", "sensor_id": "demo", "fs": 100.0,
    "dynamics": {"F": [[1, dt], [0, 1]], "H": [[1, 0]]},
    "noise":    {"Q": [[1e-6, 0], [0, 1e-4]], "R": [[0.05]]},
    "initial":  {"x0": [[0.0], [1.0]]},
    "bindings": {"engine": "linear"},
}
```

**`continuous`** — the continuous generator, discretized exactly (`expm` for
`F`, Van Loan for `Q`). Requires `fs`:

```python
spec = {
    "base": "continuous", "fs": 100.0,
    "dynamics": {"A": [[0, 1], [0, 0]], "Qc": [[0, 0], [0, 1e-2]], "H": [[1, 0]]},
    "noise":    {"R": [[0.05]]},
}
```

**A catalog class name** — parameterize a shipped model. `dynamics.params` is
bound against the class signature:

```python
spec = {"base": "BallisticModel", "sensor_id": "shot", "fs": 50.0,
        "dynamics": {"params": {"c_drag": 0.02}}}
```

`describe("BallisticModel")["params"]` tells you which parameters exist.

### Bindings

`bindings.backend: "native"` runs the C++ engine of that kind behind the Python
filter — only for a native (C++) model, see [C++ Library](15_Cxx-Library.md);
`bindings.engine_params` carries its constructor parameters.
`bindings.engine` is applied by `make_filter` (the engine belongs to the filter,
not the model); `integrator` and `solver` are passed to the model constructor.

Any name in `catalog()` works, including schemes added at runtime with
`register_integrator` / `register_solver`. An unrecognised name raises
`ValueError` rather than quietly running something else — see
[Integrators and Solvers](6_Integrators-and-Solvers.md).

---

## Worked session

```python
from kalman.api import catalog, describe, sample, run_filter, diagnostics, register_model

catalog()["models"][:3]           # ['AnalyticOscillatorModel', 'BallisticLinearModel', …]
describe("StormModel")            # params, version, engine/integrator/solver bindings

dt = 0.01
spec = {
    "base": "matrices", "sensor_id": "demo", "fs": 100.0,
    "dynamics": {"F": [[1, dt], [0, 1]], "H": [[1, 0]]},
    "noise":    {"Q": [[1e-6, 0], [0, 1e-4]], "R": [[0.05]]},
    "initial":  {"x0": [[0.0], [1.0]]},
    "bindings": {"engine": "linear"},
}

s   = sample(spec, 2.0)               # {'run_id', 'time', 'xk', 'zk'}  — 200 steps
out = run_filter(spec, s["zk"])       # {'run_id', 'xk', 'Pk', 'time'}
diagnostics(out["run_id"])            # {'nis': [...], 'innovation': [...]}

register_model(spec, "demo_cv")       # {'name': 'demo_cv', 'n': 2, 'm': 1, 'l': 0}
catalog()["authored"]                 # ['demo_cv']
```

### Run IDs

`sample`, `run_filter` and `run_control` stash the record, model and spec under
a `run_id` (`"run0"`, `"run1"`, …), which `diagnostics()` and `render_plot()`
then take. The store is an **in-process dict**: IDs do not survive a restart and
are not shared between workers. If you run REST under multiple uvicorn workers,
a `run_id` minted by one worker is invisible to the others — keep it to one
worker, or re-send the data.

---

## Authoring: spec → registered model → reviewable code

`register_model` puts a spec through five gates:

1. **structural validation** — known base; `Q`/`R` square, symmetric and PSD;
   declarative `dynamics` values must be numeric arrays, never code;
2. **framework validation** — instantiation runs `_check_shapes`, so dimensions
   must be mutually consistent;
3. **a smoke gate** — `f(x0)` and `h(x0)` must evaluate, and a couple of steps
   must actually sample;
4. **session registration** — `catalog()` and `make_model` now see the name, and
   an authored spec is indistinguishable from a built-in;
5. optional **persistence** — `save_spec(name)` writes `./specs/<name>.json`,
   reloadable with `load_spec(path)`.

```python
from kalman.api import register_model, save_spec, load_spec, scaffold, write_scaffold

register_model(spec, "demo_cv")
save_spec("demo_cv")                       # ./specs/demo_cv.json
load_spec("./specs/demo_cv.json")          # re-registers on load

src = scaffold(spec, "DemoCV")             # Python source, as a string
write_scaffold(spec, "var/staging/demo_cv.py", "DemoCV")
```

`scaffold` emits a `LinearKalmanModel`/`KalmanModel` subclass for the
`matrices`/`continuous` bases — the path from an authored spec to a real app,
with a **human as the commit gate**. It only ever *emits* source; it never
`exec`s it, never auto-imports it and never installs it. Parameterized catalog
models are already importable, so scaffolding them is rejected.

> There is deliberately **no `eval`/`exec` anywhere in `kalman.api`**. Specs are
> pure data. That is what makes it safe to expose the authoring verbs to an LLM
> over MCP.

## OSSE and forecasting

```python
from kalman.api import run_osse

run_osse({"base": "KolmogorovFlowModel", "dynamics": {"params": {}}},
         n_cycles=40, n_members=16, inflation=1.06)
# {"hist": {...}}  — per-cycle RMSE and spread for EnKF vs a free run
```

The twin experiment adds an EnKF method and a free run, and calls
`model.climatology()` first if the model has not been spun up.

`forecast` runs the model forward from a state with no measurements:

```python
from kalman.api import forecast

forecast(spec, x0=[[0.0], [1.0]], P0=None, steps=4)
# {"xk": [...],  # (steps, n, 1) propagated mean
#  "Pk": [...]}  # (steps, n, n) covariance, growing with the horizon
```

Where a model defines its own `forecast_cone` — the storm family does — that is
used instead, and you get the domain answer:

```python
forecast({"base": "StormModel", "fs": 1/3600.}, x0, steps=6)
# {"lon": …, "lat": …, "I": …, "I_sigma": …, "ellipses": …}
```

Everything else falls back to propagating `x` through `f` and `P` through
`F P Fᵀ + Q`, so every model answers something. Exposed as `POST /forecast` and
as the `forecast` MCP tool.

---

## Adding a verb

Add the function to `kalman/api/`, declare its scope, then expose it in
`kalman/mcp/server.py` and `kalman/rest/app.py` as a thin adapter. Never put
logic in a transport — that is the invariant that keeps the three front doors
behaving identically.

# Overview

The framework separates a Kalman problem into seven roles, each answering one
question. A role is an **abstract contract** in `kalman.framework`; concrete
implementations live outside it. Nothing in the design is Kalman-specific except
the engines — the same skeleton carries particle filters, ensemble assimilation
and combinatorial track finding.

---

## The seven roles

| Role | Question it answers | Contract | Concretes |
|---|---|---|---|
| **Model** | WHAT is the system? `f/h/Q/R` | `framework/model.py` | `framework/models/`, `apps/` |
| **Engine** (alias *Estimator*) | HOW to infer under uncertainty | `framework/engine.py` | `framework/engines/` |
| **Integrator** | HOW to step *time*: `x(t) → x(t+dt)` | `framework/integrator.py` | `framework/integrators/` |
| **Solver** | HOW to discretize *space* | `framework/solver.py` | `framework/solvers/` |
| **Filter** | WHEN — the forward/backward workflow | `framework/filter.py` | `framework/filters/` |
| **Provider** | WHERE the data comes from | `framework/provider.py` | `providers/`, `apps/<domain>/providers/` |
| **Controller / Router** | acting on, and choosing between, models | `framework/control.py`, `routers/` | `framework/controllers/`, `routers/` |

Dependencies point one way only: **Filter → Engine → Model**. A model never
imports an engine, a solver or a filter.

The Engine/Filter split is where the algorithm ends and the workflow begins: a
filter says *when* things run and what is kept, an engine says *how* the belief
moves. So the concrete filters are thin — `LinearKalmanFilter` is six lines —
and none of them implement `forward`/`backward` themselves. When one
predict/update step does not describe an algorithm at all, the engine takes over
the whole pass instead (`KalmanEngine.OWNS_PASS`); `CombinatorialKalmanEngine`
does this, consuming an entire detector event and returning finished tracks,
while its filter stays as thin as the rest.

### Integrator vs Solver

These are not synonyms. An *integrator* advances time (Euler / RK4 /
Dormand–Prince). A *solver* discretizes space (spectral / finite-difference) and
**composes** an integrator for its time march. For an ODE the two coincide —
`discretize` is the identity — which is why an ODE model binds only an
integrator, and a PDE model binds a solver. They are sibling packages; neither
nests inside the other. See [Integrators and Solvers](6_Integrators-and-Solvers.md).

---

## Operations: a model made from models

A bigger problem is not a new kind of filter. It is a bigger *model*, made from
the ones you have by an **operation**: a function that takes models and gives
back a model. The filter that runs the result is the ordinary one.

| Operation | What the new model is | State | Measurement |
|---|---|---|---|
| `model.add_sensor(sensor)` | the model read by one more sensor | unchanged | one more block of rows |
| `model.add_instrument(instrument)` | the same, for sensors grouped under one name with a rate and a calibration | unchanged | its sensors' rows |
| `model.joint({...})` | the model with some of its parameters estimated | state, then the parameters | unchanged |
| `model.lag(L)` | the model carrying its last `L` states | `L + 1` copies of the state | unchanged; a late sensor reads a past copy |
| `compose(a, b)` | two systems side by side | theirs, one after the other | stacked, or summed on one channel |
| `route(models, Pi)` | one system seen by several models, one per regime | shared | shared |

```python
from kalman.framework import KalmanSensor, compose

lens   = KalmanSensor("lens", G=[[1, 0]], R=0.01, delay=0.03)        # precise, and 0.03 s late
camera = compose(lens, name="camera").calibrate(gain=1.0, offset=0.0)  # an instrument, with a calibration

tracker = (cart.add_instrument(camera)                 # one more reading of the same state
               .lag(3)                                 # the last three states, for the late sensor
               .joint({"camera.offset": (1.0, 0.0)}))  # and the camera's offset estimated
record = tracker.filter(zk)                            # the ordinary filter: n = 9, m = 2
```

Each operation has three spellings that build the same object: the method
(`model.lag(3)`), the function (`lag(model, 3)`) and the class (`Lag(model, 3)`,
the one an application subclasses). C++ has the same names, `Kalman::Lag(model, 3)`
and `Kalman::Models::Lag`, and does the arithmetic; Python wraps it. Operations
nest, and the result keeps what its parts declared: models of matrices combine
into a model of matrices, which stays on the linear engine.

What other fields call these (sensor fusion, state augmentation, the
augmented-state filter, IMM) is in [The contract](19_Contract.md); how to use
each one is in [Models](4_Models.md#operations-a-model-made-from-models).
Runnable on a cart: [`examples/python/operations.py`](../examples/python/operations.py)
and [`examples/cxx/operations.cc`](../examples/cxx/operations.cc).

## Declared, not imported

The single most important convention: a model **declares** its collaborators by
enum, and the framework resolves them.

```python
class MyModel(KalmanModel):
    DEFAULT_ENGINE      = KalmanEngineKind.EXTENDED
    DEFAULT_INTEGRATOR  = KalmanIntegratorKind.RK4
    DEFAULT_SOLVER      = KalmanSolverKind.SPECTRAL
    ALLOWED_ENGINES     = {KalmanEngineKind.EXTENDED, KalmanEngineKind.UNSCENTED}
```

The kinds live in `framework/constants.py`; the registries (`ENGINE_BY_KIND`,
`INTEGRATOR_BY_KIND`, `SOLVER_BY_KIND`, `PROVIDER_BY_KIND`) and their
`resolve_*()` functions do the lookup. `KalmanFilter` validates that the engine
it resolved is in `model.ALLOWED_ENGINES` and raises `TypeError` otherwise.

The available kinds today:

| Registry | Members |
|---|---|
| `KalmanEngineKind` | `linear`, `extended`, `unscented`, `particle`, `combinatorial`, `ensemble` |
| `KalmanIntegratorKind` | `euler`, `rk4`, `dormand_prince`, `implicit_euler` *(reserved)* |
| `KalmanSolverKind` | `spectral`, `finite_difference`, `finite_volume` *(reserved)*, `finite_element` *(reserved)* |
| `KalmanProviderKind` | `synthetic`, `era5`, `turbulence`, `file`, `sensor`, `remote` |

*Reserved* members exist in the enum and the registry but raise
`NotImplementedError` when used — the registries are required to cover their
whole enum, and `tests/test_structure.py` enforces it.

---

## Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `raw` | `(T, M, 1)` | Raw input timeseries, before conditioning |
| `uk` | `(T, L, 1)` | Control input timeseries (or a callable controller) |
| `zk` | `(T, M, 1)` | Conditioned measurements, after `prepare_measurements` |
| `xk` | `(n, 1)` | State vector at one timestep |
| `xk_hat` | `(T, n, 1)` | Batch of state estimates (prior / posterior / smoothed) |
| `zk_hat` | `(T, M, 1)` | Predicted measurements in filter space, `h(xk_hat)` |
| `raw_hat` | `(T, M, 1)` | Predicted measurements in raw space, `finish_output(zk_hat)` |
| `n`, `m`, `l` | — | State, measurement and control dimensions |

---

## The pipeline

```
raw          raw input timeseries            (T, M, 1)
 │
 ▼  prepare_measurements(raw, uk) ──────────────────── raw → filter space
 │
zk          conditioned measurements         (T, M, 1)
 │
 ▼  ── filter loop: for z, u in zip(zk, uk) ─────────────────────────────
 │      predict:  x⁻, P⁻  ←  f(x, u),  F, Q
 │      update:   x⁺, P⁺  ←  x⁻ + K·(z − h(x⁻)),  H, R
 ▼
xk_hat      state estimates                  (T, n, 1)
 │
 ▼  adjust_state(xk_hat) ──────────────────── optional state-space transform
 ▼  h(xk_hat) ──────────────────────────── state → filter space
 │
zk_hat      predicted measurements           (T, M, 1)
 │
 ▼  finish_output(zk_hat) ────────────────── filter space → raw space
 │
raw_hat      predicted raw output             (T, M, 1)   mirrors raw
```

`prepare_measurements` and `finish_output` are a matched pair: if `prepare_measurements` whitens or
heterodynes the signal, `finish_output` maps predictions back so `raw_hat` is
directly comparable with `raw`. All three hooks default to identity.

---

## Package layout

```
modules/                        → the `kalman` package
  framework/                    contracts + bases + registries (the core)
    model.py engine.py integrator.py solver.py provider.py filter.py control.py
    models/ engines/ filters/ integrators/ solvers/ routers/ controllers/ io/ abstract/ native/
    sensors/                    generic sensors (position, Doppler, camera)
    filters/                    public filter names (KF/EKF/UKF/PF/EnKF/CKF)
  providers/                    generic data sources (synthetic/file/sensor/remote)
  api/                          JSON facade + KalmanSpec authoring + scoped-JWT auth
  mcp/  rest/                   transports over the facade (optional deps)
  adapters/                     specific hardware: instruments of generic sensors (netatmo)
  apps/                         domain applications
    gravitational/ gravwave  interferometer
    resonators/    oscillators  powergrids  suspensions
    weather/       flow  storms  geo  surrogate  data/  providers/
    aerospace/  astrophysics/  ballistics/  calorimeter/  fusion/  gps/  tracking/  vehicles/  vision/
  runtime/  helpers.py
tests/                          mirrors concerns, not files
share/notebooks/                executed, output-bearing narrative docs
```

**Families** (`gravitational`, `resonators`) group domains that share a
foundation or an application area. Each domain keeps its own `VARIANTS` dict of
model classes; cross-family imports are normal (`interferometer` uses
`resonators.oscillators`).

`apps/` contains packages only. Import from the canonical path
(`kalman.apps.weather.flow`) or from the stable `kalman.apps` surface — both
resolve to the same objects, and `from kalman.apps import X` is guaranteed to
keep working across reorganisations.

---

## One facade, three front doors

```
                    ┌──────────────┐
   Python  ───────▶ │              │
   REST /8000 ────▶ │  kalman.api  │ ──▶  framework  ──▶  apps
   MCP /8765  ───▶  │   (facade)   │
                    └──────────────┘
                           │
                     kalman.api.auth   ← scopes: read / send_measurement /
                                          write_models / admin
```

`kalman.api` harvests the existing metadata and registries rather than
maintaining a parallel description layer, and speaks plain dicts and lists
(NumPy arrays ↔ nested lists) so any transport can carry it. The MCP and REST
layers are **thin adapters with no logic of their own** — a new verb is added to
`kalman.api`, given a scope, and then exposed in both.

Continue with [Models](4_Models.md), or jump to [Python API](9_Python-API.md)
if you are integrating rather than modelling.

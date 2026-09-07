# libKalmanFramework — the C++ library

`libKalmanFramework` is the framework core in C++: **a standalone library with no required
dependency**. Not ROOT, not Eigen, not Torch — a C++20 compiler and nothing else.

It is a **port, not a binding**. It does not wrap the Python package, does not
link CPython, and is not generated from anything. The two implementations share
a design — Model / Engine / Filter — and meet at exactly one, deliberate seam:
**models**. A model compiled here registers itself with `Kalman::Factory`, and
the Python package reaches it through the C API (`Kalman/CAPI.h`) as an ordinary
`KalmanModel` — catalogue, API, MCP server and notebooks included. The last
section of this page has the measurements that put the seam *there* and not at
every filter step.

The dependency direction matters and runs one way:

```
    root-plus (Signal component)  ──links──▶  libKalmanFramework  ──▶  (nothing required)
```

The artefact is `libKalmanFramework.so` — it names the *framework*, and cannot be
mistaken for root-plus' own much smaller `ROOT::IOPlus::TKalmanFilter`. Headers
stay under `Kalman/` and the namespace stays `Kalman`.

A ROOT or ROOT+ component links libKalmanFramework; libKalmanFramework never reaches for ROOT. That
is what lets it also be linked by a plain C++ analysis, an embedded target, or a
GPU pipeline that has never heard of CERN.

## Layout

`include/` and `src/` mirror `modules/framework/` file for file, subdirectories
included, so a change on one side has an obvious home on the other:

| Python | C++ header | C++ class |
|---|---|---|
| `framework/constants.py` | `Kalman/Constants.h` | `EngineKind`, `StateFlavor` |
| `framework/model.py` | `Kalman/Model.h` | `Kalman::Model` |
| `framework/engine.py` | `Kalman/Engine.h` | `Kalman::Engine` (aliased `Estimator`) |
| `framework/filter.py` | `Kalman/Filter.h` | `Kalman::Filter` |
| `framework/sampler.py` | `Kalman/Sampler.h` | `Kalman::Sampler` |
| `framework/diagnostics.py` | `Kalman/Diagnostics.h` | `Kalman::Diagnostics` |
| `framework/abstract/metadata.py` | `Kalman/Abstract/Metadata.h` | `Abstract::Metadata` |
| `framework/engines/linear.py` | `Kalman/Engines/Linear.h` | `Engines::Linear` |
| `framework/engines/extended.py` | `Kalman/Engines/Extended.h` | `Engines::Extended` |
| `framework/engines/unscented.py` | `Kalman/Engines/Unscented.h` | `Engines::Unscented` |
| `framework/engines/particle.py` | `Kalman/Engines/Particle.h` | `Engines::Particle` |
| `framework/engines/ensemble.py` | `Kalman/Engines/Ensemble.h` | `Engines::Ensemble` |
| `framework/models/linear.py` | `Kalman/Models/Linear.h` | `Models::Linear` |
| `framework/models/extended.py` | `Kalman/Models/Extended.h` | `Models::Extended` |
| `framework/filters/kf.py` | `Kalman/Filters/KF.h` | `Filters::KF` |
| `framework/filters/ekf.py` | `Kalman/Filters/EKF.h` | `Filters::EKF` |
| `framework/filters/ukf.py` | `Kalman/Filters/UKF.h` | `Filters::UKF` |
| `framework/filters/pf.py` | `Kalman/Filters/PF.h` | `Filters::PF` |
| `framework/filters/enkf.py` | `Kalman/Filters/EnKF.h` | `Filters::EnKF` |
| `framework/io/record.py` | `Kalman/IO/Record.h` | `IO::Record` (aliased `Kalman::Record`) |
| `framework/io/collection.py` | `Kalman/IO/Collection.h` | `IO::Collection` |
| `framework/routers/base.py` | `Kalman/Routers/Base.h` | `Routers::Router` |
| `framework/routers/bayesian.py` | `Kalman/Routers/Bayesian.h` | `Routers::Bayesian` |
| `framework/routers/threshold.py` | `Kalman/Routers/Threshold.h` | `Routers::Threshold` |
| `framework/routers/hybrid.py` | `Kalman/Routers/Hybrid.h` | `Routers::Hybrid` |
| `framework/routers/neural.py` | `Kalman/Routers/Neural.h` | `Routers::Neural` |
| `framework/routers/nn/` | `Kalman/Routers/NN/Model.h` | `Routers::NN::Model` (the contract) |
| `framework/routers/nn/mlp.py` | `Kalman/Routers/NN/MLP.h` | `Routers::NN::MLP` (no dependency) |
| `framework/routers/nn/*.py` | `Kalman/Routers/NN/Torch.h` | `Routers::NN::Torch` (TorchScript) |
| `framework/routers/nn/*.py` | `Kalman/Routers/NN/Onnx.h` | `Routers::NN::Onnx` (ONNX) |
| `framework/filters/ikf.py` | `Kalman/Filters/IMM.h` | `Filters::IMM` |
| — (numpy) | `Kalman/Matrix.h` | `Kalman::Matrix`, `Kalman::Vector` |
| `modules/helpers.py` | `Kalman/Utils.h` | `Kalman::Utils` |
| `framework/models/native.py` | `Kalman/Factory.h` | `Kalman::Factory` + `KALMAN_REGISTER_MODEL` |
| `framework/native/core.py` | `Kalman/CAPI.h` | the `extern "C"` surface (`kalman_*`) |

`src/` follows the same tree wherever there is an implementation to hold; the
purely declarative headers (`Abstract/`, `Models/`, `Filters/`) have no `.cc`
because there is nothing to compile, not because they are missing.

**Not ported yet**, and deliberately absent rather than stubbed:
`framework/integrators/`, `framework/solvers/`, `framework/controllers/`,
`framework/native/` (the cffi accelerator kernels have no reason to exist here),
and the Combinatorial engine — a track-*finding* pass over an event of many hits
per layer, which needs a detector contract (`radii`, `seed_state`) that
`Kalman::Model` does not carry. Of the routers, *training* stays Python-only by
design — `routers/nn/` is Torch, and the optimiser and data loaders belong
there. What crosses is the finished policy, via `NN::Torch` or `NN::Onnx`.
`EngineKind` still *names* the Combinatorial kind so a model's declared default
survives a round-trip, and asking a C++ filter for it raises rather than
silently substituting a different algorithm.

## The three roles

| Role | Question | Class |
|------|----------|-------|
| **Model** | WHAT is the system? | `Kalman::Model` → `Models::Linear`, `Models::Extended` |
| **Engine** (alias *Estimator*) | HOW to infer? | `Kalman::Engine` → `Engines::Linear`, `Engines::Extended`, `Engines::Unscented`, `Engines::Particle`, `Engines::Ensemble` |
| **Filter** | WHEN does it run? | `Kalman::Filter` → `Filters::KF`, `Filters::EKF`, `Filters::UKF`, `Filters::PF`, `Filters::EnKF`, `Filters::IMM` |

Dependencies point one way only: **Filter → Engine → Model**.

`Engine::Predict` / `Update` / `Smooth` have a shared Gaussian implementation
that the Jacobian engines use as is; the sigma-point engine overrides the first
two, and the two *sample-set* engines (particle cloud, ensemble) override all
three — they are **stateful**, reseeded by `Engine::Reset()` when a run starts,
and declare `HasSmoother() == false`, so `Filter::Backward()` refuses instead of
pretending an RTS pass exists over a cloud.

Plus `Kalman::Record` (one pass, with its covariance band), `Kalman::Diagnostics`
(NIS / NEES / gains, asked *about* a run rather than being steps *of* one),
`Kalman::Sampler` (truth and measurements drawn from a model), `Kalman::Matrix` /
`Kalman::Vector` (the small dense types), and `Kalman::Utils` (Cholesky with a
pseudo-inverse fallback, NaN channel masking, central-difference Jacobians, the
Joseph form).

**No physics.** A pendulum, a drift chamber, a suspension or a power grid is an
*application*: it subclasses `Models::Linear` or `Models::Extended` and lives
where the domain knowledge lives — the same split as `modules/framework/` versus
`modules/apps/` on the Python side. libKalmanFramework ships the contract those
applications are written against and nothing more. The models in
`tests/cxx/Fixtures.h` are test fixtures, not a catalogue.

**No plotting and no I/O.** `Record::Series()` / `Sigma()` hand back the arrays a
plot needs; turning them into a `TGraphErrors`, a matplotlib axis or a CSV is the
integrator's job, because that is where the plotting dependency belongs.

## Building

```bash
make build         # configures + builds libKalmanFramework
make ctests        # the C++ suite, under ctest
make install       # into var/install by default
```

`make tests` remains the **Python** suite; the C++ tests live in `tests/cxx/` and
run under ctest, so the two never shadow each other.

### Optional backends

All three are detected quietly and none is needed for correctness. The flags land
in `Kalman/Config.h`.

| Flag | What it buys | Notes |
|---|---|---|
| `KALMAN_WITH_BLAS` | `cblas_dgemm` for matrix products at or above `KALMAN_BLAS_MIN_DIM` | On by default when a CBLAS is found; `-DKALMAN_USE_BLAS=OFF` to force the built-in kernel |
| `KALMAN_WITH_TORCH` | `NN::Torch` — TorchScript routing policies | Never used for `Kalman::Matrix` — see below |
| `KALMAN_WITH_ONNXRUNTIME` | `NN::Onnx` — ONNX routing policies | The portable half of the seam |

**Why BLAS and not Torch for the matrix type.** Measured, per n×n `double`
product on Apple Silicon:

| n | built-in `-O3` | BLAS (Accelerate) | libTorch |
|---|---|---|---|
| 2 | **8.6 ns** | 19.8 ns | 229.8 ns |
| 4 | **21.6 ns** | 26.2 ns | 228.0 ns |
| 6 | 67.8 ns | **53.8 ns** | 256.3 ns |
| 16 | 795.9 ns | **202.0 ns** | 582.9 ns |
| 32 | 5511.6 ns | **326.0 ns** | 730.0 ns |
| 128 | 211841.7 ns | **9469.6 ns** | 12303.2 ns |

A Torch op carries a ~230 ns dispatch floor — 10–25× an *entire* small matrix
product — so routing the core through tensors would make every filter slower.
BLAS is a plain function call and wins from n≈6 up, which is why the backend is
chosen **per operation** at `KALMAN_BLAS_MIN_DIM` rather than once for the whole
library: below the crossover the built-in kernel is genuinely faster.

Torch earns its keep where the Python side already uses it — ensembles, field
solvers, neural components — which is a future component, not a change to the
core.

## Using it

```cpp
#include <Kalman/Kalman.h>

// Your system. libKalmanFramework ships no physics, so the model is yours to write.
class ConstantVelocity : public Kalman::Models::Linear
{
    public:
        ConstantVelocity(double fs) : Kalman::Models::Linear("cv", 2, 1)
        {
            SetSampleRate(fs);
            Kalman::Matrix F = Kalman::Matrix::Identity(2);
            F(0, 1) = DeltaT();
            SetF(F);
            // ... H, Q, R
        }
};

ConstantVelocity model(100.);
Kalman::Filters::KF filter(&model);

auto forward  = filter.Forward(measurements);   // NaN channels are skipped
auto smoothed = filter.Backward();              // RTS over the same run

printf("mean NIS = %.3f (expect ~%zu)\n", filter.MeanNIS(), model.NMeasure());
```

`examples/cxx/kalman101.cc` is this, finished: it defines its model, samples a
track, filters, smooths, prints the diagnostics and writes a plottable table.

For a **nonlinear** system, override `f()` and/or `h()` and tell the base that
`F(t)` is no longer the Jacobian:

```cpp
class MyModel : public Kalman::Models::Extended
{
    public:
        MyModel() : Extended("mine", 2, 1) {}
        Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &u,
                         double t) const override;
};
```

Deriving from `Models::Extended` is what switches `Jacobian()` / `JacobianH()` to
central differences — on a plain `Kalman::Model` you would have to call
`SetAnalyticJacobian(false)` yourself, and forgetting it means `F(t)` is used as a
Jacobian it is not. Override `Jacobian()` directly if you have the analytic form:
faster and more accurate.

Shape mistakes are reported by `CheckShapes()`, which the filter runs before every
pass, as an exception naming the matrix and both shapes.

## The decision layer

When no single model covers a system's whole behaviour, run several and carry a
probability on each. `Kalman::Filters::IMM` does the mixing, prediction and
combination; the *decision* — which model is right now — is delegated to a
`Kalman::Routers::Router` and is the only pluggable part of the step.

```cpp
Kalman::Routers::Bayesian router;                 // the classical mu ∝ c · L
Kalman::Filters::IMM imm({&cruise, &manoeuvre}, Pi, &router);

auto record = imm.Forward(measurements);
const Kalman::Matrix &modes = imm.ModeHistory();  // (T, M): which regime, when
```

| Router | Decides on | Needs |
|---|---|---|
| `Routers::Bayesian` | `mu_j ∝ c_j · L_j` — Bayes, and the right default | nothing |
| `Routers::Threshold` | whichever model's windowed NIS sits closest to `m` | nothing |
| `Routers::Hybrid` | a geometric blend of two routers, e.g. a learned one on a Bayesian leash | nothing |
| `Routers::Neural` | a policy trained in Python | a `NN::Model` (see below) |

Each mode contributes its own declared engine, so a linear and a nonlinear mode
can sit in one bank. Every mode must share `n` and `m` — that is what makes
mixing meaningful, and it is checked at construction.

### Training in Python, deciding in C++

`Routers::Neural` is the seam between the two implementations, and it keeps the
division of labour that `routers/neural.py` states in its opening lines:

- **the router** owns feature extraction and normalisation — the NIS window, the
  front-padding, the standardisation;
- **the model** owns the architecture and the forward pass.

So `Routers::Neural` holds a `NN::Model` and contains no inference runtime at
all. That is what makes the runtimes genuinely optional rather than nominally so:

| Backend | Executes | Needs | When |
|---|---|---|---|
| `NN::MLP` | a tanh MLP from its weights | **nothing** | the architecture is the MLP; you want zero dependencies |
| `NN::Torch` | a TorchScript module | libTorch | highest fidelity — the *same* `torch.nn` module, traced |
| `NN::Onnx` | an ONNX graph | ONNX Runtime | deployment — a 10 MB runtime, not tied to a Torch version |

Python owns training; C++ owns inference; a **trained policy** is what crosses,
not a language binding — once at load time, rather than once per step.

```python
router = NeuralRouter(model=MLP(hidden=32), window=5)
ikf.fit(training_zk, labels)                  # train where the data loaders are
router.export_onnx("router.onnx")             # for NN::Onnx
router.export_torchscript("router.pt")        # for NN::Torch
```

```cpp
Kalman::Routers::NN::Onnx policy("router.onnx");   // or NN::Torch, or NN::MLP
Kalman::Routers::Neural   router(&policy, /* window */ 5);
Kalman::Filters::IMM      imm({&cruise, &manoeuvre}, Pi, &router);
```

Both exports produce the same probabilities as the Python router; measured
agreement is `2e-13` for ONNX and `1.8e-13` for TorchScript. Neither runtime is
required: with neither present the routing layer still works through `NN::MLP`,
and a `Routers::Neural` with no model falls back to the Bayesian update rather
than failing.

The export bakes the feature standardisation *and* the softmax head into the
graph, so a raw NIS window maps straight to probabilities and there is no second
file to keep in step — a normalisation shipped apart from its weights still
produces a valid-looking probability vector, which is the worst kind of bug.
`tests/test_router_export.py` pins the equivalence; measured agreement between
the Python router and the C++ one is `2e-13`.

Two things that bite:

- **The IR version.** Torch emits the newest ONNX IR it knows, and deployed
  runtimes reject it — ONNX Runtime 1.16 refuses anything above 9 with
  *"Unsupported model IR version: 10"*. `export_onnx(..., ir_version=9)` is the
  default for that reason.
- **The window must match.** A policy trained at `window=5` fed a `window=7`
  history is a shape error, and `Routers::Neural` raises rather than guessing.

The rule for both optional backends: constructing one asks for a *specific*
artifact, so a missing file or a missing runtime throws. The graceful degradation
lives one level up, in `Routers::Neural`, which routes by Bayes when it has no
usable model. A missing optional dependency costs you the *policy*, never the
filter.

## The factory: how a C++ model reaches Python

libKalmanFramework ships no physics, so the question is how physics written
*against* it — in an application, a ROOT+ analysis, a plugin — becomes visible
to something that did not compile it. The answer is one line per model:

```cpp
class ConstantVelocity : public Kalman::Models::Linear { /* F, H, Q, R ... */ };
KALMAN_REGISTER_MODEL(ConstantVelocity, "constant-velocity");
KALMAN_REGISTER_MODEL(ConstantVelocity, "constant-velocity-slow", 10., 0.5, 0.2);  // configured
```

The registrar runs at static-initialisation time and `Kalman::Factory::Instance()`
then knows the name: `Names()`, `Has()`, `Describe()` (title, dimensions, engine,
origin) and `Create()` — a fresh `std::unique_ptr<Model>` with no knowledge of
the type. Names are lowercase-hyphenated by convention so they can never be
mistaken for the CamelCase Python classes they sit beside. Every failure is loud:
a duplicate name, an unknown name, a library that registers nothing.

**Plugins.** A translation unit like `examples/cxx/example_plugin.cc` compiled
as a shared library (`libKalmanPlugin_example`) is a plugin: `Factory::LoadPlugin(path)`
dlopens it, the registrars run, and its models are in the catalogue. The plugin
may link anything it likes — ROOT, RooFit, a detector library — because the
framework only ever sees `Kalman::Model`. That is the one code path by which a
model written for ROOT+ reaches the Python side, and ROOT stays inside the plugin.

**The C API.** `Kalman/CAPI.h` is the flat `extern "C"` view of the factory and
of a model — `kalman_factory_*` to discover and load, `kalman_model_*` to
construct one and evaluate `F/H/B/L/Q/R`, `f/h`, the Jacobians and the initial
state, all row-major `double` arrays, `0`/`-1` returns and `kalman_last_error()`.
No exception crosses it. `tests/cxx/06_factory.cc` covers the registry, the
loader and this surface.

**The Python side.** `modules/framework/native/core.py` dlopens the library
through cffi in *ABI mode* (no compiler) and `NativeModel` in
`modules/framework/models/native.py` wraps a factory instance as a `KalmanModel`:

```python
from kalman.framework import NativeModel
import kalman.api as api

api.load_plugin("KalmanPlugin_example")           # by name, from the allow-listed dirs
m = NativeModel("pendulum", "sensor-1")           # a C++ model ...
rec = m.filter(m.sample(5.0).zk)                  # ... driven by the Python EKF
api.run_filter({"base": "constant-velocity", "sensor_id": "cv", "fs": 20}, zk)
```

Every registered name becomes a class in `api.catalog()["models"]`, `describe()`
reports `"backend": "native"` with its plugin, and the MCP `list_models` /
`run_filter` / `forecast` tools need no change. `fs=`, `x0=`, `P0=` given from
Python are pushed *into* the C++ instance, so both sides agree on every step.

**What crosses, and what does not.** One C call per `f`, `h` or coefficient
evaluation — cheap, and in the direction the measurements below say is
affordable. What does *not* cross is a Python callback from inside a C++ loop;
a Python-defined model keeps running on the Python engines, and a C++ model on
either side. The library is found through `KALMAN_FRAMEWORK_LIBRARY`,
`KALMANFRAMEWORK_ROOT`, or this checkout's `var/install/lib`; without it the
Python package simply reports `catalog()["native"]["library"] = None`.

**Loading a plugin runs code.** So `api.load_plugin` takes a plugin *name*, not
a path, resolved only inside allow-listed directories (`KALMAN_PLUGIN_PATH`,
default: this checkout's `var/install/lib` and `var/build/lib`); a path is
accepted only when it already lies inside one of them, after `realpath`. It is
gated by its own scope, `load_plugins` — never by `write_models`, which only
ever writes data — and the MCP tool of the same name exposes exactly that.

## Integrating with root-plus

`examples/cxx/ballistics_root.cc` is the worked demonstrator — a projectile under
gravity and quadratic drag, tracked by an EKF, drawn through ROOT. It is built
only when ROOT is present (`find_package(ROOT QUIET)` in `examples/cxx`), so the
ROOT-free build is unaffected, and it shows the whole integration in one file:

```
$ source var/install/bin/thislib.KalmanFramework.sh
$ ballistics_root
ballistics — n = 6, m = 3, l = 0, fs = 50 Hz, engine = Extended, Jacobian = numerical
mean NIS  = 2.858   (consistent near m = 3)
mean NEES = 3.865   (consistent near n = 6)
altitude RMSE : filtered = 0.465 m, smoothed = 0.130 m
wrote ballistics_root.pdf, .png and .root
```

15% of the altitude fixes are dropped at random (NaN), and the filter coasts
through the gaps on prediction with no special case anywhere in the example.

### The ROOT+ side

`root-plus` wraps results for ROOT users inside its **Signal** component, next
to its own unrelated `TKalmanFilter`, with libKalmanFramework as an *optional*
dependency of that component:

```
root-plus/
  cmake/Packages/Kalman/FindKalmanFramework.cmake
  include/ROOT/IOPlus/Signal/DSP/Filters/TKalmanRecord.h
  include/libRIOPlus_Signal.Config.h.in      # #cmakedefine KALMANFRAMEWORK_FOUND
  include/libRIOPlus_Signal.LinkDef.h        # the dictionary entry, guarded
  src/Signal/TKalmanRecord.cc
  tests/Signal/17_kalmanrecord.cc
```

`ROOT::IOPlus::TKalmanRecord` takes a `Kalman::IO::Record` and adds the two
things libKalmanFramework deliberately does not provide: a `TObject` with a
`ClassDef` so a result goes into a `TFile`, and `TGraph` / `TGraphErrors` /
`TMultiGraph` of the estimate with its band. It stores the covariance *diagonal*
only — a plot and a stored summary want the per-component sigma, and keeping the
full `n × n` per step would dominate the file for no gain; use the
`Kalman::Record` itself when the correlations matter.

`ROOT::IOPlus::TKalmanEstimate` is an alias of it, so either spelling compiles.
It is only a source-level name: ROOT I/O stores the real class, so files written
under either spelling are interchangeable.

When libKalmanFramework is not found, `KALMANFRAMEWORK_FOUND` stays undefined
and header, source and dictionary entry compile to nothing — Signal builds
unchanged, and its test reports itself skipped. Point CMake at a prefix with
`-DKalmanFramework_ROOT=<prefix>`. The class stays `ROOT::IOPlus::TKalmanRecord`,
not a `Kalman::` sub-namespace: root-plus' namespace is flat, and a nested
`Kalman` would shadow the framework's own `Kalman::` in every lookup.

> **Not** `ROOT::IOPlus::TKalmanFilter`. That is root-plus' own small
> self-contained linear filter, now its neighbour in the same directory and
> unrelated to this framework. They share a word, not an implementation, and
> both can be loaded in one process — which is exactly why the framework's own
> classes carry no `T` prefix and no `TObject` base.
>
> A ROOT+ *model* — physics that wants `TF1` or RooFit inside `f()` — does not
> go here either. It is a plugin (previous section): it links ROOT, registers
> with `Kalman::Factory`, and the Python side loads it by name.

### Linking it yourself

libKalmanFramework is linked, not loaded. From a ROOT/ROOT+ translation unit:

```cpp
#include <Kalman/Kalman.h>
#include <TGraphErrors.h>

Kalman::Record rec = filter.Forward(measurements);

auto y  = rec.Series(0);
auto ey = rec.Sigma(0);
auto t  = rec.Times();
std::vector<double> ex(rec.NSteps(), 0.0);
TGraphErrors g(rec.NSteps(), t.data(), y.data(), ex.data(), ey.data());
```

Because the classes are `Kalman::Model`, `Kalman::Filter` and so on — no `T`
prefix, no `TObject` base — they cannot be confused with root-plus' own
`ROOT::IOPlus::TKalmanFilter`, which is a separate and much smaller standalone
linear filter. Both can be present in one process.

## Differences from the Python framework

| | Python | C++ |
|---|---|---|
| Engines | Linear, Extended, Unscented, Particle, Ensemble, Combinatorial | Linear, Extended, Unscented, Particle, Ensemble |
| Non-additive noise (UKF) | augmented-state sigma points | additive only — fold the noise into `f` |
| Ensemble hooks | `model.forecast(X)`, `model.observe(X)`, `model.localization()`, `model.initial_ensemble()` | per-member `f`/`h`; `SetLocalization(rho_xy, rho_yy)` on the engine |
| Integrators / solvers | Euler, RK4, Dormand–Prince / spectral, finite-difference | none (a model owns its own stepping) |
| Controllers / routers | P, PID, LQR, MPPI, Neural, IMM | none yet — the ONNX seam is the plan |
| Applications | `modules/apps/` | none — they belong downstream |
| Missing data | NaN channel skipped | NaN channel skipped (same) |
| Covariance update | Joseph form | Joseph form (same) |
| Smoother | RTS | RTS (same) |
| Singular `S` | `np.linalg.pinv` drops null directions | eigenvalue pseudo-inverse, same semantics |
| Diagnostics | innovations, gains, NIS, NEES | NIS, NEES |

A model asking for an engine kind C++ does not implement (today: Combinatorial)
raises rather than silently substituting, so a model declaring it will not
quietly get a time-series step.

## Where the Python boundary is, and why

The framework *is* wrapped in Python — for **models**, through the factory and
the C API above. What it deliberately does not do is the other obvious thing:
wrap the C++ *engines* so the existing Python apps transparently get C++ speed.
That would mean the C++ engine calling back into Python for `model.f/h/F/Q` at
every step, and measured on `BallisticModel`:

```
model.f  (Python/numpy/scipy)   :  20.74 µs per call
whole native predict + crossing :   3.75 µs per call
```

One callback costs ~6× the entire native step it would sit inside, and an EKF
needs ~21 of them per step for the Jacobian. Profiling agrees: of 5.74 s for a
6000-step EKF, **88% is `numerical_jacobian` + `model.f`**. Amdahl caps the win at
~1.1× even if the covariance algebra became free.

That is exactly why the Python package's existing C acceleration
(`modules/framework/native/`) engages *only* when it can eliminate callbacks
entirely — constant matrices, no control callable — and then fuses all N steps
into one call, for a measured **21× on linear forward, 33.9× geometric mean**.
The limit is the language boundary, not the language.

So the seam sits where a crossing is paid once per evaluation *into* C++ — a
compiled model evaluated by a Python engine — and never once per step *out* of
it. The portable artifacts between the two sides are a **registered model**
(through the factory), **data** (a record) and a **trained ONNX policy**.

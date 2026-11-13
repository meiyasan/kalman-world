# libKalman — the C++ library

`libKalman` is the framework core in C++: **a standalone library with no required
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
    root-plus (Signal component)  ──links──▶  libKalman  ──▶  (nothing required)
```

The artefact is `libKalman.so` — it names the *framework*, and cannot be
mistaken for root-plus' own much smaller `ROOT::IOPlus::TKalmanFilter`. Headers
stay under `Kalman/` and the namespace stays `Kalman`.

A ROOT or ROOT+ component links libKalman; libKalman never reaches for ROOT. That
is what lets it also be linked by a plain C++ analysis, an embedded target, or a
GPU pipeline that has never heard of CERN.

## Layout

`include/` and `src/` mirror `modules/framework/` file for file, subdirectories
included, so a change on one side has an obvious home on the other:

| Python | C++ header | C++ class |
|---|---|---|
| `framework/constants.py` | `Kalman/Constants.h` | `EngineKind`, `StateFlavor` |
| `framework/model.py` | `Kalman/Model.h` | `Kalman::Model` |
| `framework/integrator.py` | `Kalman/Integrator.h` | `Kalman::Integrator`, `Tableau`, `RungeKuttaStep` |
| `framework/integrators/euler.py` | `Kalman/Integrators/Euler.h` | `Integrators::Euler` |
| `framework/integrators/rk4.py` | `Kalman/Integrators/RK4.h` | `Integrators::RK4` |
| `framework/integrators/dormand_prince.py` | `Kalman/Integrators/DormandPrince.h` | `Integrators::DormandPrince` |
| `framework/constants.py` (`KalmanDifferentiatorKind`) | `Kalman/Differentiator.h` | `Kalman::Differentiator`, `Differentiators::Jacobian` |
| `helpers.numerical_jacobian` | `Kalman/Differentiators/Forward.h`, `Central.h` | `Differentiators::Forward`, `Differentiators::Central` |
| `framework/solver.py` | `Kalman/Solver.h` | `Kalman::Solver` |
| `framework/solvers/spectral.py` | `Kalman/Solvers/Spectral.h` | `Solvers::Spectral` |
| `framework/solvers/finite_difference.py` | `Kalman/Solvers/FiniteDifference.h` | `Solvers::FiniteDifference` |
| `apps/weather/flow.py` (`NS2D`), `barotropic.py` | `Kalman/Solvers/Vorticity.h` | `Solvers::Vorticity` (the equation the flow models share) |
| — (numpy, scipy) | `Kalman/Algebra/Cholesky.h` | `Algebra::Cholesky`, `CholeskyFactor`, `Solve`, `Inverse` |
| — (numpy, scipy) | `Kalman/Algebra/Symmetric.h` | `Algebra::Symmetrize`, `SymmetricEigen`, `Similarity`, `ConditionNumber` |
| `helpers.van_loan_discretize`, `matrix_exponential` | `Kalman/Algebra/Exponential.h` | `Algebra::Exponential`, `Algebra::VanLoan` |
| — (numpy arrays) | `Kalman/Grid.h` | `Kalman::Field`, `Kalman::SpectralField` |
| — (`numpy.fft`) | `Kalman/FFT.h` | `Kalman::FFT` (radix 2, mixed radix for factors 3 and 5, Bluestein otherwise; cached plans; no dependency) |
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
| `framework/engines/combinatorial.py` | `Kalman/Engines/Combinatorial.h` | `Engines::Combinatorial` |
| `framework/engines/combinatorial.py` (`TrackCandidate`, `GateContext`) | `Kalman/Tracking.h` | `Tracking::Candidate`, `Tracking::GateContext`, `Tracking::Detector` |
| `framework/models/linear.py` | `Kalman/Models/Linear.h` | `Models::Linear` |
| `framework/models/extended.py` | `Kalman/Models/Extended.h` | `Models::Extended` |
| `framework/filters/kf.py` | `Kalman/Filters/KF.h` | `Filters::KF` |
| `framework/filters/ekf.py` | `Kalman/Filters/EKF.h` | `Filters::EKF` |
| `framework/filters/ukf.py` | `Kalman/Filters/UKF.h` | `Filters::UKF` |
| `framework/filters/pf.py` | `Kalman/Filters/PF.h` | `Filters::PF` |
| `framework/filters/enkf.py` | `Kalman/Filters/EnKF.h` | `Filters::EnKF` |
| `framework/filters/ckf.py` | `Kalman/Filters/CKF.h` | `Filters::CKF` |
| `framework/control.py` | `Kalman/Controller.h` | `Kalman::Controller` |
| `framework/controllers/proportional.py`, `pid.py`, `lqr.py`, `mppi.py` | `Kalman/Controllers/*.h` | `Controllers::Proportional`, `PID`, `LQR`, `MPPI` |
| `framework/controllers/boost.py`, `adaptive_qr.py` | `Kalman/Controllers/Adaptive.h` | `Controllers::Boost`, `Controllers::AdaptiveQR` |
| `framework/io/record.py` | `Kalman/IO/Record.h` | `IO::Record` (aliased `Kalman::Record`) |
| `framework/io/collection.py` | `Kalman/IO/Collection.h` | `IO::Collection` |
| `framework/routers/base.py` | `Kalman/Routers/Base.h` | `Routers::Router` |
| `framework/routers/bayesian.py` | `Kalman/Routers/Bayesian.h` | `Routers::Bayesian` |
| `framework/routers/threshold.py` | `Kalman/Routers/Threshold.h` | `Routers::Threshold` |
| `framework/routers/hybrid.py` | `Kalman/Routers/Hybrid.h` | `Routers::Hybrid` |
| `framework/routers/neural.py` | `Kalman/Routers/Neural.h` | `Routers::Neural` |
| `framework/routers/nn/` | `Kalman/Routers/NN/Model.h` | `Routers::NN::Model` (the contract) |
| `framework/routers/nn/mlp.py` | `Kalman/Routers/NN/MLP.h` | `Routers::NN::MLP` (no dependency) |
| `framework/routers/nn/*.py` | `Kalman/Routers/NN/Torch.h` | `Routers::NN::Torch` (TorchScript; in `libKalmanLearn`) |
| `framework/routers/nn/*.py` | `Kalman/Routers/NN/Onnx.h` | `Routers::NN::Onnx` (ONNX; in `libKalmanLearn`) |
| `framework/filters/ikf.py` | `Kalman/Filters/IKF.h` | `Filters::IKF` (aliased `IMM`) |
| — (numpy) | `Kalman/Matrix.h` | `Kalman::Matrix`, `Kalman::Vector` |
| `modules/helpers.py` | `Kalman/Utils.h` | `Kalman::Utils` |
| `framework/models/native.py` | `Kalman/Factory.h` | `Kalman::Factory` + `KALMAN_REGISTER_MODEL` |
| `framework/native/core.py` | `Kalman/CAPI.h` | the `extern "C"` surface (`kalman_*`) |

`src/` follows the same tree wherever there is an implementation to hold; the
purely declarative headers (`Abstract/`, `Models/`, `Filters/`) have no `.cc`
because there is nothing to compile, not because they are missing.

**Not ported yet**, and deliberately absent rather than stubbed: the *learned*
controllers (`NeuralKalmanController`, `NeuralAdaptiveQRKalmanController` — their
training is Torch, and what would cross is a trained policy through the ONNX
seam, as for the routers), the reserved integrator/solver kinds
(`ImplicitEuler`, `FiniteVolume`, `FiniteElement` — `Make()` throws for them,
as `resolve_*` does in Python). Of the routers, *training* stays Python-only by
design — `routers/nn/` is Torch, and the optimiser and data loaders belong
there. What crosses is the finished policy, via `NN::Torch` or `NN::Onnx`.
Every `EngineKind` now has a C++ engine; an integer that is not a kind still
raises rather than silently becoming a linear step.

### Roles and utilities

A role is something a user chooses or writes; it has a base class, a `Kind`
constant and a directory. A utility is arithmetic nobody chooses between.

| | What it answers | Where |
|---|---|---|
| Integrator (role) | how do I advance in time? | `Kalman/Integrators/` |
| Solver (role) | how do I handle space, for a field? | `Kalman/Solvers/` |
| Differentiator (role) | where do Jacobians come from? | `Kalman/Differentiators/` |
| Algebra (utilities) | Cholesky, solves, eigenvalues, `exp(A)`, the Riccati equation | `Kalman/Algebra/` |

"Solver" here always means a field solver. The linear solver of the framework
is Cholesky, in `Kalman/Algebra/Cholesky.h`: a covariance is symmetric
positive-definite, so nothing needs LU. `Kalman::Utils` keeps the Kalman
arithmetic (missing channels, the Joseph update, sigma-point sums) and still
exports the algebra names, so existing code compiles unchanged.

## The three roles

| Role | Question | Class |
|------|----------|-------|
| **Model** | WHAT is the system? | `Kalman::Model` → `Models::Linear`, `Models::Extended` |
| **Engine** (alias *Estimator*) | HOW to infer? | `Kalman::Engine` → `Engines::Linear`, `Engines::Extended`, `Engines::Unscented`, `Engines::Particle`, `Engines::Ensemble`, `Engines::Combinatorial` |
| **Filter** | WHEN does it run? | `Kalman::Filter` → `Filters::KF`, `Filters::EKF`, `Filters::UKF`, `Filters::PF`, `Filters::EnKF`; `Filters::IKF`, `Filters::CKF` |
| **Controller** | WHAT to command? | `Kalman::Controller` → `Controllers::Proportional`, `PID`, `LQR`, `MPPI`; `Boost`, `AdaptiveQR` |
| **Integrator** | WHEN, within one dt? | `Kalman::Integrator` → `Integrators::Euler`, `RK4`, `DormandPrince` |
| **Solver** | HOW, in space? | `Kalman::Solver` → `Solvers::Spectral`, `Solvers::FiniteDifference` |

Dependencies point one way only: **Filter → Engine → Model**.

An ODE model overrides `Model::Derivative()` and calls `UseContinuousDynamics()`;
the default `f()` then advances the derivative one sample with the model's
declared integrator (RK4 unless `SetIntegrator()` says otherwise, in
`Substeps()` pieces), exactly as a `DYNAMICS = CONTINUOUS` model does in Python. A field solver
composes an integrator the same way and marches a `Kalman::Field` through the
integrator's Butcher tableau — one `RungeKuttaStep` template serves a
`Kalman::Vector` and a spectral field alike.

A controller closes the loop: `Filter::Forward(zk, controller)` calls it once
per step with the belief, the time and the last innovation, and feeds the
command to the prediction — `KalmanFilter.forward(zk, uk=controller)` in
Python. The cost it logs is charged against `Model::ClipControl(u)`, what the
actuator delivers, not the raw ask; MPPI's rollouts clip the same way, so a
planner sees the limit it will hit. LQR solves the Riccati equation with
`Algebra::SolveDARE` (the recursion, then Newton steps on a Stein equation
summed by doubling: no scipy and no factorisation beyond Cholesky); the two
adapters scale `R` (a gain boost when the NIS says the filter is surprised) and
`Q` (from the empirical innovation covariance). The Python controllers and the
rule-based routers forward to these classes; see
[C++ models from Python](16_Cxx-Models-from-Python.md).

Track *finding* is a different shape of problem — an event of many hits per
layer in, a set of tracks out — so `Engines::Combinatorial` adds `Find()` and
`SmoothTracks()` over a whole event on top of the inherited Extended step, and
`Filters::CKF` is its leaf rather than a `Kalman::Filter`. A trackable model
implements `Kalman::Tracking::Detector` (`Radii()`, `SeedState()`) next to
`Kalman::Model`; `tests/cxx/DriftChamber.h` is the fixture, the counterpart of
`apps/tracking/driftchamber.py`. On an identical event the two searches find
the same tracks hit for hit — the test records one such case.

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
`modules/apps/` on the Python side. libKalman ships the contract those
applications are written against and nothing more. The models in
`tests/cxx/Fixtures.h` are test fixtures, not a catalogue.

**No plotting and no I/O.** `Record::Series()` / `Sigma()` hand back the arrays a
plot needs; turning them into a `TGraphErrors`, a matplotlib axis or a CSV is the
integrator's job, because that is where the plotting dependency belongs.

## Building

```bash
make build         # configures + builds libKalman
make ctests        # the C++ suite, under ctest
make install       # into var/install by default
```

`make tests` remains the **Python** suite; the C++ tests live in `tests/cxx/` and
run under ctest, so the two never shadow each other.

### The four libraries

`make build` builds four libraries, installed side by side:

| Library | CMake target | What it holds | Depends on |
|---|---|---|---|
| `libKalman` | `Kalman::Kalman` | the core: models, engines, filters, integrators, solvers, controllers, routers | nothing (BLAS when found) |
| `libKalmanApps` | `Kalman::Apps` | the application models, each the twin of a Python class | `libKalman` |
| `libKalmanLearn` | `Kalman::Learn` | learned components: the ONNX and TorchScript backends | `libKalman`; ONNX Runtime and libTorch, both optional |
| `libKalmanNet` | `Kalman::Net` | the JSON API, the REST and MCP servers, `kalman-serve` ([docs/18](18_Net.md)) | `libKalman`; nlohmann_json, cpp-httplib, OpenSSL (SQLite optional), all private |

Up to release 1.0 the core was called `libKalmanFramework`. That name is still
installed as a link to `libKalman`, and the Python package still reads the old
`KALMANFRAMEWORK_ROOT` and `KALMAN_FRAMEWORK_LIBRARY` variables.

From another CMake project:

```cmake
find_package(Kalman REQUIRED COMPONENTS Apps Learn)   # the components are optional
target_link_libraries(my_program PRIVATE Kalman::Kalman Kalman::Apps Kalman::Learn)
```

with `-DCMAKE_PREFIX_PATH=<install prefix>`.

### The `kalman` program

`bin/kalman` says what an installation can run. It links `libKalman` only and
loads `libKalmanApps` as a plugin when it is beside the library:

```sh
kalman                               # the models, one per line
kalman models ballistic              # names containing "ballistic"
kalman model car-speed               # everything one model declares
kalman plugins                       # the libraries the models came from
kalman --plugin KalmanPlugin_example models
kalman --plugin KalmanPlugin_example export fading-sensor --steps 200 -o fading.kw
kalman export car-speed -o car.kw # write a model to a file (see "Model files")
kalman model car.kw              # what a file holds
kalman models --json                 # the same, for a program
```

```
MODEL      DYNAMICS    STRUCTURE  ENGINE    n  m  l  fs [Hz]   JACOBIAN  PROCESS NOISE  MEAS. NOISE
ballistic  continuous  maps       Extended  9  3  0  128       forward   additive       additive
car-speed  continuous  maps       Extended  2  1  0  20        forward   additive       additive
storm      discrete    maps       Extended  5  3  0  0.166667  forward   additive       additive
```

| Column | Read from | Values |
|---|---|---|
| DYNAMICS | `Model::Dynamics()` | `continuous` (a `Derivative()` or a generator, decoded for the sample rate) or `discrete` (`f` is the step) |
| STRUCTURE | `Model::Structure()` | `maps` (code), `matrices`, `time-varying matrices`: what a file can hold of the model |
| ENGINE | `Model::DefaultEngine()` | the engine a filter picks for the model |
| JACOBIAN | `Model::Differentiator()` | `analytic`, `forward`, `central` |
| PROCESS NOISE | `HasProcessNoiseInput()`, `L()`, `NoiseIntensity()` | `additive`, `through L`, `in f`; `per unit time` when Q is stated as an intensity |
| MEAS. NOISE | `HasMeasurementNoiseInput()` | `additive`, `in h` |

`kalman model NAME` gives one model in full: the sample rate and the step it
implies, the integrator and its substeps, then the state vector and the
measurement component by component, and the named parameters with their
defaults.

```
$ kalman model car-speed
  sample rate         20 Hz, one sample every 50 ms
  dynamics            continuous
  integrator          RK4
  ...
  state vector, n = 2 (process noise additive)
         name      unit  x0            Q             sqrt(Q)
    [0]  distance  m     0             0.001         0.0316228
    [1]  speed     m/s   0             0.05          0.223607

  measurement, m = 1 (noise additive)
         name      unit  R             sqrt(R)
    [0]  distance  m     0.25          0.5
```

On a terminal the output is coloured (continuous in green, discrete in yellow,
noise that goes through a map in magenta); a pipe, a file and `--json` get
plain text. `--color always|never` forces it either way and `NO_COLOR` is
honoured.

The names and units are what the model declares in its constructor with
`DeclareState({{"distance", "m"}, {"speed", "m/s"}})` and
`DeclareMeasurement({{"distance", "m"}})`; a model that declares nothing shows
`x0, x1, ...` and `z0, z1, ...` with no unit. They are read with
`Model::StateComponents()` and `Model::MeasurementComponents()`, and from C
with `kalman_model_state_name/_unit` and `kalman_model_measurement_name/_unit`. A
plugin is given as a file, or as a name looked for in `KALMAN_PLUGIN_PATH`
(beside `libKalman` when unset). Only compiled models are listed: the models
written in Python are in `kalman.apps`.

### Optional backends

None is needed for correctness.

| Option | Library | What it buys | Default |
|---|---|---|---|
| `KALMAN_USE_BLAS` | `libKalman` | `cblas_dgemm` for matrix products at or above `KALMAN_BLAS_MIN_DIM` (`KALMAN_WITH_BLAS` in `Kalman/Config.h`) | on when a CBLAS is found |
| `KALMAN_USE_ONNXRUNTIME` | `libKalmanLearn` | `NN::Onnx`: ONNX routing policies, the portable half of the seam | on when ONNX Runtime is found |
| `KALMAN_USE_TORCH` | `libKalmanLearn` | `NN::Torch` (TorchScript routing policies) and `Learn::Solvers::Vorticity` (the batched field solver, CPU/CUDA/MPS) | on when a libTorch is found |
| `KALMAN_BUILD_LEARN` | | build `libKalmanLearn` at all | on |
| `KALMAN_BUILD_NET` | | build `libKalmanNet` and `kalman-serve` (skipped with a notice when a dependency is missing) | on |
| `KALMAN_USE_ZLIB` | | compress the tables of model files (`.kw`) with zlib when it is found | on |

The AI runtimes are looked for by `libKalmanLearn` only. `libKalman` never
links one, whatever is installed on the build machine: a core linked to one
libTorch, loaded into a Python process that had another, used to break
`import torch`.

libTorch is the C++ library, found like any C++ dependency: `Torch_DIR` when
given, else the conda environment's `libtorch` package
(`$CONDA_PREFIX/share/cmake/Torch`), else wherever CMake finds one. Only its
own three libraries are linked, by path. Nothing here needs Python's `torch`
package, so a conda package of `libKalmanLearn` depends on `libtorch` alone.

A process holds one libTorch. Where Python's `torch` is imported in the same
process (training a router, say), it must be the same installation; conda's
`libtorch` and `pytorch` are.

### The field solver on libTorch

`Kalman::Learn::Solvers::Vorticity` steps many 2-D vorticity fields in one
call: `dw/dt = -u.grad(w) - beta v + nu lap(w) + F`, pseudo-spectral, with any
explicit Runge-Kutta tableau, on `cpu`, `cuda` or `mps` (single precision on
`mps`, which has no double). It is what `SpectralSolver(backend="torch")` runs
in Python: `NS2D` (Kolmogorov flow) and `BetaPlaneNS2D` state their terms
(`_vorticity_terms`) and the steps happen in C++. Python's `torch` is not
imported; there is no Python-Torch implementation of the solver any more. A
solver with other physics uses `backend="numpy"`.

Measured on this machine against the Python-Torch loop it replaced, 32 members:

| Case | Python Torch (removed) | libTorch (C++) | Largest difference |
|---|---|---|---|
| Kolmogorov 64×64, 50 steps, CPU | 1074 ms | 1109 ms | 0 |
| Kolmogorov 128×128, 20 steps, CPU | 1797 ms | 1762 ms | 0 |
| Kolmogorov 128×128, 20 steps, MPS | 60 ms | 54 ms | 0 |
| Beta-plane 64×64, 50 steps, CPU | 1303 ms | 1257 ms | 0 |

Bit-identical, and the same speed: the time is the FFTs'.

## Using it

```cpp
#include <Kalman/Kalman.h>

// Your system. libKalman ships no physics, so the model is yours to write.
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
probability on each. `Kalman::Filters::IKF` does the mixing, prediction and
combination; the *decision* — which model is right now — is delegated to a
`Kalman::Routers::Router` and is the only pluggable part of the step.

```cpp
Kalman::Routers::Bayesian router;                 // the classical mu ∝ c · L
Kalman::Filters::IKF ikf({&cruise, &manoeuvre}, Pi, &router);

auto record = ikf.Forward(measurements);
const Kalman::Matrix &modes = ikf.ModeHistory();  // (T, M): which regime, when
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
Kalman::Filters::IKF      ikf({&cruise, &manoeuvre}, Pi, &router);
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

libKalman ships no physics, so the question is how physics written
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

**The C API.** `Kalman/CAPI.h` is the flat `extern "C"` view of the factory, of
a model, and of the engines — `kalman_factory_*` to discover and load,
`kalman_model_*` to construct one and evaluate `F/H/B/L/Q/R`, `f/h`, the
Jacobians and the initial state, `kalman_engine_*` to build a C++ engine and
run its predict / update / innovation / smooth steps, and `kalman_filter_run`
for a whole `Kalman::Filter` pass (forward, optionally RTS) in one call. All
row-major `double` arrays, `0`/`-1` returns and `kalman_last_error()`; no
exception crosses it. `tests/cxx/06_factory.cc` covers the registry, the loader
and the model surface.

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

The *engines* cross too, in the same direction. `NativeEngine("extended")` is a
`KalmanEngine` whose predict / update / smooth run in C++ behind the ordinary
Python `KalmanFilter` (records, diagnostics and plots unchanged), and
`NativeModel.filter_native(zk, engine=..., smooth=...)` runs the whole
`Kalman::Filter` pass fused in one call and hands back a `KalmanRecord`. In a
spec, `"bindings": {"engine": "extended", "backend": "native"}` selects it. Both
need a native model: there is no route from a C++ engine back into a
Python-defined `f`, by design.

**What crosses, and what does not.** One C call per `f`, `h` or coefficient
evaluation — cheap, and in the direction the measurements below say is
affordable. What does *not* cross is a Python callback from inside a C++ loop;
a Python-defined model keeps running on the Python engines, and a C++ model on
either side. The library is found through `KALMAN_LIBRARY`,
`KALMAN_ROOT`, or this checkout's `var/install/lib`; without it the
Python package simply reports `catalog()["native"]["library"] = None`.

**Loading a plugin runs code.** So `api.load_plugin` takes a plugin *name*, not
a path, resolved only inside allow-listed directories (`KALMAN_PLUGIN_PATH`,
default: this checkout's `var/install/lib` and `var/build/lib`); a path is
accepted only when it already lies inside one of them, after `realpath`. It is
gated by its own scope, `load_plugins` — never by `write_models`, which only
ever writes data — and the MCP tool of the same name exposes exactly that.

**Load once, known everywhere.** `load_plugin` also records the plugin name in
the shared registry (see [REST API](10_REST-API.md#the-shared-registry)). The
REST server, the MCP server and every other process on the same state
directory load it again by name, from their own allow-listed directories. This
also occurs after a restart. Subscribers get a `plugin.loaded` event that lists
the new models. Thus a model written in C++ (in a ROOT+ analysis, for
example) is published with one call, `POST /plugins {"name": "KalmanPlugin_mine"}`,
and Unity, agents and other users can then use it as a spec base.

### Models that are their matrices

A model written in Python that is nothing but its matrices needs no twin
written for it: the framework builds the C++ model itself
(`Kalman::Models::Tabulated`) and runs the whole pass on it.

| Python model | Runs as |
|---|---|
| constant F, H, Q, R on the linear engine | the linear kernel, as before |
| constant matrices on the extended engine (a plain `KalmanModel(F=, H=)`) | `Tabulated`, C++ extended engine |
| F, H, Q or R given as functions of time | `Tabulated` with tables: each function is called once per sample of the run, in the loop's order, and the pass runs in C++ |
| a C++ model used directly (`NativeModel`, a model opened from a `.kw` file) | its own C++ instance, the whole pass in one call |

The calorimeter models run this way. Their matrices change from layer to
layer; they used to read the layer from a count kept inside `R()`, and now read
it from the time the filter passes, which is what makes them functions of time
(and exportable to a `.kw` file with tables). Measured
(`tests/test_native_matrices.py`): the electromagnetic and hadronic
reconstructions of notebook 06 agree with the Python loop to 9e-17 relative;
the power-grid model (a time-varying H) is identical to the bit, 60 ms to
22 ms over 2000 samples. For the calorimeter the time barely changes (61 to
56 ms for 26 reconstructions): building the per-layer matrices is the cost, and
that is the model's own Python.

A time-varying B or L, or a noise coupling M on the measurement, is not
tabulated: such a model runs the C++ loop and is asked for its matrices at each
step.

### Named arrays

A twin is configured through named scalars (`DeclareParameter`). What is not
one number, such as the orbit of each satellite, goes through named arrays:

```cpp
DeclareArray("eccentricity", {0.01, 0.01, 0.01});   // constructor
const std::vector<double> &e = Array("eccentricity");
model.SetArray("eccentricity", {0.02, 0.02});       // calls ParametersChanged()
```

An array may change length when it is set. A model whose size follows from
it resizes itself in `ParametersChanged()` with `ResizeMeasurement(m)`, and
says in `CheckShapes()` when arrays that belong together disagree, so the
filter refuses to run instead of reading past an end. The GPS receiver does
both: it has one pseudorange per satellite and six arrays of orbital
elements.

On the Python side a twin gives them with `native_arrays()`, beside
`native_parameters()`; a `NativeModel` has `arrays()` and `set_array()`. In C:
`kalman_model_array_count`, `_array_name`, `_array_size`, `_array`,
`_set_array`. In a model file and in a settings file an array is a list under
the same key as the parameters, and `kalman model NAME` shows its length and
first values.

The GPS twin against the Python reference, 600 s at 1 Hz
(`tests/test_native_gps.py`):

| Configuration | Channels | State, largest difference | Covariance, relative | Python | Twin |
|---|---|---|---|---|---|
| 8 fixed satellites | 8 | 3.3e-9 m (of 6.4e6 m) | 2.1e-16 | 136 ms | 37 ms |
| 24 satellites rising and setting (10 degree mask) | 24, 9988 pseudoranges missing | 3.7e-9 m | 1.1e-16 | 138 ms | 49 ms |
| 5 satellites at 2 Hz | 5 | 3.7e-9 m | 1.6e-16 | 247 ms | 64 ms |

Track finding already ran in the C++ combinatorial engine, calling back into
the Python model for each propagation and each Jacobian. With the
drift-chamber twin it stays in C++ throughout. 12 layers, 6 tracks, 10
background hits a layer, 10% of hits missing (`tests/test_native_tracking.py`):

| | Time | Against the Python search |
|---|---|---|
| Python search | 1878 ms | |
| C++ engine, Python model called back | 1884 ms | same hits on every track; states within 1.2e-6 cm |
| C++ engine, C++ twin | 20 ms | same hits on every track; states within 1.0e-6 cm |

## Model files: `.kw`

A model can be written to one file and read back, by the same code for every
model (`Kalman::IO::ModelFile`, through the `Kalman::Model` interface; nothing
is written per model).

```sh
kalman export constant-velocity -o cv.kw        # a model of matrices: written whole
kalman export ballistic -o ballistic.kw         # a model given by code: a reference
kalman export ballistic --linearize -o tangent.json
kalman export car-speed --set mass=1200             # model.json on the standard output
kalman model cv.kw                              # what a file holds
```

### What a file can hold

Code cannot be put in a file, so what is written depends on what the model is
made of. A model says so with `SetStructure()`; `Models::Linear` declares
`Matrices`, `Models::Extended` `Maps`, and a model that says nothing is `Maps`,
so it is never written out as if its matrices were all of it.

| `Model::Structure()` | The file (`kind`) | Reads back |
|---|---|---|
| `Matrices`: constant F, H, B, L, Q, R | `complete`: all of it (A and Qc instead of F and Q when stated in continuous time) | anywhere, with no plugin |
| `TimeVarying`: matrices that change with time | `complete` over `--steps N` samples: the matrices that change, as tables | anywhere, with no plugin |
| `Maps`: `f` and `h` are code | `reference`: the registered name, its plugin, parameters, Q, R, x0, P0, fs | where that plugin is installed |
| any, with `--linearize` | `linearization`: the Jacobians at the initial state, as a `Matrices` model | anywhere; valid near that state only |

A linear model that overrides `F(t)`, `H(t)`, `Q(t)` or `R(t)` with something
that depends on `t` declares `SetStructure(StructureKind::TimeVarying)`; one
that overrides `f()` or `h()` declares `Maps`.

### The format

A `.kw` file ("Kalman World") is one zip archive, so the parts of a model cannot be
separated or mixed with another model's, and any zip tool opens it:

| Entry | Holds |
|---|---|
| `model.json` | the readable part: `format` (`"kalman-model"`), `format_version`, `kind`, `origin` (name, plugin, title), `structure`, `dynamics`, the `state` and `measurement` components with their units, and `spec` |
| `tables/<F,H,Q,R>.f64` | one matrix per sample for a matrix that changes with time: raw little-endian float64, row-major, of the shape `[steps, rows, columns]` given in `model.json` |

`spec` is exactly what the JSON API takes (`kalman.api.make_model`, the REST
and MCP servers): `base`, `fs`, `dynamics`, `noise`, `initial`, `labels`,
`bindings`. Numbers are written so that a double reads back unchanged.

`model.json` is always stored as it is, so the readable part stays readable.
The tables are compressed with zlib when the library is built with it
(`KALMAN_USE_ZLIB`, on by default; found on the system, optional) and stored
otherwise. A build without zlib reads stored tables only and says so when it
meets a compressed one. Every entry carries a CRC-32, checked on reading.

A file with no table may also be written as plain JSON, to a name ending in
`.json` (the `model.json` alone). A file with tables is refused that way,
since its JSON would not be the whole model, and a `model.json` whose tables
are missing is refused on reading.

`.kw` is meant as the one container of Kalman World: `format` in `model.json`
says what a file holds, so other kinds (records, controllers) can use the same
archive later without a new extension.

### Settings: `name.kw.json`

A `.kw` file holds the model with the settings it was written with. To move
away from them without touching it, a settings file sits beside it under the
same name plus `.json`:

```sh
kalman export car-speed -o car.kw
kalman settings car.kw                    # writes car.kw.json: the values of car.kw, to edit
kalman settings car.kw --set mass=1200    # or change a parameter in it from the command line
kalman model car.kw                       # opened with its settings; says what they change
kalman model car.kw --no-settings         # the model file alone
```

```json
{
  "format": "kalman-settings",
  "format_version": 1,
  "for": "car-speed",
  "fs": 20.0,
  "parameters": {"mass": 1200.0, "power": 150000.0},
  "noise": {"R": [[4.0]]},
  "initial": {"x0": [0.0, 0.0]}
}
```

| Rule | Why |
|---|---|
| It holds only `fs`, `parameters`, `noise` (`Q`, `R`) and `initial` (`x0`, `P0`); what it leaves out keeps the value of the `.kw` | it is settings, not a model |
| Any other key is refused, with the list of settings | it cannot change the matrices or the code, so it cannot turn one model into another |
| `for` names the model it was written for; settings for another model are refused | a settings file copied beside the wrong `.kw` is noticed, not applied |
| A parameter the model does not have is refused | a typo does not pass in silence |
| A whole model is not written to a name ending in `.kw.json` | that name is the settings file's |
| A constant `Q` or `R` for a matrix the `.kw` tabulates is refused | the table would be used and the constant ignored |

`kalman model`, `kalman export FILE`, `Kalman::IO::ModelFile::Open()` and
`kalman_model_load()` apply the settings file when there is one. Exporting an
opened file writes the settings in, which is how to freeze them into a new
`.kw`. `kalman.native.read_model_file` returns them under `"settings"` without
applying them.

### What is checked

`tests/cxx/17_modelfile.cc`, on 60 samples filtered by the model and by the
model read back from its file:

| Model | File | Largest difference in the estimates and their variances |
|---|---|---|
| constant matrices | archive, and plain JSON | 0 (to the bit) |
| time-varying R | archive with a table | below 1e-12: the table is taken at `t0 + k dt`, the filter's clock adds `dt` k times |
| pendulum (maps) | reference | 0 (to the bit) |

A changed byte in the archive fails its checksum; a reference to a model that
is not registered names the plugin to load; another format, or a later
version, is refused.

### From C++, C and Python

```cpp
#include "Kalman/IO/ModelFile.h"

Kalman::IO::ModelFile::Options options;
options.registered = "car-speed";                     // for a reference
Kalman::IO::ModelFile::Export(model, options).Save("car.kw");
std::unique_ptr<Kalman::Model> back = Kalman::IO::ModelFile::Load("car.kw").Import();
```

From C: `kalman_model_export_json`, `kalman_model_save`, `kalman_model_load`,
`kalman_model_open`, `kalman_model_structure`, and to build a model of matrices
by hand `kalman_model_create_matrices`, `kalman_model_set_matrix`,
`kalman_model_set_table`, `kalman_model_set_component`.

### From Python

Every model has `export()` and a `structure`, whether it is written in C++ or
in Python, and `kalman.native.open_model()` gives back a model to filter with:

```python
import numpy as np
import kalman.native as native
from kalman.framework.model import KalmanModel

model = KalmanModel("cv", fs=10, F=F, H=H, Q=Q, R=lambda t: R0 * (1 + t))
model.structure                          # KalmanStructureKind.TIME_VARYING
model.export("cv.kw", steps=200)         # R goes in as a table
back = native.open_model("cv.kw")        # tables and cv.kw.json (settings) applied
record = back.filter(z)
```

A Python model is not written by Python code: its matrices, tables and names
are handed to C++, and the one exporter writes the file.

| Python model | `structure` | `export()` writes |
|---|---|---|
| built from matrices (`F=`, `H=`), nothing overridden | `MATRICES` | `complete` |
| the same with `F`, `H`, `Q` or `R` given as a function of time | `TIME_VARYING` | `complete` with tables; needs `steps=` and a `.kw` path |
| a class with its own `f` / `h` (rockets, space, lane, ...) | `MAPS` | `reference` to the class by name, or `linearize=True` for its Jacobians at x0 |
| a C++ model (`NativeModel`) | what the C++ class declares | as `kalman export` |

A class may declare `STRUCTURE = KalmanStructureKind...` instead of having it
read from the instance.

`open_model` returns a C++ model (`NativeModel`) for a file of matrices or a
reference to a C++ model, and the Python class itself for a reference to a
Python class. `read_model_file` only reads: `{"document", "tables", "settings"}`.

Measured (`tests/test_kalman_cli.py`, 80 samples, the Python model against the
model opened from its file): constant matrices 0; a time-varying R below
1e-12; the registered C++ `fading-sensor` of the example plugin, 50 samples,
below 1e-12.

Limits that remain:

- **A reference to a Python class carries the arguments it was constructed
  with**, defaults included: every model records them when it is built
  (`model._init_arguments`). Numbers, strings, kinds (by name) and arrays of
  numbers are written. An argument that is code or an object (a callable, a
  controller) cannot be; those are listed in `origin.parameters_not_written`
  and the file gives the class's default for them. Every catalogue model that
  builds with its defaults is checked through a file
  (`tests/test_kalman_cli.py`). The class must be in `kalman.api`'s catalogue.
- **`kalman model FILE` cannot build a reference to a Python class** (it says
  so and names the class); open it from Python.
- **`api.make_model(spec)` reads a spec, not a file**: it takes neither tables
  nor a settings file, and the servers do not open files named by a client.
  Use `open_model`.
- **A time-varying B or L is not tabulated** (tables are for F, H, Q, R); such
  a model counts as `MAPS`.

## libKalmanApps: the application models in C++

`libKalman` ships no physics. The physics of `modules/apps/` lives in a
second library, `libKalmanApps` (`apps/include/Kalman/Apps/`, `apps/src/`), which
links the first and registers each model with the factory. `make build` builds
both and they install side by side.

Each model there is the **twin** of one Python class. The Python class stays
the model: its name, its constructor, `plot`, the catalogue entry and the
notebooks are unchanged. What changes is who runs the filtering pass. When the
library is present and the configuration has a C++ counterpart,
`model.filter(z)` hands the whole pass to the twin: the model's `f`/`h`, the
Jacobians and the EKF steps run inside `Kalman::Filter`, in one call. Otherwise
the Python loop runs as before.

| Python class | C++ class | Registered name | Stays in Python |
|---|---|---|---|
| `StormModel` | `Apps::Weather::Storm` | `storm`, `storm-track-only` | a `steering` flow or an `intensity_fn` (Python callables) |
| `BallisticModel` | `Apps::Ballistics::Ballistic` | `ballistic`, `ballistic-order2` | `drag_uncertainty` (speed-dependent Q), `substeps` other than 1, thrust controls, a user `f`/`h` |
| `TornadoEnvironmentModel` | `Apps::Weather::Tornado` | `tornado-environment` | nothing |
| `CarSpeedModel` | `Apps::Vehicles::CarSpeed` | `car-speed` | an integrator other than RK4, sub-stepping |
| `GPSModel` | `Apps::GPS::Receiver` | `gps` | `preprocess()` (the relativistic corrections applied to the pseudoranges before the filter); a `dtype` other than float64 |
| `DriftChamberModel` | `Apps::Tracking::DriftChamber` | `drift-chamber` | `generate_event()` (it draws with numpy's generator) |
| `CarLaneModel` | `Apps::Vehicles::CarLane` | `car-lane` | a steering computed by a controller function (closed loop), an integrator the C++ side lacks, a subclass's own `clip_u` |
| `RocketLiftoffModel` | `Apps::Astro::RocketLiftoff` | `rocket-liftoff` | an integrator the C++ side lacks |
| `RocketAscentModel` | `Apps::Astro::RocketAscent` | `rocket-ascent` | a thrust computed by a controller function, an integrator the C++ side lacks |
| `RocketDescentModel` | `Apps::Astro::RocketDescent` | `rocket-descent`, `rocket-descent-control` (`control=True`) | a command computed by a controller function, an integrator the C++ side lacks |
| `RocketLandingModel` | `Apps::Astro::RocketLanding` | `rocket-landing` | an integrator the C++ side lacks |
| `TransLunarModel` | `Apps::Astro::TransLunar` | `translunar` | an integrator other than the built-in Euler, RK4 or default Dormand-Prince |
| `InterplanetaryModel` | `Apps::Astro::Interplanetary` | `interplanetary` | a mid-course burn (`control=True`, a `TrajectoryControlManeuver`), another integrator, a perturber field not tabulated on `numpy.arange`'s grid |

The ballistic, car and rocket twins are continuous models: each overrides
`Derivative()` and calls `UseContinuousDynamics()`, and its `f` is the
framework's integration of that derivative followed by the model's own closing
step (the acceleration slot, the speed kept non-negative, the mass kept at the
dry mass, a landed stage held on the pad). Their `integrator` and `substeps`
parameters carry the Python model's integrator (Euler, RK4, Dormand-Prince
with the default tolerances: `native.native_integrator_kind`) and sub-steps.

A twin may take a control input (`CarLaneModel`, `RocketAscentModel`,
`RocketDescentModel(control=True)`): `model.filter(z, uk)` hands the twin the
command sequence with the measurements. A controller *function* (closed loop)
is Python, so it keeps the Python model in the C++ loop instead.

The space twins are continuous too, with `substeps` (the Python `n_sub`) and
`integrator` as parameters. The trans-lunar coast needs only numbers: the
gravitational parameters, and the Moon and the Sun on circles given by a
radius, a rate and a phase (an `epoch` with `use_ephemeris=True` only sets the
phases). The interplanetary cruise needs a table: the Python model samples the
perturbing planets from the ephemeris every 2 days when it is built and
interpolates it with `numpy.interp`. The table crosses as named scalars
(`bodies`, `samples`, `grid_step`, then `body<i>_gm` and `body<i>_x<k>`,
`_y<k>`, `_z<k>`), which the twin declares when `bodies` and `samples` are set
and reads back once per change; for the Earth-Jupiter cruise of notebook 08
that is 6,442 scalars, pushed in about 40 ms on each run. The twin
interpolates the way `numpy.interp` does (end values outside the grid).
Positions are in Mm, velocities in km/s and time in ks, as in Python, so
`kalman model` prints the sample rate of these two per ks, not per second.

Measured with `python tools/benchmark_twins.py --repeat 5` (2000 steps, Apple
silicon; the times move with the load on the machine, the ratios do not); the last two columns are the largest difference between the two
runs, relative to the largest entry:

| Model | Python | C++ twin | Faster | State | Covariance |
|---|---|---|---|---|---|
| `StormModel`, NH straight | 183.2 ms | 17.2 ms | 10.7x | 5.7e-09 | 1.9e-07 |
| `StormModel`, NH recurving | 193.6 ms | 17.0 ms | 11.4x | 1.4e-08 | 2.2e-07 |
| `StormModel`, track only | 177.1 ms | 16.2 ms | 10.9x | 1.1e-08 | 1.1e-10 |
| `BallisticModel`, order 3, drag | 985.1 ms | 24.6 ms | 40.0x | 1.4e-09 | 5.0e-08 |
| `BallisticModel`, order 2 | 682.6 ms | 20.7 ms | 33.0x | 3.1e-10 | 2.8e-08 |
| `BallisticModel`, orbital gravity | 1069.5 ms | 25.3 ms | 42.3x | 2.2e-10 | 2.8e-05 |
| `TornadoEnvironmentModel` | 89.8 ms | 10.6 ms | 8.5x | 8.7e-11 | 2.5e-11 |
| `CarSpeedModel` | 220.7 ms | 11.8 ms | 18.8x | 7.8e-12 | 3.2e-08 |
| `CarLaneModel`, steering program | 143.7 ms | 10.2 ms | 14.1x | 4.3e-10 | 5.0e-10 |
| `RocketLiftoffModel` | 212.2 ms | 16.3 ms | 13.0x | 9.1e-13 | 1.4e-09 |
| `RocketAscentModel`, thrust program | 311.5 ms | 19.4 ms | 16.0x | 1.1e-10 | 2.6e-12 |
| `RocketDescentModel`, guidance | 718.3 ms | 29.8 ms | 24.1x | 1.8e-09 | 2.7e-09 |
| `RocketDescentModel`, throttle/gimbal program | 672.4 ms | 30.9 ms | 21.8x | 2.1e-11 | 2.6e-07 |
| `RocketLandingModel` | 293.0 ms | 16.3 ms | 17.9x | 3.1e-11 | 4.8e-09 |
| `TransLunarModel`, free return | 5738.1 ms | 548.1 ms | 10.5x | 2.7e-12 | 1.4e-12 |
| `InterplanetaryModel`, Earth-Mars | 3183.8 ms | 309.2 ms | 10.3x | 6.7e-13 | 4.5e-10 |

(The two space rows were measured later, on a loaded machine, with `--repeat 3`:
in the same run the ballistic order-3 row read 2038.8 ms against 261.1 ms, 7.8x.
They are the notebook's scenarios: the free return at 180 s samples with 4 RK4
substeps, the Earth-Mars cruise of the 2026 window at 12 h samples with four
tabulated perturbers.)

The two runs are not bit-identical, and cannot be. One call of `f` agrees to the
last digit or two, because numpy and the C math library round a cosine
differently. Both sides then difference `f` with a step of `1e-6`, which turns
that last digit into about `1e-9` of noise in each Jacobian. The orbital row
shows the same effect at its worst: a position of `6.7e6` m differenced against
an acceleration step of `1e-6` leaves `1e-3` of noise in that Jacobian column
on *both* sides. An analytic Jacobian would remove it from both.

**How a twin stays the same model.** Nothing is defined twice:

- The twin's scalars are the *named parameters* of `Kalman::Model`
  (`DeclareParameter`, `SetParameter`, `ParametersChanged`), pushed from the
  Python instance by name: `omega`, `beta_drift`, `c_drag` and so on.
- The sample rate, `Q` and `R` are pushed from the Python instance on every
  run, and the initial state by the filter. A change made to the Python model
  after construction therefore reaches the twin.
- `Apps::Model` differences `f` and `h` the way `helpers.numerical_jacobian`
  does (forward, step `1e-6 max(1, |x_j|)`), not with the library's central
  scheme, so the two sides linearise alike.
- A subclass or an instance that redefines `f`, `h`, `F`, `H`, `B`, `L`, `Q` or
  `R` is a different model from the one that was ported, and keeps the Python
  loop. For a continuous model `derivative`, `after_step` and `substeps` count
  too (the class's `f` is kept as `derivative`), and a model adds the helpers
  its dynamics call (`guidance`, `clip_u` ...) to its `_TWIN_CONTRACT`. So
  does any engine other than the plain extended one.

`kalman.framework.models.native.USE_NATIVE_TWINS = False`, or
`KALMAN_NATIVE_TWINS=0` in the environment, switches every twin off. That is how
the "Python" column above is measured.

**Adding a twin.** Write the class under `apps/` deriving from
`Kalman::Apps::Model`, declare its parameters, register it; then mix
`NativeTwin` into the Python class and give it `native_name()` (the registered
name for this configuration, or `None`) and `native_parameters()`. Add its
cases to `tests/test_native_twins.py` and to `tools/benchmark_twins.py`.

## Integrating with root-plus

`examples/cxx/ballistics_root.cc` is the worked demonstrator — a projectile under
gravity and quadratic drag, tracked by an EKF, drawn through ROOT. It is built
only when ROOT is present (`find_package(ROOT QUIET)` in `examples/cxx`), so the
ROOT-free build is unaffected, and it shows the whole integration in one file:

```
$ source var/install/bin/thislib.Kalman.sh
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
to its own unrelated `TKalmanFilter`, with libKalman as an *optional*
dependency of that component:

```
root-plus/
  cmake/Packages/Kalman/FindKalmanFramework.cmake
  include/ROOT/IOPlus/Signal/Kalman/TKalmanRecord.h
  include/libRIOPlus_Signal.Config.h.in      # #cmakedefine KALMANFRAMEWORK_FOUND
  include/libRIOPlus_Signal.LinkDef.h        # the dictionary entry, guarded
  src/Signal/TKalmanRecord.cc
  tests/Signal/17_kalmanrecord.cc
```

`ROOT::IOPlus::TKalmanRecord` takes a `Kalman::IO::Record` and adds the two
things libKalman deliberately does not provide: a `TObject` with a
`ClassDef` so a result goes into a `TFile`, and `TGraph` / `TGraphErrors` /
`TMultiGraph` of the estimate with its band. It stores the covariance *diagonal*
only — a plot and a stored summary want the per-component sigma, and keeping the
full `n × n` per step would dominate the file for no gain; use the
`Kalman::Record` itself when the correlations matter.

`ROOT::IOPlus::TKalmanEstimate` is an alias of it, so either spelling compiles.
It is only a source-level name: ROOT I/O stores the real class, so files written
under either spelling are interchangeable.

When libKalman is not found, `KALMANFRAMEWORK_FOUND` stays undefined
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

libKalman is linked, not loaded. From a ROOT/ROOT+ translation unit:

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
| Engines | Linear, Extended, Unscented, Particle, Ensemble, Combinatorial | the same six |
| Track finding | `CombinatorialKalmanFilter.forward(hits, layer_radii=)`, model `radii` / `seed_state` | `Filters::CKF::Find(hits)`, model implements `Tracking::Detector` |
| Controllers | P, PID, LQR, MPPI, Boost, AdaptiveQR forward to C++; Neural (Torch) stays Python; scipy's DARE is the reference | the same six; LQR by `Algebra::SolveDARE`, which agrees with scipy to 1.4e-11 or better (measured on 8 systems, 4e-14 except for a pole at 0.999); no Neural |
| Closed loop | `forward(zk, uk=controller)`; `model.clip_u` | `Forward(zk, controller)`; `Model::ClipControl` |
| Engines from Python | — | `NativeEngine` behind `KalmanFilter`; `filter_native()` fused |
| Integrators / solvers | Euler, RK4, Dormand–Prince / spectral, finite-difference; `register_*` by name | the same four + two; a custom scheme is an `Integrator` subclass handed to `Model::SetIntegrator` or a `Solver` |
| Noise of a continuous model | `Qc=`, `Lc=` (with `A=` for a linear model) | `SetNoiseIntensity(Qc, Lc)`, `SetGenerator(A)`; decoded by `Utils::VanLoan`, which Python calls |
| ODE models | `DYNAMICS = CONTINUOUS`, `f` returns dx/dt; `DEFAULT_INTEGRATOR`, `SOLVER_SUBSTEPS` | override `Derivative()`, call `UseContinuousDynamics()`, `SetIntegrator()`, `SetSubsteps()` |
| Spectral solver backend | the numpy backend forwards to `Solvers::Vorticity` when the solver states its physics; `backend="torch"` runs the C++ libTorch solver below | built-in FFT, one field at a time (`Solvers::Spectral`, `Solvers::Vorticity`); libTorch, batched, CPU/CUDA/MPS (`Learn::Solvers::Vorticity`) |
| Non-additive noise (UKF) | augmented-state sigma points, when the map really uses the noise argument | the same (`HasProcessNoiseInput`, `HasMeasurementNoiseInput`) |
| Ensemble hooks | `model.forecast(X)`, `model.observe(X)`, `model.localization()`, `model.initial_ensemble()`; the analysis arithmetic is the C++ kernel `enkf_analysis` | per-member `f`/`h`; `SetLocalization(rho_xy, rho_yy)` on the engine |
| Routers | Bayesian, Threshold, Hybrid forward to C++; Neural trains in Torch | Bayesian, Threshold, Hybrid (a Python child is called back), Neural (a trained policy: `NN::MLP`, ONNX, TorchScript); IKF |
| Applications | `modules/apps/` | `libKalmanApps`: `StormModel`, `BallisticModel`, `TornadoEnvironmentModel`, `CarSpeedModel`, `TransLunarModel`, `InterplanetaryModel` so far; none in the core library |
| Jacobians | `DIFFERENTIATOR`: analytic, forward (default) or central differences, computed in C++ | `Model::SetDifferentiator`; `Differentiators::Forward`, `::Central` |
| Sampling | `model.sample()`: the C++ loop, numpy's draws | `Kalman::Sampler` (its own generator) |
| JSON API, REST, MCP | `kalman.api`, `kalman.rest`, `kalman.mcp` (all verbs, Python-written models included) | `libKalmanNet`, `kalman-serve` (C++ models; see [Net](18_Net.md)) |
| Missing data | NaN channel skipped | NaN channel skipped (same) |
| Covariance update | Joseph form | Joseph form (same) |
| Smoother | RTS | RTS (same) |
| Singular `S` | `np.linalg.pinv` drops null directions | eigenvalue pseudo-inverse, same semantics |
| Diagnostics | innovations, gains, NIS, NEES | NIS, NEES |

Every engine kind resolves on both sides; a value that is not a kind raises
rather than silently substituting a different algorithm.

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

That is exactly why the Python package's native path
(`modules/framework/native/`) engages *only* where it can eliminate callbacks
entirely — the arithmetic of a step once the model has been evaluated, and,
for constant matrices with no control callable, all N steps fused into one
call. The limit is the language boundary, not the language.

Those kernels are now this library's own `kf_*` / `ukf_*` / `pf_*` / `diag_*`
entry points (`src/Kernels.cc`): a fused linear pass *is* a `Kalman::Filter`
over `Engines::Linear`, an update *is* `Utils::Correct`, so a Python model here
and a C++ model in ROOT+ run one implementation. The Python package used to
carry a hand-rolled C copy of the same arithmetic on stack arrays, which was
faster — 21× on the linear forward pass against numpy, where the C++ classes
with their heap-allocated `Matrix`/`Vector` temporaries measure 2.4× (2.6×
geometric mean over the kernels, `tests/bench_native.py`). One implementation
was chosen over the second copy; the speed is recoverable inside the C++
types (small-buffer storage for the n ≤ 16 case) without touching either side
of the seam, and that is the next performance step, not a design limit.

So the seam sits where a crossing is paid once per evaluation *into* C++ — a
compiled model evaluated by a Python engine — and never once per step *out* of
it. The portable artifacts between the two sides are a **registered model**
(through the factory), **data** (a record) and a **trained ONNX policy**.

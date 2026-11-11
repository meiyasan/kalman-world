# Models

A model answers **WHAT is the system**: the dynamics `f`, the observation `h`,
and the noise covariances `Q` and `R`. The Jacobians `F/H/B/L/M` are optional
refinements — the base class computes them numerically if you do not supply
them.

There is one contract, `kalman.framework.KalmanModel`, and three ways to fill
it in. All three are runnable by **any** engine.

---

## Path 1 — matrices

Pass the discrete matrices. `f(x) = Fx + Bu + Lw` and `h(x) = Hx + Mv` are then
derived for you, and `F()`, `H()` … return what you passed.

```python
import numpy as np
from kalman.framework import KalmanModel

dt = 0.01
model = KalmanModel("cv", fs=1/dt,
                    F=np.array([[1, dt], [0, 1]]),
                    H=np.array([[1.0, 0.0]]),
                    Q=np.diag([1e-6, 1e-4]),
                    R=np.array([[0.05]]),
                    x0=np.array([[0.0], [1.0]]))
```

`F`, `H`, `B`, `L`, `M` may each be a constant array **or a callable** of `t`
(`M(t)`) for time-varying systems.

## Path 2 — continuous

Pass the continuous generator `A` (and, for process noise, `Qc` with an optional
`Lc`) instead of `F`/`Q`. The model discretizes exactly: `expm(A·dt)` for `F`,
Van Loan for `Q`. This needs `fs=`.

```python
model = KalmanModel("cv", fs=100.0,
                    A=np.array([[0, 1], [0, 0]]),      # ẋ = Ax
                    Qc=np.array([[0, 0], [0, 1e-2]]),  # continuous process noise
                    H=np.array([[1.0, 0.0]]),
                    R=np.array([[0.05]]))
```

`A=` and `F=` are mutually exclusive, as are `Qc=` and `Q=`. The continuous
inputs stay available for inspection through `model.Ac()`, `model.Qc()`,
`model.Lc()` — useful with `helpers.symbolic_discretize(model.Ac())`.

This is the path to prefer whenever the physics is written as an ODE: you state
the continuous system once and the discretization is exact rather than a
first-order approximation you maintain by hand.

## Path 3 — functions

Subclass and override the nonlinear maps. `F`/`H` then come from
`numerical_jacobian` unless you also provide analytic ones.

```python
from kalman.framework import KalmanModel
from kalman.framework.constants import KalmanEngineKind

class PendulumModel(KalmanModel):
    NAME = "Pendulum"
    DESCRIPTION = "Nonlinear pendulum, angle observed."
    VERSION = "1.0"
    DEFAULT_ENGINE = KalmanEngineKind.EXTENDED

    def __init__(self, sensor_id="p", g=9.81, L=1.0, **kw):
        self.n = 2                       # BEFORE super(): construction evaluates F(zeros)
        self.g, self.L = g, L
        super().__init__(sensor_id, **kw)

    def f_continuous(self, x, u=None, t=None):
        theta, omega = x[0, 0], x[1, 0]
        return np.array([[omega], [-(self.g / self.L) * np.sin(theta)]])

    def h(self, x, v=None, t=None):
        return x[:1]                     # observe the angle
```

Two rules that bite here:

- **Set `self.n` before `super().__init__()`.** Construction runs
  `_check_shapes`, which evaluates `F(zeros)` and therefore `f` at `x = 0`.
- **Prefer overriding `f_continuous` over `f`.** `f_continuous` returns the
  derivative `ẋ`; the model's bound integrator turns it into the discrete step.
  Override `f()` directly only when the discrete map is genuinely what you have.
  `dt` is never passed into `f()` — only absolute `t` is. Derive
  `dt = 1/self.fs`.

---

## What a model may also define

| Member | Purpose |
|---|---|
| `NAME` / `DESCRIPTION` / `VERSION` | metadata surfaced by `describe()` and every transport |
| `DEFAULT_ENGINE` / `ALLOWED_ENGINES` | which estimator(s) this system admits |
| `DEFAULT_INTEGRATOR` / `ALLOWED_INTEGRATORS` | time-stepping scheme, see [page 6](6_Integrators-and-Solvers.md) |
| `DEFAULT_SOLVER` / `ALLOWED_SOLVERS` / `SOLVER_SUBSTEPS` | spatial discretization for field models |
| `preprocess` / `inprocess` / `postprocess` | signal conditioning hooks around the filter loop |
| `states` / `observables` | labelled state and measurement enums, used by plots — synthesized as `X0…Xn-1` / `Z0…Zm-1` when not declared |
| `clip_u()` | actuator limits, honoured by the control-cost accounting |
| `control_cost()` | default `Qcost`/`Rcost` for controllers |
| `plot(record, sample=…)` | the model's signature figure |

Engines consume only what they need: UKF and PF use `f/h/Q/R`; the EKF also
reads `F/H`; the linear KF reads `F/H` as constants.

---

## Composite models

`CompositeKalmanModel` combines N submodels. State evolution is always
block-diagonal — submodels evolve independently. Only the *measurement*
composition varies, set by `obs_mode`:

| Mode | Measurement vector | Use case |
|---|---|---|
| `INDEPENDENT` | `z = [h₁(x₁); h₂(x₂); …]` stacked | separate sensors, one measurement each |
| `FUSED` | `z = Σ hᵢ(xᵢ)` summed | all submodels contribute additively to one observable |

```python
from kalman.framework import CompositeKalmanModel
from kalman.framework.constants import KalmanObservableReadout

composite = CompositeKalmanModel(m1, m2, obs_mode=KalmanObservableReadout.INDEPENDENT)
```

Only operators touching `z` are mode-aware (`h`, `H`, and `preprocess`, which
splits by `m.m` in INDEPENDENT mode and passes through in FUSED). `f`, `F` and
`Q` are always block-diagonal; `inprocess` and `postprocess` are identity.
Override `_bind_preprocess` / `_bind_inprocess` / `_bind_postprocess` in a
subclass to inject domain logic while keeping the composite wiring intact — see
the worked example in [`../README.md`](../README.md).

---

## The shipped catalog

46 model classes are discoverable through `kalman.api.catalog()["models"]`.

| Family / domain | Models |
|---|---|
| **framework** | `KalmanModel`, `LinearKalmanModel`, `ExtendedKalmanModel`, `UnscentedKalmanModel`, `CompositeKalmanModel` |
| **astro.rocket** | `RocketAscentModel`, `RocketDescentModel`, `RocketLiftoffModel`, `RocketLandingModel` (+ `RocketLaunchModel`, `RocketTakeoffModel` legacy aliases) |
| **astro.space** | `TransLunarModel`, `InterplanetaryModel` |
| **ballistics** | `BallisticModel`, `BallisticLinearModel` |
| **gravitational.gravwave** | `PolarizationModel` |
| **gravitational.interferometer** | `InterferometerModel`, `InterferometerK1/H1/L1/V1` |
| **resonators.oscillators** | `OscillatorModel`, `AnalyticOscillatorModel`, `OscillatorBankModel` (alias `OscillatorCompositeModel`), `HarmonicOscillatorModel`, `InharmonicOscillatorModel`, `PhasorOscillatorModel` |
| **resonators.powergrids** | `PowerGridStateModel` |
| **resonators.suspensions** | `SuspensionModel` |
| **weather.flow** | `KolmogorovFlowModel` |
| **weather.storms** | `StormModel` (one model for typhoons, hurricanes and cyclones; `hemisphere=` is the only basin difference), plus `TornadoEnvironmentModel` |
| **tracking** | `DriftChamberModel` |
| **vision** | `DetectionModel`, `BoxTrackModel` |
| **vehicles** | `CarSpeedModel`, `CarLaneModel` |
| **fusion** | `SensorFusionModel` |
| **gps** | `GPSModel` |
| **calorimeter** | `CalorimeterLinearModel`, `CalorimeterElectromagneticModel`, `CalorimeterHadronicModel` |

Inspect any of them without reading the source:

```python
from kalman.api import describe
describe("StormModel")
# {'name': 'StormModel', 'description': …, 'version': …, 'params': [...],
#  'default_engine': 'extended', 'default_integrator': 'rk4', …}
```

---

## Adding a model to the repository

- **A new variant in an existing domain** — add `apps/<domain>/<variant>.py`, put
  shared enums/helpers in that domain's `base.py`, then register it in the domain
  `__init__` (`import`, `VARIANTS`, `__all__`). Nothing else changes.
- **A new domain** — create a package with per-variant submodules, an `__init__`
  re-exporting its public names, and a `VARIANTS` dict. Flat `apps/*.py` modules
  are not accepted; `tests/test_structure.py` enforces this.
- **From a spec, without writing code** — `kalman.api.scaffold()` emits a
  reviewable `modules/apps/<name>.py` source string from a `KalmanSpec`. It only
  ever *emits*; a human commits. See [Python API](9_Python-API.md).

Full conventions for contributors are in [`../AGENTS.md`](../AGENTS.md).

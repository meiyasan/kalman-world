# The Kalman contract

One vocabulary for the whole library, in C++ and in Python. This page is the
reference: every public term, what it means, who owns it, and how the words of
the ensemble filter and of track finding map onto it.

The **Status** columns say where the code stands: *in place*, or the step of
the consolidation that brings it. Until a step lands, the code still carries
the older name given in the last table.

## The rules

1. **The contract speaks control engineering:** `f`, `h`, predict, update,
   prior, posterior, measurement, initial state. That is what a model, the
   base engine and the filter expose.
2. **`f` and `h` take one state or many.** There is no second pair of names for
   the many-states form. `step` and `sense` are their readable aliases; a
   model's author writes `f` and `h` only.
3. **A community's words live inside its filter.** The ensemble filter keeps
   forecast, project, analysis, localization, inflation, members. Track finding
   keeps hit, layer, seed, hole, candidate, chi-square gate. Each such name is
   implemented by the generic method and carries a comment that names it.
4. **No speed is lost.** A path that is batched or multi-threaded stays so, and
   is measured before and after a change.
5. **Same names in C++ and Python** (CamelCase and snake_case). C++ is the
   implementation; Python parses arguments and wraps.
6. **A replaced public name is removed**, not kept as an alias.

## The model: what the system is

`f` moves the state one step, `h` says what the sensors read:

```
x(k+1) = f(x(k), u(k), t(k)) + w(k)        w: process noise, covariance Q
z(k)   = h(x(k), t(k))       + v(k)        v: measurement noise, covariance R
```

`x` is the state, `u` the command applied, `t` the time of the step (absolute
time, in the model's unit; the step length is never passed, a model derives it
from its sample rate), `z` the measurement.

| Term | Meaning | Python | C++ | Status |
|---|---|---|---|---|
| `f` | where the state goes next: state in, state out | `model.f(x, u, t)` | `model.f(x, u, t)` | in place |
| `h` | what the sensors read at a state: state in, measurement out | `model.h(x, t)` | `model.h(x, t)` | in place |
| `f`, `h` on many states | the same two functions given N states side by side; a model declares when its own code handles the batch | `model.f(X, u, t)`, `model.h(X, t)` | overloads of `f` and `h` | in place |
| `step`, `sense` | readable aliases of `f` and `h`; never written by a model | `model.step(...)`, `model.sense(...)` | `model.Step(...)`, `model.Sense(...)` | in place |
| `Q`, `R` | process and measurement noise covariances | `model.Q(t)`, `model.R(t)` | `model.Q(t)`, `model.R(t)` | in place |
| initial state | where the system starts and how sure that is (the prior): a mean and a covariance, or N samples of it | `model.initial_state(x0, P0)`, `model.initial_state(samples=N)` | `model.InitialState(x0, P0)`, `model.InitialState(N, normal)` | in place |
| states | the components of the state, with units | `model.states` | `model.States()` | in place |
| observable | a quantity that can be measured, with its unit | `model.observables` | `model.Observables()` | in place |
| sensor | what acquires a measurement: what it reads, its noise, its observables | `KalmanSensor`, `model.sensors` | `Kalman::Sensor`, `model.Sensors()` | in place |
| instrument | a set of sensors and what they share: calibration, pose, rate, delay | `KalmanInstrument`, `model.instruments` | `Kalman::Instrument`, `model.Instruments()` | step 12 |
| measurement | the numbers that come out of the sensors | `zk` | `zk` | in place |
| parameter | a named number of the model, which can be estimated | `model["c_drag"]` | `model.Parameter("c_drag")` | in place |

## The engine and the filter: how it is estimated

| Term | Meaning | Owner | Python | C++ | Status |
|---|---|---|---|---|---|
| predict | advance the belief one step in time; uses `f` | engine | `engine._predict` | `Engine::Predict` | in place |
| update | correct the belief with a measurement; uses `h` | engine | `engine._update` | `Engine::Update` | in place |
| prior, posterior | the belief before and after a measurement | engine | record flavours | `StateFlavor` | in place |
| innovation | the measurement minus its prediction | engine | `filter.innovation` | `Diagnostics` | in place |
| forward | the pass through the whole series: predict and update at each step | filter | `filter.forward(z)` | `Filter::Forward` | in place |
| backward | the smoothing pass, back through the series | filter | `filter.backward()` | `Filter::Backward` | in place |
| reset | restart a run from the initial state | filter, engine | `filter.reset()` | `Filter::Reset` | in place |

## Operations: a model made from another

Each gives back a model that any filter runs, and they nest. In both languages
each exists as a method of the model and as a function.

| Operation | Gives | Python | C++ | Status |
|---|---|---|---|---|
| joint | the model with some parameters estimated along with its state | `model.joint({...})` | `model.Joint({...})` | in place |
| add a sensor | the model read by one more sensor on the same state (sensor fusion) | `model.add_sensor(sensor)` | `model.AddSensor(sensor)` | in place |
| add an instrument | the same for a set of sensors | `model.add_instrument(instrument)` | `model.AddInstrument(instrument)` | step 12 |
| lag | the model carrying its last L states (delayed sensors, fixed-lag smoothing) | `model.lag(L)` | `model.Lag(L)` | step 13 |
| compose | models side by side, each with its own state; also sensors into an instrument | `a.compose(b)` | `a.Compose(b)` | step 14 |
| route | several models of one system, with a probability on each (the IMM of the literature) | `KalmanModel.route(models, Pi)` | `Model::Route(models, Pi)` | step 15 |

## Control: acting on the system

| Term | Meaning | Status |
|---|---|---|
| controller | computes the command `u` from the estimate: proportional, PID, LQR, MPC, MPPI | in place |
| tuner | retunes `Q` and `R` from the innovations; commands nothing | step 8 |

## The three vocabularies

The same mathematics was named three times: by control engineering (Kalman,
1960), by weather forecasting, where the ensemble filter comes from (Evensen,
1994), and by particle physics, where track finding comes from. The contract
uses the first. The other two stay inside their own filter.

### Concepts every filter has

| Concept | Contract | Ensemble filter (weather) | Track finding (particle physics) |
|---|---|---|---|
| what is estimated | state | state | track parameters |
| the data | measurement | observation | hit |
| one step along the data | time step | assimilation cycle | detector layer |
| the whole data set | measurement sequence | observation window | event |
| where it starts | initial state (prior) | initial ensemble | seed |
| the dynamics | `f` | forecast model | propagation, extrapolation |
| the sensor model | `h` | observation operator | projection onto the measurement |
| first half of a step | predict | forecast | propagate to the next layer |
| second half of a step | update | analysis | update with the hit |
| the estimate before the data | prior | background, or forecast | predicted state |
| the estimate after the data | posterior | analysis | filtered state |
| the data minus its prediction | innovation | innovation | residual |
| process noise | `Q` | model error | material effects, multiple scattering |
| measurement noise | `R` | observation error | hit resolution |
| the pass through the data | forward | assimilation | track fit |
| the pass back | backward | smoother | smoothing |

### Concepts only one community has

| Concept | Contract | Ensemble filter | Track finding |
|---|---|---|---|
| many states standing for the uncertainty | (a covariance instead) | ensemble, members | |
| limiting false long-range correlations | covariance mask | covariance localization | |
| keeping the spread from collapsing | | inflation | |
| a test against a known truth | simulation study | twin experiment (OSSE) | Monte Carlo study |
| accepting or rejecting a measurement | validation gate | quality control | chi-square cut |
| a step with no accepted measurement | missed detection | | hole |
| one hypothesis among several | hypothesis | | track candidate |
| choosing between hypotheses | data association | | ambiguity resolution |

Two words that mislead:

- **Localization** in the ensemble filter is not about finding a position. It
  is a table of weights between 0 and 1, computed from distances, laid over the
  sampled covariance so that an observation only corrects what is near it. On a
  model it is called `covariance_mask`.
- **CKF** here is the *combinatorial* Kalman filter of particle physics. In
  control engineering the same letters mean the cubature Kalman filter.

### Where each community's word lives in the code

| The contract's method | Ensemble engine | Track finding |
|---|---|---|
| `model.f` on the members | `engine.forecast(model)`: calls it, then draws the process noise | propagates a candidate to the next layer |
| `model.h` on the members | `engine.project(model)`: projects the members into measurement space | scores a hit against the prediction |
| update | `engine.analyze(model, z)` | the update inside the chi-square gate |
| `model.initial_state(samples=N)` | `engine.initial_ensemble(model)` | `model.seed_state(hit)`, one start per hit |
| `model.covariance_mask()` | `engine.localization` | |

What only one engine needs from a model is a small mixin beside the contract:
`covariance_mask` for the ensemble filter, `radii` and `seed_state` for track
finding. These are the classes `KalmanEnsembleModel` and `KalmanDetectorModel`
(`Engines::EnsembleModel` and `Tracking::Detector` in C++), which a model
inherits beside the contract; the engine checks for the class.

## Names being replaced

Already gone: `model.forecast(X)` and `model.observe(X)`, now `f` and `h` given
many states; `model.initial_ensemble(N)`, now `model.initial_state(samples=N)`;
`initial_state_vector`, `initial_state_covariance` and the `get_initial_state`
variants, now `initial_state(x0, P0)`; `model.localization()`, now
`model.covariance_mask()`; `sensor_id`, the first argument of a model, now
`sensor` (a name, a `KalmanSensor`, or a list); the label list (`add_sensor("X")`,
`remove_sensor`, `clear_sensors`, `sensor(i)`), now a sensor's labels.
"Twin" now only means the twin experiment of the weather application: the C++
class that stands for a Python model is its *counterpart* (`NativeCounterpart`,
was `NativeTwin`; `KALMAN_NATIVE_COUNTERPARTS`, was `KALMAN_NATIVE_TWINS`).

| Older name | Contract name | Step |
|---|---|---|
| `Boost`, `AdaptiveQR` as controllers | tuners | 8 |
| readouts `FUSED`, `INDEPENDENT` | `SUMMED`, `STACKED` | 14 |
| `CompositeKalmanModel` | `compose` | 14 |
| `InteractiveKalmanFilter`, IKF | `route` and the interacting engine | 15 |

## Names proposed, to be confirmed

These are clearer names for hooks that work but say little. They are a
proposal: none is applied until confirmed.

| Today | Proposed | What it does |
|---|---|---|
| `preprocess` | `prepare_measurements` | turns the raw input into measurements |
| `inprocess` | `adjust_state` | adjusts the state inside a pass |
| `postprocess` | `finish_output` | adjusts what a pass gives back |
| `gk` | `raw` | the input of a pass before it is prepared |
| `clip_u` | `limit_control` | what the actuator can deliver of a command |
| `KalmanProxyMixin` | `KalmanRunMixin` | gives a model its `filter`, `smooth` and `sample` shortcuts |
| `EnumBis` | `LabelEnum` | the base of the label enums; it would also stop hiding a member's `.name` |

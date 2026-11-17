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
| sensor | what acquires a measurement: what it reads, its noise, its observables, its rate | `KalmanSensor`, `model.sensors` | `Kalman::Sensor`, `model.Sensors()` | in place |
| instrument | a set of sensors and what they share: a name, a rate, a calibration (`"<name>.gain"`, `"<name>.offset"`, two parameters of the model) | `KalmanInstrument`, `model.instruments` | `Kalman::Instrument`, `model.Instruments()` | in place |
| generic sensors | a position reading, a Doppler reading, a camera, for any model that has the components they read: a geometry and a noise, nothing of any product | `PositionSensor`, `DopplerSensor`, `CameraSensor` (`kalman.framework`) | `Kalman::Sensors::Position`, `Doppler`, `Camera` | in place |
| setting | a number a sensor's readings depend on (a focal length, where it is); a parameter `"<name>.<setting>"` of the model that takes the sensor, which `joint` estimates | `sensor.settings` | `Sensor::settings` | in place |
| adapter | one piece of hardware: an instrument of generic sensors with what its data sheet and its mounting say, and where its data comes from. None is shipped | a `KalmanInstrument` subclass, in `kalman.adapters` | a class deriving from `Kalman::Instrument` | written by whoever uses the hardware |
| detector | an instrument whose sensors are read one after the other, each at its `station`: the layers track finding runs through | `KalmanInstrument(sequential=True)`, `chamber.instruments` | `Instrument::sequential`, `Tracking::Detector::Layers` | in place |
| measurement | the numbers that come out of the sensors | `zk` | `zk` | in place |
| parameter | a named number of the model, which can be estimated | `model["c_drag"]` | `model.Parameter("c_drag")` | in place |
| observability | whether the measurements can tell the state components apart, asked before any data | `model.observability(steps)` | `Kalman::Observability(model, steps)` | in place |

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
| add an instrument | the same for a set of sensors | `model.add_instrument(instrument)` | `model.AddInstrument(instrument)` | in place |
| lag | the model carrying its last L states: a sensor with a `delay` reads the state as it was, and block `j` is the state `j` steps ago given what came since (fixed-lag smoothing) | `model.lag(L)`, `KalmanSensor(delay=)` | `model.Lag(L)`, `Sensor::delay` | in place |
| compose | sensors into an instrument | `compose(s1, s2, name=...)`, `s1 \| s2` | `Kalman::Compose({s1, s2}, name)`, `s1 \| s2` | in place |
| compose | models side by side, each with its own state; readout `STACKED` (each part its measurement) or `SUMMED` (one shared); a part's parameter is `"<part>.<name>"` | `compose(a, b)`, `a.compose(b)` | `Kalman::Compose({a, b})`, `a.Compose(b)` | in place |
| route | several models of one system (the *modes*), with a probability on each; run by the interacting engine (the IMM of the literature); the decision layer is a `KalmanRouter` | `route(models, Pi)`, `KalmanModel.route(models, Pi)` | `Kalman::Route(models, Pi)`, `Model::Route(models, Pi)` | in place |

## Control: acting on the system

| Term | Meaning | Status |
|---|---|---|
| controller | computes the command `u` from the estimate: proportional, PID, LQR, MPC, MPPI | in place |
| tuner | retunes `Q` and `R` from the innovations; commands nothing: `KalmanTuner` / `Kalman::Tuner`, given to a filter beside a controller (`tuner=`) | in place |

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

### The operations, and what other fields call them

| What it does | Contract | Elsewhere |
|---|---|---|
| a parameter estimated with the state | `joint` | state augmentation, joint (or dual) estimation |
| one more sensor on the same state | `add_sensor` | sensor fusion |
| a set of sensors with one calibration | instrument, `add_instrument` | sensor calibration, bias estimation |
| a measurement that arrives late | a sensor's `delay`, on a model made by `lag` | out-of-sequence measurement, delayed-state filter |
| the state a few steps ago, given what came since | `lag`, `model.block(j)` | fixed-lag smoothing |
| systems with separate states in one filter | `compose` | decoupled (block-diagonal) systems |
| one system, several regimes | `route` | multiple-model estimation, jump Markov system |

Target tracking has its own words for the last one, kept inside the
interacting engine (`Engines::Interacting`):

| Concept | Contract | Multiple-model tracking |
|---|---|---|
| one of the models of the system | a model given to `route` | mode, regime |
| the algorithm | the interacting engine, run by the ordinary filter | IMM (interacting multiple model) filter |
| every belief blended before the prediction | (inside the engine) | mixing, interaction |
| how likely each model is now | `mode_probabilities` | mode probability |
| who sets those probabilities | router (`KalmanRouter`) | mode probability update |
| how regimes follow each other | transition matrix `Pi` | Markov chain, transition probability matrix |

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
`CompositeKalmanModel` and its readouts `FUSED` / `INDEPENDENT`, now `compose`
(`Composite`) with `SUMMED` / `STACKED`.
`InteractiveKalmanFilter` (IKF) and `Filters::IKF`, now `route` (`Router`,
`Models::Router`) on the interacting engine; its `mu_log` and `nis_log` are
`mode_probabilities` and `nis_history`; the storm models' `ikf_bank` is
`regime_bank`.
"Twin" now only means the twin experiment of the weather application: the C++
class that stands for a Python model is its *counterpart* (`NativeCounterpart`,
was `NativeTwin`; `KALMAN_NATIVE_COUNTERPARTS`, was `KALMAN_NATIVE_TWINS`).

| Older name | Contract name | Step |
|---|---|---|

## Names made clearer

Hooks and helpers that worked but said little were renamed:

| Was | Is | What it does |
|---|---|---|
| `preprocess` | `prepare_measurements` | turns the raw input into measurements |
| `inprocess` | `adjust_state` | adjusts the state inside a pass |
| `postprocess` | `finish_output` | adjusts what a pass gives back |
| `gk` | `raw` | the input of a pass before it is prepared |
| `clip_u` / `ClipControl` | `limit_control` / `LimitControl` | what the actuator can deliver of a command |
| `KalmanProxyMixin` | `KalmanRunMixin` | gives a model its `filter`, `smooth` and `sample` shortcuts |
| `EnumBis` | `LabelEnum` | the base of the label enums; a member's `.name` is its name again |

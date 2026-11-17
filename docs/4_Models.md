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
from kalman.framework.constants import KalmanEngineKind, KalmanDynamicsKind

class PendulumModel(KalmanModel):
    NAME = "Pendulum"
    DESCRIPTION = "Nonlinear pendulum, angle observed."
    VERSION = "1.0"
    DEFAULT_ENGINE = KalmanEngineKind.EXTENDED

    def __init__(self, sensor="p", g=9.81, L=1.0, **kw):
        self.n = 2                       # BEFORE super(): construction evaluates F(zeros)
        self.g, self.L = g, L
        super().__init__(sensor, **kw)

    DYNAMICS = KalmanDynamicsKind.CONTINUOUS     # f returns dx/dt

    def f(self, x, u=None, w=None, t=None):
        theta, omega = x[0, 0], x[1, 0]
        return np.array([[omega], [-(self.g / self.L) * np.sin(theta)]])

    def h(self, x, v=None, t=None):
        return x[:1]                     # observe the angle
```

Two rules that bite here:

- **Set `self.n` before `super().__init__()`.** Construction runs
  `_check_shapes`, which evaluates `F(zeros)` and therefore `f` at `x = 0`.
- **Say what `f` returns.** With `DYNAMICS = KalmanDynamicsKind.CONTINUOUS`,
  `f` is the derivative dx/dt and the framework decodes it into the discrete
  step with the model's integrator (RK4 unless you choose another). With
  `DISCRETE`, the default, `f` is the step itself and is used as written. Write
  physics as continuous; keep discrete for a step that has no derivative behind
  it. The filter sees a discrete step either way, and the Jacobian is taken on
  that step.
- **A continuous model can state its noise in continuous time too.** Pass
  `Qc=` (an intensity, per unit time) and optionally `Lc=` (where it enters)
  instead of `Q=`. The per-step `Q` is decoded for the sample rate in use, so
  the same model works at any `fs`:

  ```python
  model = PendulumModel(fs=50.0, Qc=np.diag([0.0, 0.5]))   # a random torque
  ```

  A linear model does the same with a generator matrix: `KalmanModel(A=..., Qc=..., Lc=...)`
  gives `F = exp(A / fs)` and the matching `Q`. The decoding (the matrix
  exponential and the Van Loan formula) runs in C++, `Kalman::Utils::VanLoan`.
  `R` stays a per-reading covariance: a measurement is taken at an instant.
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
| `prepare_measurements` / `adjust_state` / `finish_output` | signal conditioning hooks around the filter loop |
| `states` / `observables` | labelled state and measurement enums, used by plots — synthesized as `X0…Xn-1` / `Z0…Zm-1` when not declared |
| `limit_control()` | actuator limits, honoured by the control-cost accounting |
| `control_cost()` | default `Qcost`/`Rcost` for controllers |
| `plot(record, sample=…)` | the model's signature figure |

Engines consume only what they need: UKF and PF use `f/h/Q/R`; the EKF also
reads `F/H`; the linear KF reads `F/H` as constants.

---

## Operations: a model made from models

The three paths above write a model from nothing. The operations make one from
models you already have. Each takes models and returns a model that any filter
runs, so they chain:

| Operation | Use it when | Section |
|---|---|---|
| `add_sensor(model, sensor)` | another sensor reads the same state | [Sensors](#sensors) |
| `add_instrument(model, instrument)` | sensors come as a set, with one rate and one calibration | [Instruments](#instruments) |
| `joint(model, {...})` | a parameter of the model is not known | [Joint models](#joint-models-estimating-parameters-with-the-state) |
| `lag(model, L)` | a sensor reports late, or you want the state of `L` steps ago improved by what came after | [A sensor that reports late](#a-sensor-that-reports-late-and-the-lagged-model) |
| `compose(a, b)` | several systems, each with its own state, in one filter | [Composite models](#composite-models) |
| `route(models, Pi)` | one system that changes regime | [`route`](#a-system-that-changes-regime-route) |

Every one is also a method (`model.add_sensor(sensor)`, `a.compose(b)`,
`KalmanModel.route(models, Pi)`) and a class (`Fusion`, `Joint`, `Lag`,
`Composite`, `Router`); the three build the same object. The class is what an
application subclasses, as `OscillatorBankModel(Composite)` does.

```python
radar = shell.add_sensor(KalmanSensor("doppler", doppler, R=1.0))     # one more sensor
model = radar.joint({"c_drag": (1e-3, 1e-6)})                         # and the drag estimated
ExtendedKalmanFilter(model).filter(zk)
```

Three rules hold for all of them:

- **The result keeps what the parts declare.** Models of matrices give a model
  of matrices, on the linear engine; one part that is code makes the result code.
- **Parameters are forwarded by name.** A sensor's or an instrument's are
  `"<owner>.<name>"`, a part's `"<part>.<name>"`, a regime's `"<mode>.<name>"`,
  so `joint` reaches any of them through the operations around it.
- **The arithmetic is C++.** Each is a `Kalman::Models::…` class and a function
  in `Kalman/Operations.h`; the Python object wraps it. A model written in
  Python is called back for its `f` and `h`.

[`examples/python/operations.py`](../examples/python/operations.py) and
[`examples/cxx/operations.cc`](../examples/cxx/operations.cc) run the six on a
cart, with the same names in both languages.

---

## Composite models

`compose` puts several models side by side, each with its own state. The parts
evolve independently, so the state is theirs one after the other and `F` and
`Q` are block-diagonal. What is measured depends on the readout:

| Readout | Measurement | Use |
|---|---|---|
| `STACKED` (default) | `z = [h₁(x₁); h₂(x₂); …]` | each part has its own measurement and brings its sensors |
| `SUMMED` | `z = h₁(x₁) + h₂(x₂) + …` | several signals on one channel: the lines of a spectrum, a signal and its background |

```python
from kalman.framework import compose, KalmanReadout

both  = compose(m1, m2)                                  # also m1.compose(m2), Composite(m1, m2)
lines = compose(m1, m2, readout=KalmanReadout.SUMMED)    # one measurement, R of the first part
both.part("m2")                                          # where the second part sits in the state
```

A parameter of a part is `"<part>.<name>"` (the part goes by its sensor's
name, or by `names=[...]`), so `joint` reaches it:
`both.joint({"m1.damping": (0.5, 1e-4)})`. Models of matrices side by side are
a model of matrices, on the linear engine; one part that is code makes the
composite code, each part linearising itself. Parts of the same kind share
their state names, so `record.plot(component="position", dim=2)` picks the
third part's. The arithmetic is `Kalman::Models::Composite`.

This is not `add_sensor`, where a second sensor reads the *same* state.

---

## A system that changes regime: `route`

`route(models, Pi)` makes one model of several models of the *same* system,
one per regime, with a probability on each; `Pi` is the Markov matrix that
says how likely the system is to go from one regime to another in a step.
Unlike `compose`, the models share the state and the measurement.

```python
from kalman.framework import route

car = route([dry, icy], Pi)                  # also KalmanModel.route([...], Pi), Router([...], Pi)
record = car.filter(zk)                      # the combined estimate
car.mode_probabilities                       # (T, 2): which regime, when
```

The filter is the ordinary one: the routed model asks for the interacting
engine (the Interacting Multiple Model of the literature), which keeps one
belief per regime. The decision layer (`router=`) and its training are in
[Control and Routing](8_Control-and-Routing.md). A parameter of one regime is
`"<mode>.<name>"`. There is no smoother.

---

## Joint models: estimating parameters with the state

`model.joint(...)` (also `joint(model, ...)` and the class `Joint`) estimates some of a model's parameters along with its
state. Each parameter the inner model keeps (`model["c_drag"]`) can join the
state as a constant that may drift a little:

```
x = [ s ; theta ]                 s: the inner state,  theta: the parameters
s_{k+1}     = f(s_k, u_k, t ; theta_k)
theta_{k+1} = theta_k + w,        w ~ N(0, drift^2)
z_k         = h(s_k, t ; theta_k)
```

```python
from kalman.apps.ballistics import BallisticModel
from kalman.framework import joint

shell = BallisticModel("XYZ", order=3, fs=32, m=4.2, c_drag=0.0,
                       measurement_uncertainty=10, process_uncertainty=0.3)
model = shell.joint({"c_drag": (1e-3, 1e-6)})   # (initial std, drift per step); or joint(shell, {...})
model.initial_state([[100, 100, 0], [200, 100, 0], [300, 100, -9.81]])   # the 9 states of the shell
record = model.filter(zk)                                      # record.xk[:, 9] is c_drag
```

| | |
|---|---|
| Initial state | the inner `x0`, then each parameter's current value; `P0 = diag(P0_inner, sigma^2)` |
| `initial_state(x0, P0)` | takes the joint state, or one the inner model takes (handed to it, the parameters start again from its values) |
| `f`, `h` | set the inner parameters from the state, evaluate, put the previous values back |
| `F`, `H` | by blocks: the inner model's own Jacobian, the parameter columns differenced with its scheme, `[0, I]` below |
| `Q`, `R` | `diag(Q_inner, drift^2)` and `R_inner`; a `Q` that takes the state gets the inner state and the parameters |
| Engine | the inner model's, or the extended engine when the inner model is linear |

A parameter given a time `tau` wanders and returns to its starting value
instead of drifting for ever, which suits a sensor bias or a clock offset:
`{"offset": (sigma, drift, tau)}`, or `{"sigma": ..., "drift": ..., "tau": ...}`.
Its variance then settles at `drift^2 / (1 - exp(-2 dt / tau))`.

**Can the data tell them apart?** `model.observability(steps=50)` answers
before any data: how many directions of the state the measurements determine
along the model's path from its initial state, and for each state component
its share (0 to 1) in the directions they do not. A gain read together with
the amplitude it multiplies is the classic case: `z = gain * a` fixes the
product, not the two.

```python
report = gauge.joint({"gain": (0.5, 0.0)}).observability()
report["rank"], report["observable"]      # 1, False
report["weakness"]                        # {"X0": 0.36, "gain": 0.64}: one direction, shared by both
```

A joint model runs this check when a pass starts and warns when an estimated
parameter lies in such a direction: its estimate would rest on its prior. The
answer is local: a system at rest can hide a parameter that motion reveals, so
it is asked at the initial state the pass starts from.

A parameter that changes neither `f` nor `h` is refused when the model is
built: a model that reads it only in its constructor would leave the
estimate on its prior. The model is `Kalman::Models::Joint`, and the pass
runs in `Kalman::Filter`: around the inner model's C++ counterpart when it has one
that declares the parameters, otherwise around a callback model that calls the
inner model's Python methods and sets the parameters back on it. The inner
model is taken as it is when the joint one is built. Notebook 01, Step 7
estimates the drag coefficient of a shell this way.

---

## Sensors

A **sensor** is what acquires a measurement: what it reads of the state, how
noisy it is, and what its channels are called. Each channel is an
**observable**, a quantity that can be measured, with its unit; the
**measurement** is the numbers that come out. A model lists its sensors:

```python
shell = BallisticModel("radar", order=3, fs=32, m=4.2, c_drag=6e-4,
                       measurement_uncertainty=10, process_uncertainty=0.5)
shell.sensors        # [KalmanSensor('radar', ..., channels=3)]: X, Y, Z in m
```

The first argument of a model is its sensor. A name, as here, names the sensor
the model is built with (its own `h` and `R`). A `KalmanSensor` in its place
*is* the measurement, and a list gives several; a name in the list keeps the
model's own sensor beside the others:

```python
gps   = KalmanSensor("gps", G=[[1.0, 0.0]], R=0.25, labels={"X": "m"})
wheel = KalmanSensor("wheel", lambda x: [x[1, 0]], R=[0.04], labels={"V": "m/s"})
car   = KalmanModel([gps, wheel], fs=10, F=F, Q=Q, x0=x0, P0=P0)     # no H, no R: the sensors are them
both  = BallisticModel(["radar", doppler], order=3, fs=32, ...)      # its own sensor, and one more
```

`model.add_sensor(sensor)` (also `add_sensor(model, sensor)`) gives a **new
model**, read by one more sensor on the same state; the model it was called on
is left as it is. One filter update then weighs every sensor by its noise,
which is sensor fusion:

```
z = [ h(x, t) ; g(x, t) ]          R = diag(R_model, R_sensor)
```

```python
from kalman.framework import KalmanSensor

def radial_velocity(x):                      # the velocity along the line of sight
    x = np.asarray(x).reshape(-1)
    p, v = x[[0, 3, 6]], x[[1, 4, 7]]
    return [float(v @ p / np.linalg.norm(p))]

doppler = KalmanSensor("doppler", radial_velocity, R=1.0, labels={"V_RADIAL": "m/s"})
radar = shell.add_sensor(doppler)
radar.sensors        # [radar, doppler]
radar.observables    # X, Y, Z, V_RADIAL: derived from the sensors
record = radar.filter(zk)                    # zk has four columns
```

| `KalmanSensor(name, ...)` | |
|---|---|
| `g` | a function `g(x)` or `g(x, t=...)` returning its `k` readings |
| `G=` | instead of `g`, a `k x n` matrix for readings linear in the state; a model of matrices stays one, on the linear engine |
| `R=` | its noise: a `k x k` covariance, `k` variances, or one variance for all |
| `labels=` | its observables: `{"X": "m", ...}`, a list of names, or a `KalmanObservable` enum; named after the sensor by default |
| `jacobian=` | `dg/dx`; differenced with the model's scheme when not given |
| `rate=` | how often it reports; by default at every step of the model |

A sensor with a `rate` reports less often than the model steps. Between two of
its reports its channels are missing (NaN), which every engine skips, so
sensors of different rates need no special filter. `model.measurements` lays
each sensor's own readings on the model's steps:

```python
gps   = KalmanSensor("gps", G=[[1.0, 0.0]], R=0.25, rate=2)       # twice a second
wheel = KalmanSensor("wheel", G=[[0.0, 1.0]], R=0.04)             # at every step
car   = KalmanModel([gps, wheel], fs=10, F=F, Q=Q, x0=x0, P0=P0)
zk    = car.measurements({"gps": z_gps, "wheel": z_wheel})        # (N, 2), NaN where the GPS is silent
record = car.filter(zk)
```

`model.sample()` leaves the same gaps. The fusion application's `SensorSpec` is
such a sensor.

Three kinds of sensor on the same shell:

```python
P = np.zeros((3, 9)); P[0, 0] = P[1, 3] = P[2, 6] = 1.0     # picks x, y, z out of the state

# a second position instrument: redundant, fused by its noise
lidar = KalmanSensor("lidar", G=P, R=2.0**2, labels={"X": "m", "Y": "m", "Z": "m"})

# one that is not aligned with the axes: its noise is given in its own axes
tilted = KalmanSensor("tilted", G=Rot @ P, R=np.diag([1.0, 1.0, 25.0]))

# a camera: two pixel coordinates, by a projection that divides by the depth
def pixels(x):
    p = Rot @ (P @ np.asarray(x).reshape(-1) - camera_position)
    return [focal * p[0] / p[2], focal * p[1] / p[2]]
camera = KalmanSensor("camera", pixels, R=0.5**2, labels={"U": "px", "V": "px"})

tracked = shell.add_sensor(lidar).add_sensor(camera)        # they nest: one flat list of sensors
tracked.observables   # X, Y, Z, lidar_X, lidar_Y, lidar_Z, U, V: a name already taken gets its sensor's in front
```

One camera gives the direction of a point and not how far it is; two at
different places give both, with no special code: the filter triangulates
through the two sensors' Jacobians (`tests/test_add_sensor.py`, `TestCameras`:
the depth of a point 40 m away is left with metres of doubt by one camera and
found to 5 cm by two).

### Generic sensors

Three readings come ready-made in the framework, for any model that has the
components they read. Each knows a geometry and a noise, and nothing of any
product; it takes the number of state components and the indices it reads:

```python
from kalman.framework import PositionSensor, DopplerSensor, CameraSensor

# a shell with state (px, vx, ax, py, vy, ay, pz, vz, az)
P, V = [0, 3, 6], [1, 4, 7]
gps     = PositionSensor("gps", 9, P, sigma=10.0)                    # z = x[axes], or rotation @ x[axes]
doppler = DopplerSensor("doppler", 9, P, V, sigma=1.0, at=[0, 0, 0]) # the velocity along the line of sight
camera  = CameraSensor("camera", 9, P, focal=800.0, sigma=1.5,       # two pixel coordinates
                       at=[0, 0, -50], rotation=None)

seen = shell.add_sensor(gps).add_sensor(doppler).add_sensor(camera)
```

| | Reads | Channels | Settings |
|---|---|---|---|
| `PositionSensor(name, n, axes, sigma, rotation=None)` | the chosen components, in its own axes when rotated; a matrix, so a model of matrices stays one | one per axis | none |
| `DopplerSensor(name, n, position, velocity, sigma, at=None)` | `v · (p − at) / |p − at|`, in 1, 2 or 3 dimensions | 1 | `x`, `y`, `z`: where it is |
| `CameraSensor(name, n, position, focal, sigma, at=None, rotation=None)` | `focal · q_x / q_z`, `focal · q_y / q_z` with `q = rotation · (p − at)` | 2 | `focal`, `x`, `y`, `z` |

A sensor's **settings** are the numbers its readings depend on. The model that
takes the sensor has each as a parameter, `"<name>.<setting>"`, so `joint`
estimates it with the state. A camera whose focal length is known to 100 px:

```python
model = target.add_sensor(CameraSensor("camera", 6, P, focal=700.0, sigma=1.5, at=[0, 0, -50]))
model.parameters()                                   # camera.focal, camera.x, camera.y, camera.z
record = model.joint({"camera.focal": (100.0, 0.0)}).filter(zk)
record.xk[-1, 6]                                     # 800.2 +/- 0.3 px on data taken at 800 px
```

Each is one C++ class, `Kalman::Sensors::Position`, `Doppler` and `Camera`
(`include/Kalman/Sensors/`), with its Jacobian written out. The Python class
holds the C++ sensor and computes nothing: `sensor.g(x)`, its Jacobian, its
noise and its settings are asked of it, and a model takes it as it is, so
nothing is called back into Python during a pass.

### Adapters: one piece of hardware

The generic sensors know no product. A given device is an **adapter**: an
instrument made of generic sensors, with what its data sheet and its mounting
say (noise, rate, delay, where it is, its calibration) and where its data
comes from. Adapters live in `kalman.adapters`, beside `kalman.apps`: the
applications are specific models, the adapters specific hardware, and the
framework knows neither. None is shipped yet; one looks like this:

```python
from kalman.framework import KalmanInstrument, DopplerSensor, PositionSensor

class BenchRadar(KalmanInstrument):
    """The radar of one test bench: its data sheet and its mounting, in one place."""

    def __init__(self, n, position, velocity, provider=None):
        super().__init__("bench-radar", [
            PositionSensor("fix", n, position, sigma=0.5),               # m, from the data sheet
            DopplerSensor("doppler", n, position, velocity, sigma=0.15,  # m/s
                          at=[12.0, -3.5, 1.8]),                         # where it is mounted [m]
        ], rate=10.0, provider=provider)                                 # Hz; where its data comes from
        self.calibrate(gain=1.0, offset=0.0)                             # then estimated with joint

tracked = model.add_instrument(BenchRadar(6, [0, 2, 4], [1, 3, 5], provider=stream))
zk = tracked.measurements()                # fetched from the radar, laid on the model's steps
tracked.joint({"bench-radar.offset": (1.0, 0.0)}).filter(zk)
```

`provider=` is a provider ([Providers](7_Providers.md)) or any function that
returns the instrument's own readings, one column per channel, at the
instrument's rate. `instrument.read()` splits them by sensor, and
`model.measurements()` with no argument fetches every instrument that has one;
the sensors without a provider are missing (NaN), which the engines skip. In
C++ an adapter is a class deriving from `Kalman::Instrument`.

### Instruments

A detector is an instrument too, of another kind: its sensors are its layers,
read one after the other (`sequential`), each at its `station`.
`chamber.instruments` gives it for a model track finding runs on; it is not
added to another model.

An **instrument** is a set of sensors on one piece of hardware: a weather
station, a receiver, a camera. Sensors are composed into one, and the model
takes it whole:

```python
from kalman.framework import compose

station = compose(thermometer, co2, name="station", rate=0.5)    # or: thermometer | co2
room    = model.add_instrument(station)       # a new model, read by both
room.instruments                              # [model's own sensor, station]
```

Its sensors share the instrument's name, its `rate` (unless a sensor states
its own), and its **calibration** when it is given one: every reading then
goes through one gain and one offset,

```
z = gain * g(x, t) + offset
```

which the model declares as two parameters, `"station.gain"` and
`"station.offset"`. Being parameters, `joint` can estimate them along with the
state, which is calibrating the instrument while it measures:

```python
thermo  = compose(probe, name="thermo").calibrate(gain=1.0, offset=0.0)
alone   = room.add_instrument(thermo).joint({"thermo.offset": (2.0, 0.0)})
alone.observability(50)["weakness"]     # temperature and thermo.offset share a direction: not determined

checked = room.add_instrument(thermo).add_sensor(reference).joint({"thermo.offset": (2.0, 0.0)})
checked.observability(50)["observable"] # True: a calibrated thermometer settles it
record  = checked.filter(zk)            # the last state component is the offset
```

One thermometer cannot tell its own offset from the temperature, and
`observability` says so before any data; a second one, already calibrated,
makes the offset a quantity the filter finds (`examples/python/instrument.py`:
1.5 found to 0.01 from 400 readings). A calibrated reading is not a matrix
`H x`, so the model runs on the extended engine.

### A sensor that reports late, and the lagged model

Some sensors answer late: a camera that takes 0.3 s to process a frame reports
where the object *was*. Give the sensor its `delay`, then make the model carry
its past states with `lag`:

```python
camera = KalmanSensor("camera", G=[[1.0, 0.0]], R=0.01, delay=0.3)
model  = cart.add_sensor(camera).lag(3)       # 3 steps of 0.1 s: the sensors first, then the lag
record = model.filter(zk)
```

`model.lag(L)` (also `lag(model, L)`) is a new model whose state is the inner
one `L + 1` times over, `[x_k; x_{k-1}; ...; x_{k-L}]`: each step computes the
new present state and shifts the others down. The late sensor's rows of `h`
read the block of its delay, so the update corrects that past state and,
through the covariance between the blocks, the present one. No filter changes;
a model of matrices stays one, on the linear engine. On a cart with a noisy
position sensor and a precise one 3 steps late (`tests/test_lag.py`), the
position error is 0.11 with the lag, 0.21 with the late sensor left out, and
1.1 with its delay ignored. Without `lag`, a filtering pass refuses a late
sensor rather than read it at the wrong time.

The same model gives **fixed-lag smoothing** as the filter runs:
`record.xk[:, model.block(j)]` is the state `j` steps before each row, given
everything measured up to that row. Its components are named after the inner
ones with `@-j`.

The sensor a model is built with reads its own `h` and `R` and carries the
name given as first argument; a model names its channels in its constructor
with `self._name_channels(["X", "Y", "Z"])`. `add_sensor` takes a
`KalmanSensor` only. `KalmanObservable` stays the type of a channel label; a model's
`observables` are those of its sensors, in the order of `z`.

The operations nest: `shell.add_sensor(doppler).joint({"c_drag": ...})`
estimates a parameter of the shell with both sensors. The model is
`Kalman::Models::Fusion`; the dynamics run on the model's C++ counterpart when it has
one, and `g` is called back from the C++ loop. Notebook 01, Step 6 adds a Doppler radar to a tracked shell.

---

## The shipped catalog

46 model classes are discoverable through `kalman.api.catalog()["models"]`.

| Family / domain | Models |
|---|---|
| **framework** | `KalmanModel`, `LinearKalmanModel`, `ExtendedKalmanModel`, `UnscentedKalmanModel`, `Composite`, `Joint`, `Fusion`, `Lag` |
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

## Writing a model to a file

`model.export("name.kw")` writes any model to one file and
`kalman.native.open_model("name.kw")` reads it back; what the file can hold
follows `model.structure` (matrices, matrices that change with time, or code).
See [Model files](15_Cxx-Library.md).

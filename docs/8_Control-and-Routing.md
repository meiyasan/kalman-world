# Control and Routing

Two ways to act on the estimation loop rather than just observe it:

- a **controller** computes `u` from the current estimate and feeds it back into
  the plant;
- a **router** chooses *which model* is currently right, for systems that switch
  regime.

---

## Controllers

### The contract

A controller is a **callable**: `u = ctrl(x̂, P, t, nu=ν)`. The filter receives
it through `uk=` and treats it exactly like a control sequence.

```python
from kalman.framework import KalmanFilter, LQRKalmanController

ctrl = LQRKalmanController(model, Qcost=np.diag([10., 1.]), Rcost=np.array([[0.1]]))
ctrl.fit()                                   # LQR/neural controllers train first

kf = KalmanFilter(model)
truth = model.sample(5.0)
record = kf.forward(truth.zk, uk=ctrl)       # closed loop

total = sum(c for _, c in ctrl.cost_log)
```

Every call logs `(t, cost)`, `(t, u)` and, if the controller adapts the input
matrix, `(t, B)`. `reset()` clears all three.

Subclasses override `_policy` (required) and `_update` (optional — returning a
new `B` rebinds the model's input matrix as a side effect).

### Cost accounting

Cost is charged against the command the actuator can **actually deliver**
(`model.limit_control(u)`), not the controller's raw ask. A fixed-gain controller with
no notion of an actuator limit can request far more effort than the plant ever
receives; charging `Rcost` against that fictitious effort would unfairly inflate
its cost relative to a controller whose raw output already respects the limit.
`u_log` still keeps the raw command for diagnostics.

### Shipped controllers

| Class | Key parameters | Character |
|---|---|---|
| `ProportionalKalmanController` | `Kp` | simplest possible feedback |
| `PIDKalmanController` | `Kp`, `Ki`, `Kd` | classical three-term |
| `LQRKalmanController` | `Qcost`, `Rcost` (+ `fit()`) | optimal linear-quadratic gain |
| `MPCKalmanController` | `horizon`, `u_min`, `u_max`, `du_max`, `n_iterations` | model predictive control: a plan under limits, remade at every step |
| `MPPIKalmanController` | `n_rollouts`, `horizon`, `noise_sigma`, `lam` | sampling-based MPC, honours actuator limits natively |
| `NeuralKalmanController` | `hidden`, `features` (+ `fit()` / `fit_mppi()`) | learned policy, can distil MPPI |

### Model predictive control

`MPCKalmanController` plans `horizon` commands ahead at every step, applies the
first, and plans again. Limits on the command are constraints of the plan, not
a clip applied after it:

```python
from kalman.framework import MPCKalmanController

model = CarLaneModel("camera", fs=50, x0=x0)
Qcost, Rcost, x_ref = model.control_cost()
mpc = MPCKalmanController(model, horizon=40,
                          u_min=-model.steer_limit, u_max=model.steer_limit,   # steering within 0.1 rad
                          du_max=0.01,                                         # and at most 0.01 rad per step
                          Qcost=Qcost, Rcost=Rcost, x_ref=x_ref)
u = mpc(x_hat, P, t)          # the first command of the plan
mpc.plan()                    # the whole plan, (horizon, l)
```

| | |
|---|---|
| Cost | the same `Qcost`, `Rcost`, `x_ref` as every controller, summed over the horizon, plus a terminal cost: the Riccati solution, so that with no limit in the way the command is the LQR command whatever the horizon |
| Limits | `u_min`, `u_max` on the command and `du_max` on its change per step (a number, or one per command) |
| Prediction | the model's own `f`: exact for a model of matrices, linearised along the previous plan otherwise, `n_iterations` passes per step |
| Solver | Hildreth's method on the dual of the quadratic programme: no dependency, the same answer for the same input |

Beside the others: LQR is one gain and knows no limit; MPPI replans by rolling
random command sequences through `f`; MPC replans deterministically. On the
lane-keeping car started 3 m off the lane centre (250 steps, no noise):

| Controller | Cost | Largest steering asked | Per step |
|---|---|---|---|
| LQR, clipped by the actuator | 257.2 | 0.288 rad | 0.07 ms |
| MPC, horizon 40 | 257.2 | 0.100 rad | 9 ms |
| MPPI, 100 rollouts, horizon 15 | 454.6 | 0.093 rad | 141 ms |

For this car, LQR clipped after the fact happens to be as good as planning with
the limit: the two costs agree. MPC never asks for more than the actuator
gives, and it is the one that takes a rate limit. Where clipping does cost,
MPC is ahead: a mass driven by a limited force from 6 m away costs 404.5 under
MPC against 412.2 under clipped LQR (`tests/cxx/13_controllers.cc`).

Notebook 12 (section 5b) runs it on the same car with cheaper steering, where
LQR asks for ten times the limit: MPC then has the lowest cost of the three
(327.2, against 331.6 for LQR and 341.7 for MPPI) and the smallest overshoot.

The planner is `Kalman::Controllers::MPC`; the Python class forwards to it and
has no implementation of its own, so it needs libKalman.

### Tuners

A tuner is a different role: it acts on the *filter*, not on the system. Once
per step, with the innovation in hand, it retunes the model's `Q` or `R`, and
it commands nothing. A filter takes a controller and a tuner separately:

```python
filter.forward(zk, uk=lqr, tuner=adaptive)     # a command from lqr, the noise retuned by adaptive
filter.forward(zk, tuner=adaptive)             # no command, the noise retuned
```

| Class | Key parameters | Character |
|---|---|---|
| `BoostKalmanTuner` | `boost_factor`, `threshold`, `window` | transient response boost on innovation spikes |
| `AdaptiveQRKalmanTuner` | `window`, `q_min`, `q_max` | retunes `Q`/`R` online from innovation statistics |
| `NeuralAdaptiveQRKalmanTuner` | `window`, `hidden`, `q_min`, `q_max` | learned `Q`/`R` adaptation across regimes |

They derive from `KalmanTuner` (`Kalman::Tuner` in C++, with
`Tuners::Boost` and `Tuners::AdaptiveQR`), which is a controller whose
command is fixed at zero: what a tuner writes is its update.

All take `model` first and accept `x_ref`, `Qcost`, `Rcost`. Defaults come from
`model.control_cost()` when the model defines it, else identity matrices.

### Through the API

```python
from kalman.api import run_control

run_control(model_spec,
            {"controller": "lqr",
             "cost": {"Qcost": [[10, 0], [0, 1]], "Rcost": [[0.1]], "x_ref": [[0], [0]]},
             "params": {}},
            duration=5.0)
# {"run_id": …, "xk": [...], "cost": 2026.5, "cost_log": [...]}
```

The controller name is matched by prefix, so `"lqr"`, `"LQR"` and
`"LQRKalmanController"` all resolve. `fit()` is called automatically when the
controller defines it.

---

## Routers (IKF)

A router decides how much weight each model in a bank deserves right now.
`InteractiveKalmanFilter` runs M filters in parallel — one per regime — and each
step: mixes their states, runs every model's Kalman step, logs each model's NIS,
asks the router for updated probabilities, and recombines
`x̂ = Σ μⱼ x̂ⱼ`.

```python
from kalman.framework.filters import InteractiveKalmanFilter
from kalman.framework.routers import BayesianRouter, ThresholdRouter, NeuralRouter, HybridRouter
from kalman.framework.routers.nn import MLP

ikf = InteractiveKalmanFilter(models, Pi)                              # Bayesian default
ikf = InteractiveKalmanFilter(models, Pi, router=ThresholdRouter())
ikf = InteractiveKalmanFilter(models, Pi, router=NeuralRouter(model=MLP(hidden=32)))
record = ikf.forward(zk)
```

`Pi` is the Markov transition matrix between regimes.

| Router | Decides from | Needs training |
|---|---|---|
| `BayesianRouter` | model likelihoods (the textbook IKF update) | no |
| `ThresholdRouter` | NIS crossing a threshold | no |
| `NeuralRouter` | a learned function of the NIS/likelihood features | yes |
| `HybridRouter` | Bayesian prior blended with a learned correction | yes |

Learning routers are trained on labelled regime sequences:

```python
ikf.fit(training_zk_list, labels_list)     # collects NIS features, trains the router
record = ikf.forward(test_zk)
```

Swapping the router never touches the filter — that is the point of the split.
Network architectures are pluggable through `kalman.framework.routers.nn`.

---

## Worked examples

- [`12_KalmanAI.ipynb`](../share/notebooks/12_KalmanAI.ipynb) — the control and
  routing layer end to end: P/PID/LQR/MPPI/neural comparison, adaptive `Q`/`R`,
  IKF routing.
- [`14_StormTracking.ipynb`](../share/notebooks/14_StormTracking.ipynb) — IKF over
  storm recurvature regimes.

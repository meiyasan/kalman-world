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
(`model.clip_u(u)`), not the controller's raw ask. A fixed-gain controller with
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
| `MPPIKalmanController` | `n_rollouts`, `horizon`, `noise_sigma`, `lam` | sampling-based MPC, honours actuator limits natively |
| `NeuralKalmanController` | `hidden`, `features` (+ `fit()` / `fit_mppi()`) | learned policy, can distil MPPI |
| `BoostKalmanController` | `boost_factor`, `threshold`, `window` | transient response boost on innovation spikes |
| `AdaptiveQRKalmanController` | `window`, `q_min`, `q_max` | retunes `Q`/`R` online from innovation statistics |
| `NeuralAdaptiveQRKalmanController` | `window`, `hidden`, `q_min`, `q_max` | learned `Q`/`R` adaptation across regimes |

The last two are *not* control in the actuator sense — they act on the filter's
own noise model. That is deliberate: the same callable hook is the natural place
to adapt tuning online.

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

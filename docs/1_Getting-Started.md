# Getting Started

Ten minutes, four steps: install, define a system, filter it, check that the
filter is telling the truth.

If you have not installed anything yet, `make setup` from the repository root
does the whole job — see [Installation](2_Installation.md).

---

## 1. A filter with no model class

The fastest entry point needs no subclass at all. Give `KalmanFilter` the
dimensions and the matrices, and it builds a generic linear model for you.

```python
import numpy as np
from kalman.framework import KalmanFilter

dt = 0.01
kf = KalmanFilter(
    n=2, m=1,                      # 2 states (position, velocity), 1 measurement
    F=[[1, dt], [0, 1]],           # constant-velocity dynamics
    H=[[1, 0]],                    # we only observe position
    Q=np.diag([1e-6, 1e-4]),       # process noise
    R=[[0.05]],                    # measurement noise
    fs=1 / dt,
)
```

Every coefficient stays editable afterwards — `kf.F`, `kf.H`, `kf.Q`, `kf.R`,
`kf.x0`, `kf.P0` are read/write properties that route into the underlying model.

## 2. Get some data

With no instrument at hand, sample the model itself. `sample()` returns a
`KalmanRecord` carrying the truth (`xk`), the noisy measurements (`zk`) and the
time base.

```python
truth = kf.sample(3.0)             # 3 seconds
truth.time.shape, truth.xk.shape, truth.zk.shape
# ((300,), (300, 2), (300, 1))
```

For real data, see [Providers](7_Providers.md) — a provider returns the same
shapes, so nothing downstream changes.

## 3. Run the filter

```python
record = kf.forward(truth.zk)      # forward pass (filtering)
record.xk.shape, record.Pk.shape   # ((300, 2), (300, 2, 2))

smoothed = kf.backward()           # RTS smoother over the stored forward pass
```

`kf.filter(zk, smooth=True)` does both in one call. The returned
`KalmanRecord` is the universal currency: `time`, `xk` (state estimates),
`Pk` (covariances), `zk` (measurements), plus `len()`, indexing and iteration.

## 4. Look at it, and check it

```python
fig = kf.model.plot(record, sample=truth)   # estimate vs truth, with ±σ bands
fig = kf.diagnostics(sample=truth)          # innovation, gain, NIS, NEES
```

The diagnostics dashboard is the part people skip and then regret. A
well-tuned filter has **NIS ≈ m** (the measurement dimension) on average; the
example above sits at ~0.99 for `m = 1`. Much larger means `R` or `Q` is too
small (the filter is overconfident); much smaller means they are too large (it
is ignoring the data). See
[Filters and Diagnostics](5_Filters-and-Diagnostics.md).

---

## The same thing with a named model

`KalmanModel` is the real contract. Constructing it directly gives you a name,
a sampling rate and somewhere to hang domain logic later.

```python
import numpy as np
from kalman.framework import KalmanModel
from kalman.filters import LinearKalmanFilter

dt = 0.01
model = KalmanModel(
    "demo",
    fs=1 / dt,
    F=np.array([[1, dt], [0, 1]]),
    H=np.array([[1.0, 0.0]]),
    Q=np.diag([1e-6, 1e-4]),
    R=np.array([[0.05]]),
    x0=np.array([[0.0], [1.0]]),   # start at the origin, moving at 1 m/s
)

kf = LinearKalmanFilter(model)     # or ExtendedKalmanFilter / UnscentedKalmanFilter
record = kf.forward(model.sample(5.0).zk)
```

## Using a domain model instead

46 models ship with the framework. Nothing above changes — swap the model.

```python
from kalman.apps import BallisticModel
from kalman.filters import ExtendedKalmanFilter

model = BallisticModel("shot", fs=50)     # 9 states, 3 observables, nonlinear
kf = ExtendedKalmanFilter(model)
truth = model.sample(10.0)
record = kf.forward(truth.zk)
```

List what is available without leaving Python:

```python
from kalman.api import catalog, describe
catalog()["models"]           # 43 names
describe("BallisticModel")    # params, version, engine/integrator/solver bindings
```

---

## Where to go next

- **I want to model my own system** → [Models](4_Models.md)
- **Which filter should I use?** → [Filters and Diagnostics](5_Filters-and-Diagnostics.md)
- **My dynamics are an ODE/PDE** → [Integrators and Solvers](6_Integrators-and-Solvers.md)
- **I want to drive this from another program** → [Python API](9_Python-API.md),
  then [REST API](10_REST-API.md) or [MCP Server](11_MCP-Server.md)
- **Show me worked physics** → [`share/notebooks/00_Kalman101.ipynb`](../share/notebooks/00_Kalman101.ipynb)
  and its 15 siblings.

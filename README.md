# kalman-framework

A modular Python framework for Kalman-family filters, supporting linear, extended (EKF),
and unscented (UKF) variants, with composable multi-sensor models.

---

## Notation

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `gk` | `(T, M, 1)` | Raw input timeseries (before any conditioning) |
| `uk` | `(T, L, 1)` | Control input timeseries |
| `zk` | `(T, M, 1)` | Conditioned measurements (after `preprocess`) |
| `xk` | `(n, 1)` | State vector at a single timestep |
| `xk_hat` | `(T, n, 1)` | Batch of state estimates (prior / posterior / smoothed) |
| `zk_hat` | `(T, M, 1)` | Predicted measurements in filter space — output of `h(xk_hat)` |
| `gk_hat` | `(T, M, 1)` | Predicted measurements in raw space — output of `postprocess(zk_hat)` |
| `n` | — | State dimension |
| `m` | — | Measurement dimension |
| `l` | — | Control input dimension |

---

## Filter pipeline

```
gk          raw input timeseries            (T, M, 1)
 │
 ▼  preprocess(gk, uk) ──────────────────── gk → filter space
 │
zk          conditioned measurements         (T, M, 1)
 │
 ▼  ── filter loop: for z, u in zip(zk, uk) ─────────────────────────────────
 │      predict:  x⁻, P⁻  ←  f(x, u),  F, Q
 │      update:   x⁺, P⁺  ←  x⁻ + K·(z − h(x⁻)),  H, R
 │                                    ↑
 │                    h is mode-aware (INDEPENDENT stacks, FUSED sums)
 ▼
xk_hat      state estimates                  (T, n, 1)   prior / posterior / smoothed
 │
 ▼  inprocess(xk_hat) ──────────────────── optional state-space transform
 │
 ▼  h(xk_hat)   ← mode-aware ────────────── state → filter space
 │
zk_hat      predicted measurements           (T, M, 1)   filter space
 │
 ▼  postprocess(zk_hat) ─────────────────── filter space → raw space
 │
gk_hat      predicted raw output             (T, M, 1)   mirrors gk
```

`preprocess` and `postprocess` are intended as a matched pair: if `preprocess` whitens
or conditions the signal, `postprocess` should map predictions back to the original domain
so that `gk_hat` is directly comparable to `gk`.

---

## Model hierarchy

```
KalmanModelABC          (abstract contract: f, h, F, H, B, L, M, Q, R, pre/in/postprocess)
    └── KalmanModel         (numerical Jacobian defaults, covariance inference)
            ├── KalmanLinearModel     (constant or time-varying F, H matrices)
            ├── KalmanExtendedModel   (nonlinear f, h with analytic Jacobians)
            ├── KalmanUnscentedModel  (nonlinear f, h for UKF)
            └── KalmanCompositeModel  (composes N submodels — see below)
```

### Filter engines

| Class | Algorithm |
|-------|-----------|
| `KalmanFilter` | EKF (works for linear and nonlinear models) |
| `LinearKalmanFilter` | Standard linear KF |
| `MultiRateKalmanFilter` | Multi-rate / asynchronous measurements |
| `UnscentedKalmanFilter` | UKF |

---

## KalmanCompositeModel

Combines N submodels into a single model. State evolution is always block-diagonal
(submodels evolve independently). Observation composition depends on `obs_mode`.

### Observation modes (`KalmanObservableReadout`)

| Mode | Measurement vector | Use case |
|------|--------------------|----------|
| `INDEPENDENT` | `z = [h₁(x₁); h₂(x₂); ...]` — stacked | separate sensors, each with its own measurement |
| `FUSED` | `z = Σ hᵢ(xᵢ)` — summed | all submodels contribute additively to the same observable |

### Mode-awareness

Only operators that touch the measurement vector `z` need to be mode-aware.
`_bind_h` and `_bind_H` implement the dispatch. All other operators follow fixed rules:

| Operator | Rule | Why |
|----------|------|-----|
| `f`, `F`, `Q` | block-diagonal | state evolution is always independent |
| `h`, `H` | **mode-aware** | measurement composition depends on `obs_mode` |
| `preprocess` | INDEPENDENT: split by `m.m`, delegate per submodel; FUSED: pass-through | runs before the filter loop on raw `gk`; must align `z` with `h(x)` |
| `inprocess` | identity (pass-through) | `_bind_h` composes submodel contributions; no splitting needed |
| `postprocess` | identity (pass-through) | `_bind_h` already ran; output is in the correct space |

### Subclassing for domain-specific processing

Override `_bind_preprocess`, `_bind_inprocess`, or `_bind_postprocess` in a subclass
to inject domain logic while keeping the composite wiring intact.

```python
class ThermalNoiseComposite(KalmanCompositeModel):

    def _bind_preprocess(self, mode):
        independent_default = super()._bind_preprocess(mode)

        def preprocess(gk, uk=None, **kwargs):
            # apply notch / whitening filter before delegating
            gk = self._whiten(gk)
            return independent_default(gk, uk, **kwargs)

        return preprocess

    def _bind_postprocess(self):
        def postprocess(zk_hat):
            # colour the predictions back to the raw domain
            return self._colour(zk_hat)
        return postprocess
```

---

## Quick start

```python
from modules.models.linear    import KalmanLinearModel
from modules.models.composite import KalmanCompositeModel
from modules.framework.constants import KalmanObservableReadout
from modules.filters.kf        import LinearKalmanFilter
import numpy as np

# --- two independent 1-D constant-velocity submodels ---
F = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])
Q = np.diag([1e-4, 1e-3])
R = np.array([[0.1]])

m1 = KalmanLinearModel("sensor_a", F=F, H=H, Q=Q, R=R, x0=np.zeros((2,1)))
m2 = KalmanLinearModel("sensor_b", F=F, H=H, Q=Q, R=R, x0=np.zeros((2,1)))

# stacked: z = [h1(x1); h2(x2)]
composite = KalmanCompositeModel(m1, m2, obs_mode=KalmanObservableReadout.INDEPENDENT)

kf = LinearKalmanFilter(composite)
kf.forward(gk, record=True)
record = kf.get_record()   # contains gk_hat, xk_hat, Pk_hat
```

---

## Processing hooks contract

```
preprocess(gk, uk)  → (zk, uk)      before the filter loop   shape: (T, M, 1)
inprocess(xk_hat)   → xk_hat        before h is applied       shape: (T, n, 1)
postprocess(zk_hat) → gk_hat        after h is applied        shape: (T, M, 1)
```

All three default to identity in `KalmanCompositeModel`. Override them in subclasses
for signal conditioning (`preprocess`), coordinate transforms (`inprocess`), or
domain reconstruction (`postprocess`).

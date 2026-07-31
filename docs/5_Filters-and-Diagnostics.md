# Filters and Diagnostics

A filter owns the **workflow** (WHEN): the forward pass, the backward smoother,
and the run state. It implements no covariance math — that lives in the engine
it composes. So every filter below is a thin leaf over the same
`KalmanFilter`, differing only in which engine it binds.

```python
from kalman.filters import LinearKalmanFilter
kf = LinearKalmanFilter(model)     # engine chosen by the leaf's KIND
```

If you construct `KalmanFilter(model)` directly, the engine comes from the
model's `DEFAULT_ENGINE`. Either way the chosen engine is validated against
`model.ALLOWED_ENGINES`, and a mismatch raises `TypeError` at construction —
not silently at step 4000.

---

## Choosing a filter

| Class | Engine kind | Use when | Smoother |
|---|---|---|---|
| `LinearKalmanFilter` | `linear` | `F`/`H` are constant matrices | yes |
| `ExtendedKalmanFilter` | `extended` | nonlinear `f`/`h`, mild curvature, Jacobians exist | yes |
| `UnscentedKalmanFilter` | `unscented` | strong nonlinearity, or non-additive noise `f(…,w)` | yes |
| `ParticleFilter` | `particle` | multi-modal or non-Gaussian posteriors | no |
| `EnsembleKalmanFilter` | `ensemble` | very high dimension (fields, grids) | no |
| `CombinatorialKalmanFilter` | `combinatorial` | many hits per layer, tracks to *find* not just follow | yes |
| `InteractiveKalmanFilter` | (routes M filters) | regime switching between several models | per-model |

Rules of thumb: start linear, move to EKF only when the dynamics are actually
nonlinear, and to UKF when the EKF's linearisation visibly biases the estimate
(NIS drifts, innovations go non-white). Reach for PF only when the posterior is
genuinely multi-modal — it costs far more and has no smoother. Above ~10⁴
states, dense covariance is impossible and EnKF is the only practical option.

### The linear fast path

The linear engine and the native C acceleration use `F()` matrices and never
call `f()`. Two consequences worth internalising:

- Changes to `f()` do not affect them — and must not break them.
- Setting `integrator=` on a purely linear model changes nothing. See
  [Integrators and Solvers](6_Integrators-and-Solvers.md).

---

## Running one

```python
record = kf.forward(zk)                 # forward pass, records by default
smoothed = kf.backward()                # RTS smoother over the stored pass
record = kf.filter(zk, smooth=True)     # both, in one call
truth = kf.sample(10.0)                 # synthetic truth + measurements
```

`forward(gk, uk=None, *, record=True, **kwargs)` accepts a control sequence
`uk` — or a **callable controller**, see
[Control and Routing](8_Control-and-Routing.md). `record=False` skips
building the `KalmanRecord` when you only want the run state.

### The record

Every run returns a frozen `KalmanRecord`:

| Field | Shape | Meaning |
|---|---|---|
| `time` | `(T,)` | time base |
| `xk` | `(T, n)` | state estimates |
| `Pk` | `(T, n, n)` | state covariances |
| `zk` | `(T, m)` | measurements (or reconstructed output after `postprocess`) |

Plus `len()`, `record[k]`, iteration over `(t, x, P, z)`, and the derived
`n`, `m`, `duration`, `sample_rate`. It validates its own shapes and rejects
NaN/Inf on construction, so a corrupted run fails at the boundary rather than
three plots later.

```python
fig = model.plot(record, sample=truth, sigma=2)   # estimate, truth, ±2σ band
```

---

## The specialists

### EnKF — high-dimensional assimilation

```python
from kalman.filters import EnsembleKalmanFilter

enkf = EnsembleKalmanFilter(model, n_members=32, inflation=1.05, seed=0)
record = enkf.forward(zk)
```

The covariance is carried by the ensemble and never formed densely. If the model
defines `localization()`, it supplies the Gaspari–Cohn tapers; if it defines
`initial_ensemble(N, rng)`, members are seeded from climatology instead of a
Gaussian `x0/P0` draw. `backward()` raises — there is no RTS smoother.

For a twin-experiment (OSSE) harness comparing EnKF against a free run, see
`kalman.apps.weather.TwinExperiment` and `kalman.api.run_osse()`.

### CKF — track finding

Where the other filters estimate one trajectory from a series of single
measurements, the CKF reconstructs **several** tracks from an event where every
detector layer holds many hits. Per layer it opens a branch per hit passing a χ²
gate (plus a "hole" branch for detector inefficiency), prunes to a bounded beam,
then arbitrates shared hits so the reported tracks are disjoint.

```python
from kalman.filters import CombinatorialKalmanFilter

ckf = CombinatorialKalmanFilter(model, chi2_cut=None, max_candidates=6, max_holes=2)
tracks = ckf.forward(hits_by_layer, layer_radii=radii)
```

`chi2_cut=None` derives the gate from the measurement dimension at p = 0.99.

### IMM — regime switching

Runs M filters in parallel, one per regime, mixing their states each step and
re-weighting by likelihood. The *routing* policy is pluggable.

```python
from kalman.filters import InteractiveKalmanFilter
from kalman.framework.routers import BayesianRouter, ThresholdRouter, NeuralRouter
from kalman.framework.routers.nn import MLP

ikf = InteractiveKalmanFilter(models, Pi)                              # Bayesian default
ikf = InteractiveKalmanFilter(models, Pi, router=ThresholdRouter())
ikf = InteractiveKalmanFilter(models, Pi, router=NeuralRouter(model=MLP(hidden=32)))

ikf.fit(training_zk_list, labels_list)     # learning routers only
record = ikf.forward(test_zk)
```

---

## Diagnostics — is the filter actually working?

Run `forward()` or `filter()` first; everything below is computed post-hoc from
the stored pass, so no engine changes are needed.

```python
nu, S = kf.innovations()    # per-step innovation and its covariance
K     = kf.gains()          # per-step Kalman gain
nis   = kf.nis()            # Normalized Innovation Squared — no truth needed
nees  = kf.nees(truth)      # Normalized Estimation Error Squared — needs truth
fig   = kf.diagnostics(sample=truth)     # the whole dashboard as one figure
```

### Reading them

| Statistic | Healthy | Too large means | Too small means |
|---|---|---|---|
| **NIS** | mean ≈ `m`, inside the χ²(m) band | filter overconfident: `R` and/or `Q` too small | filter ignoring data: `R`/`Q` too large |
| **NEES** | mean ≈ `n`, inside the χ²(n) band | covariance underestimates real error | covariance inflated |
| **innovation** | zero-mean, white | unmodelled dynamics or a bias | over-smoothing |
| **gain** | settles to a steady value | still transient — run longer, or `P0` is off | measurements being discarded |

NIS is the one to check first, because it needs no ground truth and therefore
works on real data. A structured (non-white) innovation sequence means the model
is wrong in a way no amount of `Q` tuning will fix.

### A tuning trap worth naming

When comparing several filter variants in one loop, give **each its own RNG
stream** (`seed=`) and share only the observation stream. Otherwise the variants
cross-contaminate and the comparison is meaningless. This was a real bug in this
codebase once.

---

Next: [Integrators and Solvers](6_Integrators-and-Solvers.md) for how the
prediction step is actually computed, or [Providers](7_Providers.md) for feeding
the filter real data.

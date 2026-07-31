# kalman-framework

A modular Python framework for Kalman-family estimation — linear (KF), extended
(EKF), unscented (UKF), particle (PF), ensemble (EnKF), combinatorial (CKF) and
interacting-multiple-model (IMM) — with composable multi-sensor models, 46 domain
applications, and one JSON API exposed over REST and MCP.

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

| | |
|---|---|
| [Getting Started](docs/1_Getting-Started.md) · [Installation](docs/2_Installation.md) · [Overview](docs/3_Overview.md) | first filter, setup, architecture |
| [Models](docs/4_Models.md) · [Filters and Diagnostics](docs/5_Filters-and-Diagnostics.md) · [Integrators and Solvers](docs/6_Integrators-and-Solvers.md) | the modelling surface |
| [Providers](docs/7_Providers.md) · [Control and Routing](docs/8_Control-and-Routing.md) | data in, action out |
| [Python API](docs/9_Python-API.md) · [REST API](docs/10_REST-API.md) · [MCP Server](docs/11_MCP-Server.md) · [Authentication](docs/12_Authentication.md) · [Deployment](docs/13_Deployment.md) | API access |
| [Troubleshooting](docs/14_Troubleshooting.md) | error messages, decoded |

`make setup` installs everything and self-checks; [`AGENTS.md`](AGENTS.md) has the
conventions for changing the code. The rest of this file is the reference for
notation, the filter pipeline and composite models.

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
            ├── LinearKalmanModel     (constant or time-varying F, H matrices)
            ├── ExtendedKalmanModel   (nonlinear f, h with analytic Jacobians)
            ├── UnscentedKalmanModel  (nonlinear f, h for UKF)
            └── CompositeKalmanModel  (composes N submodels — see below)
```

### Filters

Each is a thin `KalmanFilter` leaf binding the matching engine
(`kalman.filters`). See [docs/5](docs/5_Filters-and-Diagnostics.md).

| Class | Algorithm | Smoother |
|-------|-----------|----------|
| `KalmanFilter` | resolves the model's `DEFAULT_ENGINE` (EKF by default) | yes |
| `LinearKalmanFilter` | standard linear KF | yes |
| `ExtendedKalmanFilter` | EKF — Jacobian linearisation | yes |
| `UnscentedKalmanFilter` | UKF — sigma points; non-additive noise | yes |
| `ParticleFilter` | bootstrap / SIR particle filter | no |
| `EnsembleKalmanFilter` | perturbed-observation EnKF, for fields and grids | no |
| `CombinatorialKalmanFilter` | branch-and-prune track *finding* | yes |
| `InteractiveKalmanFilter` | IMM over a bank of models, pluggable routing | per-model |

---

## CompositeKalmanModel

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
class ThermalNoiseComposite(CompositeKalmanModel):

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
import numpy as np
from kalman.framework import LinearKalmanModel, CompositeKalmanModel
from kalman.framework.constants import KalmanObservableReadout
from kalman.filters import LinearKalmanFilter

# --- two independent 1-D constant-velocity submodels ---
dt = 0.01
F = np.array([[1, dt], [0, 1]])
H = np.array([[1.0, 0.0]])
Q = np.diag([1e-4, 1e-3])
R = np.array([[0.1]])

m1 = LinearKalmanModel("sensor_a", F=F, H=H, Q=Q, R=R, x0=np.zeros((2, 1)), fs=1/dt)
m2 = LinearKalmanModel("sensor_b", F=F, H=H, Q=Q, R=R, x0=np.zeros((2, 1)), fs=1/dt)

# stacked: z = [h1(x1); h2(x2)]
composite = CompositeKalmanModel(m1, m2, obs_mode=KalmanObservableReadout.INDEPENDENT)

kf = LinearKalmanFilter(composite)          # n = 4, m = 2
gk = composite.sample(2.0).zk               # or your own (T, m) measurements
kf.forward(gk, record=True)
record = kf.get_record()                    # contains gk_hat, xk_hat, Pk_hat
```

---

## Processing hooks contract

```
preprocess(gk, uk)  → (zk, uk)      before the filter loop   shape: (T, M, 1)
inprocess(xk_hat)   → xk_hat        before h is applied       shape: (T, n, 1)
postprocess(zk_hat) → gk_hat        after h is applied        shape: (T, M, 1)
```

All three default to identity in `CompositeKalmanModel`. Override them in subclasses
for signal conditioning (`preprocess`), coordinate transforms (`inprocess`), or
domain reconstruction (`postprocess`).

---

## Oscillator reconstruction: resampling artifacts and fixes

The rotating-frame oscillator models (`OscillatorModel` and friends, notebook
`02_Oscillators`) run the filter at a **decimated** rate `fs_dec = 4·dfc` and
reconstruct the band signal by upsampling the phasor estimate back to the original
rate and remodulating: `heterodyne → downsample → Kalman → upsample → remodulate`.
The up/down resampling (`scipy.signal.resample_poly`) introduced two artifacts that
have since been fixed in `OscillatorModel._resample` / `reconstruct`.

### 1. Asymmetric tail decay in the reconstructed time series

**Symptom.** The reconstructed waveform matched the truth at the start but the
**end** collapsed to ~zero (the envelope decayed away over the last fraction of a
second), while the head was fine — a left/right asymmetry.

**Cause.** `resample_poly` implicitly **zero-pads** beyond the data, so its FIR runs
into zeros at the boundaries. On *upsampling* this is asymmetric: the leading edge
recovers (the FIR step-onset settles to the right value) but the trailing ~half-filter
of output ramps linearly into the zero-pad, suppressing the tail. The Kalman filter
and smoother were **not** the cause — a bare `resample_poly` round-trip reproduced it.

**Fix.** `_resample` now **edge-replicates** before resampling: it pads both ends with
the edge sample, resamples the padded signal, then trims the padded region back out,
so the FIR always sees real data across the original span. Output length is unchanged.
Result: tail tracks truth (relative RMS error ~0.4% vs a full tail collapse).

> **Subtlety — the pad must be a whole number of `down`-blocks.** Trimming the pad back
> out by a *rounded* offset `round(pad·up/down)` leaves a fractional-sample group delay
> (~2 ms in the violin-mode setup). A delay `δt` rotates every off-carrier phasor by
> `2π·Δf·δt`, so an on-resonance (DC) envelope is unaffected but a detuned mode picks up
> a phase error that grows with `Δf` — silently capping the subtraction of detuned modes
> at ~30 dB. Padding by a multiple of `down` makes `pad·up/down` an exact integer (zero
> delay); detuned single-mode subtraction then reaches 50–60 dB, and the 3-peak FUSED
> demo goes from ~30 dB to ~40–54 dB.

### 2. "Bad suppression range" — apparent leakage outside the window

**Symptom.** In the subtraction ASD the reconstruction (the line being subtracted)
appeared to spread a broad skirt far beyond the analysis window `[fc ± dfc]`, and two
small spurious spikes sat at `fc ± fs_dec`, suggesting the filter was acting (and
subtracting) well outside its declared band.

**Cause — two separate effects:**

- **Most of the "skirt" was FFT leakage, not real content.** An unwindowed
  periodogram of a strong narrow line has `1/Δf` sinc sidelobes spanning the whole
  axis; they are *identical* for the truth and the reconstruction (same tone), so the
  reconstruction was faithful, not leaking. Applying a window (e.g. Hann) before the
  FFT collapses the skirt and reveals the genuine band-edge drop at `[fc ± 2·dfc]`
  (the decimated Nyquist). The window `[fc ± dfc]` is a **mode-selection mask**, not a
  brick-wall on the output.
- **The two spikes at `fc ± fs_dec` were real upsampling images.** Upsampling from
  `fs_dec` leaves spectral copies of the band at `±k·fs_dec`, which the anti-image
  filter only partially suppresses. `resample_poly`'s default Kaiser window (`β = 5`)
  left the first image at ≈ −73 dB.

**Fix.** Because the carried signal lives in `[fc ± dfc]` while the decimated Nyquist
is `2·dfc` (a full octave of guard band), a steeper stopband is essentially free:
`_resample` now uses a sharp **Kaiser `β = 12`** anti-alias/anti-image filter, dropping
the first image to ≈ −125 dB with no change to in-band content. For pristine output,
`reconstruct(record, band_limit=True)` additionally low-passes the upsampled phasor at
`2·dfc`, erasing the `fc ± fs_dec` images entirely (≈ −145 dB, below the FFT floor);
it is opt-in and off by default. When plotting these spectra, **window the FFT** so the
true band limit is visible rather than rectangular-window leakage.

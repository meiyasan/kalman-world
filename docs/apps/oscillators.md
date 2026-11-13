# Oscillators

A signal that contains narrow lines: a resonance ringing in a detector, a tone
in a recording. The filter follows the amplitude and phase of each line, so the
line can be measured or removed.

## The model

A line at frequency `f0` is slow to change in amplitude and phase. Multiplying
the signal by a wave at a nearby carrier frequency `fc` moves the line down to
near zero frequency, where it becomes a slowly turning arrow: its **envelope**.

```
state        x = [Re, Im] of the envelope
dynamics     the arrow turns at f0 - fc and shrinks at the damping rate
measurement  z = the demodulated signal
```

The damping rate comes from the quality factor `Q_f`: the larger it is, the
longer the line rings and the narrower it is.

| Class | What it tracks |
|---|---|
| `AnalyticOscillatorModel` | one line, from a complex signal |
| `OscillatorModel` | the lines inside one frequency window of a real signal |
| `OscillatorBankModel` | several windows of the same signal at once |

## How it is built

- **Time stepping.** The motion of the envelope is linear, so its step is exact:
  the matrix exponential of the turn and the decay over one sample.
- **A window is a mask.** `OscillatorModel` takes a list of candidate
  frequencies and keeps only those inside `fc ± dfc`. The signal is demodulated
  and its sample rate reduced before the filter sees it, and the measurement
  noise is rescaled to match.
- **Lines that share a window** are estimated together by one filter, which is
  what separates two lines closer than the window is wide.
- **A bank** sends the same signal to every window.
- **It runs in C++.** A window and a bank are composites of constant matrices:
  block-diagonal `F` and `Q`, and an `H` that sums the lines of a window
  (`[I I ...]`) or stacks the windows. The composite says so (its `structure`
  is MATRICES), so `filter()` runs the linear Kalman pass and the smoother on
  those matrices in C++ (`kf_forward_linear`, `kf_backward_rts`), and `sample()`
  runs its state chain there too. The demodulation, the decimation and the
  reconstruction (`preprocess`, `reconstruct`) are signal processing around the
  filter and stay in numpy and scipy.

## Using it

```python
from kalman.apps import OscillatorModel

window = OscillatorModel("mic", frequencies=[180.0], fc=180.0, dfc=5.0, Q=1e4, fs=4096)
estimate = window.filter(signal)             # signal: the real samples
```

Notebook: [02_Oscillators](../../share/notebooks/02_Oscillators.ipynb).

## Reconstruction: resampling artifacts and their fixes

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


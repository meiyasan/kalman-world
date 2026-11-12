# Power grid

The mains voltage is a 50 or 60 Hz wave plus harmonics at multiples of that
frequency. Their size is the measure of power quality. The filter follows the
amplitude and phase of each harmonic sample by sample, where a Fourier
transform needs a whole window.

## The model

```
state        x = [A cos φ, A sin φ] for each harmonic
dynamics     x stays constant                             (F = I)
measurement  z(t) = Σ over harmonics of  x₁ cos ωt - x₂ sin ωt       (the raw voltage)
```

The amplitude and phase of a steady wave do not change, so nothing moves in the
state. The oscillation is in the measurement: `H` changes at every sample.

## How it is built

- **Time stepping.** There is none to do: `F` is the identity. All the time
  dependence sits in `H(t)`.
- **No demodulation.** The raw voltage goes straight into the filter. Compare
  the oscillator models, which first move the line down to zero frequency.
- **Process noise** sets how fast a harmonic may change. A small value gives a
  quiet estimate that is slow to notice a fault.
- **Distortion.** The total harmonic distortion follows from the estimated
  amplitudes, with its uncertainty.

## Using it

```python
from kalman.apps import PowerGridStateModel

grid = PowerGridStateModel("feeder", f0=50.0, harmonics=5, fs=1024.0,
                           measurement_uncertainty=1.0, process_uncertainty=0.05)
estimate = grid.filter(voltage)              # voltage: the sampled waveform, in volts
```

Notebook: [03_PowerGrid](../../share/notebooks/03_PowerGrid.ipynb).

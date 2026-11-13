# Gravitational-wave polarization

A gravitational wave stretches space in two directions at once, and two
detector channels record it. The way the two channels combine, the
**polarization**, tells how the source is oriented. The filter follows the
amplitude, the phase and the polarization as the wave sweeps up in frequency.

## The model

```
state        x = [log amplitude, the phase, the two polarization angles]
                 (8 numbers: each one with its rate of change)
measurement  z = the two channels, real and imaginary parts          (4 numbers)
```

The amplitude is kept as its logarithm so that it cannot turn negative. A
negative amplitude with the phase shifted by half a turn is the same signal,
and a filter free to choose between the two loses the phase.

## How it is built

- **Time stepping.** The state moves linearly: one fixed matrix per step.
- **A nonlinear measurement.** The channels depend on the state through sines
  and cosines, so this is an extended Kalman filter. The model gives the
  derivative of the measurement in closed form.
- **It runs in C++.** `filter()` hands the whole pass to the C++ twin
  `Kalman::Apps::Gravitational::Polarization` (registered `polarization`; see
  `kalman model polarization` for its components and units). The noise matrices
  and the sample rate are pushed from the Python model; turning a complex
  signal into four real columns (`preprocess`) stays in Python.
- **The signals are in toy units**: an amplitude near 1 and a few hertz. A real
  wave has a strain near 1e-21 between ten hertz and a kilohertz.

## Using it

```python
from kalman.apps import PolarizationModel

wave = PolarizationModel("pair", fs=100.0, sigma_n=1.0)
```

Notebook: [05_GravWavePolarization](../../share/notebooks/05_GravWavePolarization.ipynb).

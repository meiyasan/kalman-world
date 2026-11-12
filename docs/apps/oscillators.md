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

## Using it

```python
from kalman.apps import OscillatorModel

window = OscillatorModel("mic", frequencies=[180.0], fc=180.0, dfc=5.0, Q=1e4, fs=4096)
estimate = window.filter(signal)             # signal: the real samples
```

Notebook: [02_Oscillators](../../share/notebooks/02_Oscillators.ipynb).

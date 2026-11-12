# GPS and relativity

A receiver works out where it is from the signals of several satellites. The
satellite clocks run at a different rate from clocks on the ground, because the
satellites move fast and sit higher in the Earth's gravity. Left uncorrected,
that difference becomes a position error.

## The model

```
state        x = [position (3), clock offset, clock drift]
dynamics     the receiver stays put; the clock offset grows with the drift
measurement  z = for each satellite, its distance plus the clock offset     (a pseudorange)
```

## How it is built

- **Time stepping.** The motion is linear and its step is exact.
- **A nonlinear measurement.** The distance to a satellite is not linear in the
  position, so this is an extended Kalman filter. The model gives the derivative
  of the distance in closed form.
- **The satellites move.** Their positions are computed from their orbits at
  each measurement time.
- **Relativity.** `apply_relativistic=True` removes the relativistic part from
  the pseudoranges before filtering. Switching it off shows the error: the
  filter then finds a clock drift of about 38 microseconds a day.
- **What is idealised.** The receiver clock is perfect, the Earth does not
  rotate, and there is no atmosphere. Real satellite clocks are also detuned
  before launch to cancel the steady part of the effect; here they are not, so
  that the filter can find it.

## Using it

```python
from kalman.apps import GPSModel

receiver = GPSModel("rx", receiver_lat=45.0, receiver_lon=10.0, n_sat=8)
pseudoranges = receiver.simulate(600.0, seed=0)      # ten minutes, one fix a second
estimate = receiver.filter(pseudoranges)
```

Notebook: [11_GPSRelativity](../../share/notebooks/11_GPSRelativity.ipynb).

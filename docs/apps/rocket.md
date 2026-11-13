# Rockets

Four models of one vehicle, from the pad to orbit and back.

| Class | Phase | State |
|---|---|---|
| `RocketLiftoffModel` | vertical climb while the propellant burns | altitude, speed, mass |
| `RocketAscentModel` | guided climb to orbit, in two dimensions | position and velocity; thrust is the control |
| `RocketDescentModel` | powered descent of a booster that can tilt | position, velocity, tilt, tilt rate, mass |
| `RocketLandingModel` | the final vertical landing burn | altitude, speed, mass |

## The model

Mass is a state because the rocket burns it:

```
altitude' = speed
speed'    = thrust / mass - gravity - drag / mass
mass'     = -burn rate                       (zero once the propellant is gone)
```

Even at constant thrust the acceleration grows as the rocket gets lighter. The
sensor measures position only. The filter recovers the speed and the remaining
mass.

## How it is built

- **Time stepping.** Each model writes its rate of change and takes one RK4
  step per sample, the framework's fourth-order Runge-Kutta integrator.
- **Guidance inside the model.** The descent and landing models contain their
  own autopilot: it throttles the engine to follow a braking profile and tilts
  the booster toward the pad. The filter therefore predicts a controlled
  vehicle.
- **Units.** Liftoff, descent and landing use metres, seconds and kilograms.
  The ascent to orbit uses megametres and kiloseconds, so that positions and
  velocities are numbers of similar size.
- **Process noise.** Each model passes `L = I`, so the noise acts on every
  state. Without it the framework finds that the step does not respond to
  noise and applies none.
- **C++.** All four have a twin in `libKalmanApps` (`rocket-liftoff`,
  `rocket-ascent`, `rocket-descent` and `rocket-descent-control`,
  `rocket-landing`): the derivative, the guidance and the closing step (mass
  kept at the dry mass, a landed stage held on the pad) in C++, so
  `model.filter(z)` and, with a thrust or throttle/gimbal program,
  `model.filter(z, uk)` run wholly in `Kalman::Filter`. A custom integrator,
  or a subclass that changes the dynamics, keeps the Python physics.

## Using it

```python
from kalman.apps import RocketLiftoffModel

rocket = RocketLiftoffModel()
flight = rocket.sample(160.0)                # altimeter readings through the burn
estimate = rocket.filter(flight.zk)
print(rocket.metrics(estimate))              # burnout, apogee, peak acceleration
```

Notebook: [06_RocketScience](../../share/notebooks/06_RocketScience.ipynb).

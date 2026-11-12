# Suspension

A quarter of a car: the body, a wheel, and the spring and damper between them,
rolling over a rough road. Sensors measure how the body moves. The filter
recovers the motion of the wheel as well.

## The model

```
state        x = [body height, body speed, wheel height, wheel speed]
input        u = the height of the road under the wheel
dynamics     body:   the spring and the damper pull it toward the wheel
             wheel:  the same forces, reversed, plus the tyre pressing on the road
```

The car has two resonances: the body bouncing on its springs at about 1 Hz, and
the wheel hopping on the tyre at about 10 Hz.

## How it is built

- **Time stepping.** The equations are linear, so the model gives the matrix of
  the continuous motion and the framework turns it into an exact step with a
  matrix exponential. No integrator is involved and no step error is made.
- **Process noise** is a random acceleration on each mass, carried through the
  same exact step. `process_uncertainty` is in m/s^1.5.

## Using it

```python
from kalman.apps import SuspensionModel

car = SuspensionModel("corner", m_s=300.0, m_u=50.0, k_s=20_000.0, b_s=1500.0, k_t=180_000.0,
                      measurement_uncertainty=0.002, process_uncertainty=0.05, fs=200.0)
readings = car.sample(4.0, uk=road).zk        # road: its height under the wheel, one value per step
estimate = car.filter(readings, road)
```

Notebook: [04_MechanicalSuspension](../../share/notebooks/04_MechanicalSuspension.ipynb).

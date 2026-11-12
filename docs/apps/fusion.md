# Sensor fusion

Several sensors measure the same moving object, each with its own noise and its
own rate. One filter combines them into an estimate better than any sensor
gives alone.

## The model

```
state        x = [position, velocity]
dynamics     constant velocity, disturbed by a random acceleration
measurement  one row per sensor: a position sensor reads x₁, a velocity sensor reads x₂
```

## How it is built

- **Sensors are data.** A model is built from a list of `SensorSpec`: a name,
  what it observes, its noise, and its rate. Adding a sensor adds a row to `H`
  and an entry to `R`.
- **Different rates.** A sensor that has nothing new at a given step
  contributes no measurement at that step, and the filter skips it.
- **Weights.** Each sensor counts in proportion to one over its variance. A
  sensor ten times noisier counts a hundred times less.
- **Time stepping.** Linear and exact.

## Using it

```python
from kalman.apps import SensorFusionModel, SensorSpec

car = SensorFusionModel("car", sensors=[
    SensorSpec("gps",      observes="position", uncertainty=3.0, rate=5.0),
    SensorSpec("odometry", observes="velocity", uncertainty=0.1, rate=50.0),
], fs=50.0)
```

Notebook: [09_SensorFusion](../../share/notebooks/09_SensorFusion.ipynb).

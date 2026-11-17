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

## The same model, from operations

`SensorFusionModel` is a convenience over the framework's operations, and gives
the same record as the model built with them:

```python
from kalman.framework import KalmanModel, KalmanSensor, compose

car = KalmanModel("gps", fs=50.0, F=F, Q=Q, H=[[1.0, 0.0]], R=[[3.0**2]])       # its own sensor: the GPS
car = car.add_sensor(KalmanSensor("odometry", G=[[0.0, 1.0]], R=0.1**2))       # one more, on the velocity
```

The operations also cover what a list of rows cannot:

- **An instrument that is not calibrated.** `compose(sensor, name="lidar").calibrate(gain, offset)`
  gives the model the parameters `lidar.gain` and `lidar.offset`, and
  `model.joint({"lidar.offset": (1.0, 0.0)})` estimates the offset with the
  state, as long as another sensor reads the same quantity without it.
- **A sensor that reports late.** `KalmanSensor(..., delay=0.2)` on a model with
  `.lag(10)`: the model carries its last ten states and the sensor reads the
  oldest. Read as if it were current, a 0.2 s delay on a target at 8 m/s is a
  1.6 m error.

Steps 6 to 8 of the notebook show them one at a time.

## Generic sensors

`PositionSensor`, `DopplerSensor` and `CameraSensor` are part of the framework
(`from kalman.framework import ...`), for any model: each is told the indices
of the state components it reads. They are written in C++ with their Jacobians
and described in [Models](../4_Models.md#generic-sensors).

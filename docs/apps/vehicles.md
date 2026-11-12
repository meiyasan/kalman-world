# Vehicles

Two models of a car: how fast it accelerates in a straight line, and how it
keeps to its lane.

## The speed test

```
state        x = [distance, speed]
dynamics     speed' = (drive - drag - rolling - slope) / mass
             drive = the smaller of  grip × weight  and  power / speed
measurement  z = distance                                  (GPS or radar)
```

Off the line the tyres limit the force. Above some speed the engine's power
does. The filter recovers the speed and the acceleration from position alone.

## Lane keeping

```
state        x = [offset from the lane centre, heading, speed]
control      u = steering angle
dynamics     offset' = speed × sin(heading)
             heading' = speed × tan(steering) / wheelbase
measurement  z = offset                                    (a lane camera)
```

## How they are built

- **Time stepping.** Both models give only the rate of change
  (`f_continuous`) and leave the stepping to an integrator. `CarSpeedModel` uses
  the framework default, RK4. `CarLaneModel` declares Euler, the scheme it was
  tuned with.
- **Control.** The lane model takes a steering input. Its notebook compares
  controllers that compute that input from the filter's estimate: proportional,
  PID, LQR, a sampling planner, and a small network trained to imitate it.
- **C++.** `CarSpeedModel` has a twin in `libKalmanApps`, which takes the same
  RK4 step. `CarLaneModel` does not: its filter is driven by a controller
  written in Python.

## Using it

```python
from kalman.apps import CarSpeedModel

car = CarSpeedModel("CAR", power=150_000, mass=1500)
run = car.sample(15.0)                       # 15 s of simulated position fixes
estimate = car.filter(run.zk)
print(car.metrics(estimate))                 # 0-100 km/h, standing 400 m, top speed
```

Notebook: [12_KalmanAI](../../share/notebooks/12_KalmanAI.ipynb).

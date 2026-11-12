# Applications at a glance

Each application is a model of one physical system, built from the same pieces
of the framework. Each page says what the model is, how it was built, how it
steps in time, and how to run it.

| Application | The system | Filter | Time stepping |
|---|---|---|---|
| [Ballistics](apps/ballistics.md) | a projectile under gravity and drag | extended | the model's own step |
| [Oscillators](apps/oscillators.md) | narrow lines in a signal | linear | exact |
| [Power grid](apps/powergrid.md) | the harmonics of the mains voltage | linear | none: the state is constant |
| [Suspension](apps/suspension.md) | a car body and wheel on a rough road | linear | exact |
| [Gravitational-wave polarization](apps/polarization.md) | a wave seen by two channels | extended | exact |
| [Calorimeter](apps/calorimeter.md) | a particle shower through layers | linear, stepping over depth | exact |
| [Rockets](apps/rocket.md) | liftoff, ascent, descent and landing | extended | RK4 |
| [Particle tracking](apps/tracking.md) | charged tracks in a drift chamber | combinatorial, stepping over layers | RK4 along the path |
| [Space navigation](apps/space.md) | a cruise to the Moon or a planet | extended | RK4 |
| [Sensor fusion](apps/fusion.md) | several sensors on one object | linear | exact |
| [Computer vision](apps/vision.md) | objects detected in video frames | linear | exact |
| [GPS and relativity](apps/gps.md) | a receiver and its satellites | extended | exact |
| [Vehicles](apps/vehicles.md) | a speed run and lane keeping | extended | RK4 and Euler |
| [Turbulent flow](apps/flow.md) | a 2-D fluid seen through probes | ensemble | a spectral solver with RK4 |
| [Storm tracking](apps/storms.md) | a cyclone and a tornado environment | extended, several regimes | the model's own step |

"Exact" means the motion is linear, so its step is a matrix computed once with
no approximation. RK4, the fourth-order Runge-Kutta scheme, is the default
integrator of the framework. A model uses it only when it describes its motion
by a rate of change; [Turbulent flow](apps/flow.md) explains how that differs
from the solver that computes gradients in space.

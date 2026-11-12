# Turbulent flow

A two-dimensional turbulent fluid, known only through a hundred scattered
probes. The state is the whole vorticity field on a 64 × 64 grid: 4096 numbers.
The filter rebuilds the field between the probes.

This is the application where the difference between a **solver** and an
**integrator** matters, so this page explains it.

## The model

```
state        x = the vorticity ω at every grid point           (n = 64 × 64 = 4096)
dynamics     ∂ω/∂t = -u·∇ω + ν ∇²ω + χ cos(k_f y)              (Navier-Stokes, Kolmogorov forcing)
measurement  z = ω at 100 fixed probe points, plus noise
```

`u` is the velocity of the fluid, which follows from the vorticity. The first
term carries the vorticity along with the flow, the second smooths it, and the
third is a steady stirring that keeps the turbulence alive.

## Solver and integrator: two different jobs

The equation has derivatives in **space** (`∇ω`, `∇²ω`) and one in **time**
(`∂ω/∂t`). The framework gives each to a different object.

**The solver handles space.** `NS2D` is a spectral solver. It never takes a
difference between neighbouring grid points. It transforms the field into waves
with a Fourier transform, where a derivative is a multiplication:

```
ω(x, y)   --FFT-->   ω̂(kx, ky)

∂ω/∂x     <->   i kx ω̂
∂ω/∂y     <->   i ky ω̂
∇²ω       <->   -(kx² + ky²) ω̂
stream function    ψ̂ = ω̂ / (kx² + ky²)
velocity           u <-> i ky ψ̂ ,   v <-> -i kx ψ̂
```

For a smooth periodic field this gradient is exact to rounding error, where a
difference between neighbours is only approximate. The product `u·∇ω` is not a
multiplication of waves, so the solver goes back to the grid, multiplies point
by point, and transforms again. That product creates waves too short for the
grid, which would come back as false long ones; the solver removes them by
zeroing the shortest third of the waves (the 2/3 rule).

What comes out is one function, `rhs_hat(ω̂)`: given the field, the rate of
change of every wave. Space is gone. What is left is 4096 ordinary equations in
time.

**The integrator handles time.** It knows nothing about grids or gradients. It
receives a function that turns a state into a rate of change, and advances the
state by one step `dt`. The default is fourth-order Runge-Kutta (RK4): it
evaluates `rhs_hat` four times per step and combines the results. Euler and
Dormand-Prince are the other choices, passed as `integrator=`.

So the gradient is computed by the solver, four times per time step, each time
the integrator asks for the rate of change.

| | Space derivatives | Time stepping |
|---|---|---|
| Turbulent flow | spectral solver: FFT, multiply by `i k` | RK4, composed by the solver |
| Car speed test | none: the state is two numbers | RK4, the model's default integrator |
| Lane keeping | none | Euler, declared by the model |
| Ballistics | none | the model's own closed-form step |

A model without a field has no solver. It needs at most an integrator, and
several models (ballistics, storms) write their step out directly and use
neither.

A finite-difference solver also ships with the framework
(`FiniteDifferenceSolver`). It does take differences between neighbours, and it
suits fields that are not periodic.

## How it is built

- **No covariance matrix.** A 4096 × 4096 covariance is too large to store and
  update, and the flow is chaotic, so there is no useful matrix `F` either. The
  model therefore allows only the ensemble and particle engines. The ensemble
  filter (EnKF) carries a few dozen copies of the field and reads the
  uncertainty from how far they spread.
- **One filter step is many solver steps.** `f` advances the field by one
  assimilation cycle, which is `cycle` RK4 steps of the solver (30 by default).
- **Localisation.** With few copies, two distant points can look correlated by
  chance. A taper that falls to zero with distance stops a probe from changing
  the field far away from it.
- **A learned forecast can stand in for the solver.** `model.surrogate = fn`
  replaces the solver with a trained network inside the same filter. On its own
  the network drifts; restarting it from the solver every few cycles keeps it
  accurate. Notebook 13 measures both.

## Using it

```python
from kalman.apps import KolmogorovFlowModel, TwinExperiment

flow = KolmogorovFlowModel("probes", grid=64, n_obs=100)
solver = flow.solver                        # the spectral solver, usable on its own

experiment = TwinExperiment(flow, n_cycles=60, n_members=16)   # simulate a truth, observe it, filter
```

Notebook: [13_Kolmogorov](../../share/notebooks/13_Kolmogorov.ipynb).

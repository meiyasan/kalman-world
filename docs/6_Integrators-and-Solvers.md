# Integrators and Solvers

Two distinct roles, often confused:

- an **integrator** advances *time*: `x(t) → x(t+dt)` given a continuous
  derivative `ẋ = f(x, u, t)`;
- a **solver** discretizes *space* (spectral, finite-difference, …) and
  **composes** an integrator for its time march.

For an ODE, `discretize` is the identity and the two coincide — which is why an
ODE model binds only an integrator, and a field (PDE) model binds a solver.
They are sibling packages; neither nests inside the other.

---

## Integrators

### The contract

```python
class KalmanIntegrator:
    KIND = None            # a KalmanIntegratorKind member
    tableau = None         # explicit Runge–Kutta Butcher tableau (c, A, b), or None

    def step(self, f, x, u=None, t=None, dt=1.0): ...      # override this
    def integrate(self, f, x, u=None, t=None, dt=1.0, n=1): ...   # default loop
```

Integrators own **no run state** and are **dtype-preserving** — they also step
complex spectral fields, so never coerce to `float` inside a `step()`.

### Shipped

| Kind | Class | Order | Notes |
|---|---|---|---|
| `euler` | `EulerIntegrator` | 1 | 1-stage explicit; cheapest |
| `rk4` | `RK4Integrator` | 4 | the default |
| `dormand_prince` | `DormandPrinceIntegrator` | 5(4) | embedded pair, adaptive substeps |
| `implicit_euler` | *reserved* | 1 | raises `NotImplementedError` — needs a Newton solve |

### Choosing one

Declare it on the model, or override per instance:

```python
from kalman.framework.constants import KalmanIntegratorKind

class MyModel(KalmanModel):
    DEFAULT_INTEGRATOR  = KalmanIntegratorKind.DORMAND_PRINCE
    ALLOWED_INTEGRATORS = {KalmanIntegratorKind.RK4, KalmanIntegratorKind.DORMAND_PRINCE}
    SOLVER_SUBSTEPS     = 4      # sub-divide each 1/fs filter step into 4 integrator steps

model = MyModel("m", fs=100, integrator=KalmanIntegratorKind.RK4)   # per-instance override
```

`SOLVER_SUBSTEPS` is the knob for stiff-ish dynamics at a low sampling rate:
the filter step stays `1/fs`, but the integrator takes that many smaller steps
inside it.

> **The integrator only matters if `f()` is integrated.** The linear engine and
> the native C fast path use `F()` matrices and never call `f()`. On a
> matrices-only model, setting `integrator=` changes nothing at all. It takes
> effect when the model overrides `f_continuous` (or a solver drives a field).

### Verifying it took effect

```python
model.default_integrator()       # -> the resolved instance
```

---

## Solvers

### The contract

```python
class KalmanSolver:
    KIND = None
    DEFAULT_INTEGRATOR = KalmanIntegratorKind.RK4

    def __init__(self, *, integrator=None, backend=None, device=None): ...
    def discretize(self, field): ...                     # identity by default
    def integrate(self, rhs, state, u=None, t=None, dt=1.0, n=1): ...   # delegates
    def solve(self, field, steps=1, u=None, t=None): ... # override
```

Composition, not inheritance: a solver *has-a* integrator, because different
discretizations pair with different steppers.

### Shipped

| Kind | Class | Mesh | Physics hook |
|---|---|---|---|
| `spectral` | `SpectralSolver` | regular **periodic**, square or rectangular | `rhs_hat(wh) → dω̂/dt` (spectral space) |
| `finite_difference` | `FiniteDifferenceSolver` | regular rectangular, `periodic` / `dirichlet` / `neumann` | `rhs(field, t) → ∂f/∂t` (real space) |
| `finite_volume` | *reserved* | — | raises `NotImplementedError` |
| `finite_element` | *reserved* | — | raises `NotImplementedError` |

Both accept square **or rectangular** grids (`n=(ny, nx)`) and physical domain
lengths (`length=(Ly, Lx)`); wavenumbers scale as `2π/L`.

```python
from kalman.framework.solvers import SpectralSolver, FiniteDifferenceSolver

s  = SpectralSolver(n=(32, 64), dt=0.01, length=(2*np.pi, 4*np.pi))
fd = FiniteDifferenceSolver(n=32, dt=1e-3, bc="neumann", nu=0.05)
out = fd.solve(field, steps=5)
```

The built-in `FiniteDifferenceSolver` default physics is the diffusion equation
`∂f/∂t = ν∇²f`, so the class is usable and testable on its own. A real
application subclasses it — `kalman.apps.weather.flow.NS2D` subclasses
`SpectralSolver` and supplies the Kolmogorov forcing.

### The Butcher-tableau protocol

Field solvers do **not** call `integrator.step()`. They read
`integrator.tableau` — the explicit Runge–Kutta `(c, A, b)` — and run the stages
themselves, so the same stepping scheme works in NumPy or Torch, on real or
complex fields. That is what makes integrators genuinely swappable inside a
solver.

The consequence: an integrator with `tableau = None` cannot drive a field
solver.

```python
SpectralSolver(n=32, dt=0.01, integrator=resolve_integrator(KalmanIntegratorKind.IMPLICIT_EULER))
# TypeError: Implicit Euler Integrator has no explicit tableau; the spectral solver
#            needs an explicit Runge–Kutta method.
```

`DormandPrinceIntegrator` *does* expose a tableau, but field solvers use its
5th-order weights as a **fixed-step RK5** — the adaptive error control is
dropped in that path.

### GPU / Torch backend

Any solver takes `backend` (`"numpy"` default, `"torch"`) and `device`
(`"cpu"`, `"cuda"`, `"mps"`), settable per instance or globally:

```bash
$ export KALMAN_BACKEND=torch KALMAN_DEVICE=mps
```

```python
SpectralSolver(n=256, dt=0.005, backend="torch", device="cuda")
```

The Torch path advances a batched `(N, g, g)` ensemble as one tensor operation —
the EnKF forecast's dominant cost — and converts at the NumPy boundary, so
callers are unchanged.

> **Torch/MPS has no float64.** Use float32 on `mps`, float64 elsewhere.

## Customization

The kind enums are the authority for **shipped** schemes. Your own scheme does
not need a member in them.

### 1 — Swap in a different shipped scheme

By kind, on the class or per instance. Nothing else to do.

### 2 — Subclass a shipped solver for your own physics

The usual extension point: inherit `SpectralSolver` / `FiniteDifferenceSolver`,
override `rhs_hat` / `rhs`, and keep the mesh, dealiasing, backends and tableau
machinery.

### 3 — Write a scheme of your own

Subclass the base and implement `step()` (integrator) or `solve()` (solver). No
`KIND` is needed — for a custom scheme the **base class is the contract**, and
`ALLOWED_INTEGRATORS` / `ALLOWED_SOLVERS` do not apply, since a closed enum
cannot describe a kind it has never heard of.

```python
from kalman.framework import KalmanIntegrator

class HeunIntegrator(KalmanIntegrator):
    NAME = "Heun"
    tableau = ([0.0, 1.0], [[], [1.0]], [0.5, 0.5])   # optional: lets field solvers use it

    def step(self, f, x, u=None, t=None, dt=1.0):
        k1 = f(x, u, t)
        k2 = f(x + dt * k1, u, t + dt)
        return x + 0.5 * dt * (k1 + k2)

model = MyModel("m", fs=50, integrator=HeunIntegrator())   # instance
model = MyModel("m", fs=50, integrator=HeunIntegrator)     # or the class
type(model.default_integrator())                           # HeunIntegrator
```

### 4 — Register it under a name

Registration makes the scheme addressable everywhere a kind name is accepted —
including a [`KalmanSpec`](9_Python-API.md)'s `bindings`, and therefore over
REST and MCP — without editing `constants.py`.

```python
from kalman.framework import register_integrator, register_solver, available_integrators

register_integrator("heun", HeunIntegrator)          # also works as a decorator
available_integrators()
# ['dormand_prince', 'euler', 'heun', 'implicit_euler', 'rk4']

MyModel("m", fs=50, integrator="heun")               # by name
{"bindings": {"integrator": "heun"}}                 # in a spec

unregister_integrator("heun")                        # and back out again
```

Registered names show up in `catalog()["integrators"]` / `["solvers"]`, so a
client can discover them the same way it discovers the shipped ones.

To **rebind a shipped kind** process-wide, pass the enum member rather than its
name — deliberately explicit, because it changes what everyone else gets:

```python
register_integrator(KalmanIntegratorKind.EULER, HeunIntegrator)
```

`register_*` refuses a class that is not a subclass of the right base
(`TypeError`), and refuses a *name* that collides with a shipped kind
(`ValueError` — pass the enum member if that is really what you meant).

### 5 — Add a kind to the framework itself

For a scheme that belongs in the repository, add the member to
`framework/constants.py` **and** its registry entry in the same commit;
`tests/test_structure.py` requires every registry to cover its whole enum.
Registration is for downstream code; the enum is for shipped schemes.

---

## Errors, not silent fallbacks

An unrecognised kind raises. A typo that quietly ran a different scheme used to
be possible and is not any more:

```python
resolve_integrator("nonsense")
# ValueError: unknown integrator kind 'nonsense'; choose from
#             ['dormand_prince', 'euler', 'implicit_euler', 'rk4'] or register_integrator(...)

resolve_solver("nonsense")   # ValueError
resolve_engine("nonsense")   # ValueError
```

`None` still means "the default" (RK4 / spectral / extended), which is what a
model's `DEFAULT_*` resolution relies on.

---

## Remaining rough edges

**The mesh contract is per-class, not inherited.** `KalmanSolver` carries only
`backend`, `device` and the composed integrator. Grid attributes (`n`, `shape`,
`length`, `dt`) and the physics hook are re-declared independently by each
concrete solver, and the hook is even named differently between them (`rhs_hat`
in spectral space vs `rhs` in real space). A solver written from the base class
up inherits no mesh handling and no validation — subclass a shipped solver
unless you are adding a whole discretization family.

**No error-control contract.** `step()` returns only the new state, with no
error estimate, so embedded pairs cannot adapt at the framework level;
`DormandPrinceIntegrator` handles its own substepping internally, and field
solvers use it as fixed-step RK5.

---

## Summary table

| I want to… | Route | Status |
|---|---|---|
| use RK4 instead of Euler | `DEFAULT_INTEGRATOR` or `integrator=` | supported |
| sub-step within one filter step | `SOLVER_SUBSTEPS` | supported |
| new PDE physics on an existing mesh | subclass `SpectralSolver` / `FiniteDifferenceSolver` | supported, intended |
| run a field solver on GPU | `backend="torch"`, `device=…` | supported |
| plug my own stepper into one model | custom class or instance via `integrator=` | supported, no `KIND` needed |
| address my scheme by name / from a spec | `register_integrator("name", cls)` | supported |
| rebind a shipped kind globally | `register_integrator(Kind.EULER, cls)` | supported, explicit |
| add a kind to the framework | `constants.py` + registry, in-tree | supported for contributors |
| catch a typo'd kind | — | raises `ValueError` |
| inherit mesh handling in a new solver family | — | **not supported**, subclass a shipped solver |

Next: [Providers](7_Providers.md).

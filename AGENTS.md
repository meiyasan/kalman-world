# AGENTS.md — working on this codebase with an LLM

Read this before changing anything. It records the conventions the code relies
on, which are easy to violate from a cold read. Written for coding agents
(Claude Code, Cursor, Copilot, …) and for humans reviewing their output.

---

## 1. What this project is

A Kalman-filtering framework: a small set of composable **contracts** in
`modules/framework/`, and **domain applications** in `modules/apps/` built on
them. `modules/` is published as the `kalman` package (see
`[tool.setuptools] package-dir` in `pyproject.toml`), so on disk it is
`modules/apps/…` but in imports it is `kalman.apps.…`.

### The seven roles

| Role | Question it answers | Base (always in `framework/`) | Concretes |
|---|---|---|---|
| **Model** | WHAT is the system? `f/h/Q/R` | `framework/model.py` | `framework/models/`, `apps/` |
| **Engine** (alias **Estimator**) | HOW to infer under uncertainty | `framework/engine.py` | `framework/engines/` |
| **Integrator** | HOW to step *time*: `x(t) → x(t+dt)` | `framework/integrator.py` | `framework/integrators/` |
| **Solver** | HOW to discretize *space* | `framework/solver.py` | `framework/solvers/` |
| **Filter** | WHEN — the forward/backward workflow | `framework/filter.py` | `filters/` |
| **Provider** | WHERE the data comes from | `framework/provider.py` | `providers/`, `apps/<domain>/providers/` |
| **Controller / Router** | acting on, and selecting between, models | `framework/control.py`, `routers/` | `framework/controllers/`, `routers/` |

**Integrator vs Solver** is a real distinction, not a synonym: an *integrator*
advances time (Euler/RK4/Dormand–Prince); a *solver* discretizes space
(spectral/finite-difference/…) and **composes** an integrator for its time
march. For an ODE they coincide, so a model binds only an integrator. They are
**sibling packages** — never nest one inside the other.

---

## 2. Non-negotiable invariants

`tests/test_structure.py` enforces most of these. If you break one, that suite
fails — fix the code, do not weaken the test.

1. **`framework/` owns every abstract contract, base class, and registry
   resolver.** Concrete implementations live outside it. A new base class goes
   in `framework/`, full stop.
2. **Plug-ins are declared, not imported.** A model names its engine/integrator/
   solver via a `Kind` enum (`DEFAULT_ENGINE`, `DEFAULT_INTEGRATOR`,
   `DEFAULT_SOLVER`) and *never* imports a concrete engine/solver. Resolution
   goes through the `*_BY_KIND` registry + `resolve_*()`.
3. **Every registry covers its whole kind enum**, and every concrete class tags
   the matching `KIND`. Adding an enum member means adding a registry entry
   (a reserved stub that raises `NotImplementedError` is acceptable).
4. **Every app domain is a package** with per-variant submodules, an `__init__`
   re-exporting its public names, and a `VARIANTS` dict of `KalmanModel`
   subclasses. No new flat `apps/*.py` modules.
5. **Public names are stable.** `from kalman.apps import X` must keep working
   after any reorganisation; `kalman.apps.X` must be the *same object* as the
   domain's `X`.
6. **Optional dependencies are lazily imported and guarded** (`torch`, `mcp`,
   `fastapi`, `bcrypt`, `pyjwt`, `httpx`, `cdsapi`, `xarray`, `h5py`). The core
   must import and the suite must pass without them. Entry points print an
   install hint and exit 0 rather than crashing.
7. **No `eval`/`exec` in `modules/api/`.** Spec authoring is data-only; code is
   only ever *emitted* for a human to review (`scaffold`), never executed.

---

## 3. Layout

```
modules/                        → the `kalman` package
  framework/                    contracts + bases + registries (the core)
    model.py engine.py integrator.py solver.py provider.py filter.py control.py
    models/ engines/ integrators/ solvers/ routers/ controllers/ io/ abstract/ native/
  filters/                      public filter names (KF/EKF/UKF/PF/EnKF/CKF/IKF)
  providers/                    generic data sources (synthetic/file/sensor/remote)
  api/                          JSON facade + KalmanSpec authoring + scoped-JWT auth
  mcp/  rest/                   transports over the facade (optional deps)
  apps/                         domain applications
    astro/         rocket  space
    gravitational/ gravwave  interferometer
    resonators/    oscillators  powergrids  suspensions
    weather/       flow  storms  geo  surrogate  data/  providers/
    ballistics/  calorimeter/  fusion/  gps/  tracking/  vehicles/  vision/
  runtime/  helpers.py
tests/                          mirrors concerns, not files
share/notebooks/                executed, output-bearing narrative docs
```

**Families** (`astro`, `gravitational`, `resonators`) group related domains and
expose a `DOMAINS` map. Grouping is by shared foundation or application area —
*not* by superficial similarity. Domains inside a family keep their own
`VARIANTS`; cross-family imports are normal (e.g. `interferometer` uses
`resonators.oscillators`).

`apps/` contains **packages only** — there are no flat domain modules and no
shims. Import from the canonical path (`kalman.apps.weather.flow`, …) or from
the stable `kalman.apps` surface.

---

## 4. How to add things

**A new model variant in an existing domain** — add
`apps/<domain>/<variant>.py`, put shared enums/helpers in that domain's
`base.py`, then register it in the domain `__init__` (`import`, `VARIANTS`,
`__all__`). Nothing else needs to change.

**A new domain** — create the package with the layout above. If it clearly
belongs to a family, put it there; otherwise leave it top-level.

**A nonlinear model** — prefer overriding `f_continuous(x, u, t)` (the
derivative) and let the bound integrator produce the discrete `f()`. Only
override `f()` directly if the discrete map is what you genuinely have.
Pre-set `self.n` **before** `super().__init__()` — construction runs
`_check_shapes`, which evaluates `F(zeros)` and hence `f` at `x=0`.

**A new engine / integrator / solver / provider** — subclass the `framework/`
base, tag `KIND`, and add it to the registry. Follow the existing
`resolve_*()` pattern including the `%autoreload` name-fallback.

**Registering the MCP server with a client** — `make mcp-clients` lists the
MCP clients detected on this machine; `make mcp-install` (or
`python tools/mcp_install.py --all`) registers the server with all of them,
`--dry-run` shows the diff first and `--remove` unregisters. The script knows
each client's config location and container key (`mcpServers`, VS Code's
`servers`, Codex's TOML table), edits only our own entry and backs the file up
first. To add a client, append it to `CLIENTS` in that script.

**A new API verb** — add it to `modules/api/`, declare its scope (`read`,
`send_measurement`, `write_models`, `admin`), then expose it in `mcp/` and
`rest/` as a thin adapter. Never put logic in a transport.

---

## 5. Numerical conventions that bite

- `dt` is **never** passed into `f()`; only absolute `t` is. Derive
  `dt = 1/self.fs` (models do this; `SOLVER_SUBSTEPS` sub-divides it).
- Integrators are **dtype-preserving** — they also step complex spectral fields.
  Do not coerce to `float` in a `step()`.
- Solvers accept square *or* rectangular grids (`n=(ny, nx)`) and physical
  domain lengths (`length=(Ly, Lx)`); wavenumbers scale as `2π/L`. Validate
  incoming fields against the mesh and fail with a clear message.
- The linear engine and the native C fast path use `F()` matrices and never
  call `f()`. Changes to `f()` do not affect them — and must not break them.
- Torch/MPS has **no float64**: use float32 on `mps`, float64 elsewhere.
- Comparing several filter variants in one loop: give each its **own RNG
  stream** (`seed=`) and share only the observation stream, or results
  cross-contaminate. This is a real bug we already fixed once.

---

## 6. Verification — required before claiming done

```bash
make setup                     # one-command onboarding (deps, extras, accel, self-check)
make tests                     # or: python -m pytest tests/ -q
python -m pytest tests/test_structure.py -q     # architecture invariants
make linter                    # ruff, correctness-focused
```

Current baseline: **453 passed, 2 skipped** (skips are optional-dependency
gates). The suite must stay green; a red suite is never "expected".

Install the optional extras to exercise the transports for real rather than
skipping them — `make setup`. The MCP
server is covered by a **live stdio session** test that spawns
`python -m kalman.mcp` and drives it as a client; it caught a real breakage
when SDK 2.0 renamed `FastMCP` to `MCPServer`, which a mocked test would have
missed.

If you touch `apps/` layout, also verify the notebooks still import:

```bash
python - <<'PY'
import json, re, glob, sys, io, contextlib
sys.path.insert(0, ".")
for nb in glob.glob("share/notebooks/*.ipynb"):
    src = "\n".join("".join(c["source"]) for c in json.load(open(nb))["cells"] if c["cell_type"]=="code")
    for s in re.findall(r"^(?:from\s+kalman[\w.]*\s+import\s+\([^)]*\)|from\s+kalman[\w.]*\s+import\s+[^\n(]+|import\s+kalman[\w.]*)", src, re.M):
        try:
            with contextlib.redirect_stdout(io.StringIO()): exec(compile(s,"<nb>","exec"), {})
        except Exception as e: print(f"{nb}: {s[:60]} -> {e}")
PY
```

**Notebooks are expensive** (some take ~15 min to execute and carry embedded
figures/animations). Prefer editing import paths textually — identical objects
keep stored outputs valid — over re-executing. Never re-execute casually.

---

## 7. Working style expected here

- **Verify, don't assert.** "Tests pass" means you ran them and are quoting the
  count. Physics claims get checked against an analytic result.
- **Report failures plainly**, with the output. A partially-working change
  described as working is worse than a failing one.
- **Preserve numerics across refactors.** When moving a model, keep its existing
  scheme (e.g. `CarLaneModel` stays explicit-Euler; "upgrading" it to RK4 would
  silently shift trajectories). Behaviour changes are a separate, deliberate
  commit.
- **Incremental over big-bang.** Restructure one domain at a time, suite green
  after each.
- **Don't commit** unless asked. Never commit secrets — `.cdsapirc` and
  credential files are gitignored and must stay that way. Outbound service
  credentials live in `providers/credentials.py`, distinct from the inbound JWT
  auth in `api/auth.py`.
- **Ask when a decision is the maintainer's** — naming, taxonomy, scope. Don't
  silently pick.

---

## 8. Known deferred work

- `apps/weather/surrogate.py` (the CNN forecast surrogate) should eventually
  become a learned `KalmanSolverKind.NEURAL` solver rather than a loose module.
- `IMPLICIT_EULER` integrator and the finite-volume/finite-element solvers are
  reserved stubs that raise `NotImplementedError`.
- Several `apps/` models still override `f()` directly and could move to
  `f_continuous` (rocket, space, tracking, storms).
- Notebook 14 still names `TyphoonModel`; `StormModel` (basin-agnostic) is the
  canonical class, with `TyphoonModel`/`HurricaneModel`/`CycloneModel` as basin
  presets.
- `README.md`'s filter table predates PF/CKF/IKF/EnKF and is stale.

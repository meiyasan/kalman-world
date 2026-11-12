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
| **Filter** | WHEN — the forward/backward workflow | `framework/filter.py` | `framework/filters/` |
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
    models/ engines/ filters/ integrators/ solvers/ routers/ controllers/ io/ abstract/ native/
    filters/                    public filter names (KF/EKF/UKF/PF/EnKF/CKF/IKF)
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
tests/                          mirrors concerns, not files (pytest)
  cxx/                          the libKalman suite (plain binaries, ctest)
include/Kalman/  src/           libKalman — the standalone C++ port (see §10)
examples/cxx/                   standalone C++ programs using libKalman
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

**A nonlinear model** - there is one method, `f`. A physical model declares
`DYNAMICS = KalmanDynamicsKind.CONTINUOUS` and writes `f` as the derivative
dx/dt; the framework decodes it into the discrete step with the model's
integrator, so engines only ever see a step. Leave `DYNAMICS` at `DISCRETE`
(the default) and write the step itself only when a discrete map is what you
genuinely have (a learned step, a layer-to-layer propagation, a rule). Never
write an integration loop inside a model. `after_step()` is the hook for what
must follow a decoded step (a clamp, a slot that is not integrated).
Pre-set `self.n` **before** `super().__init__()` — construction runs
`_check_shapes`, which evaluates `F(zeros)` and hence `f` at `x=0`.

**A new engine / integrator / solver / provider** — subclass the `framework/`
base, tag `KIND`, and add it to the registry. Follow the existing
`resolve_*()` pattern including the `%autoreload` name-fallback.

**A new filter** — it should be a *thin leaf*: `NAME`, `DESCRIPTION`, `KIND`,
and at most an `__init__` forwarding hyper-parameters to its engine (see
`filters/ukf.py`). A filter that defines `forward` / `backward` / `smooth` is a
smell: that is algorithm, and algorithm belongs to the engine. If one
predict/update step genuinely does not describe your algorithm — a
combinatorial search over a whole event, say — set `OWNS_PASS = True` on the
**engine** and implement `KalmanEngine.forward()` / `backward()` there.
`KalmanFilter` then hands the engine its data and turns the returned objects
into records via their `to_record(model, flavor)`; the result type (a track, a
cluster) stays private to the engine, and the leaf is thin again.
`CombinatorialKalmanEngine` is the worked example.

**Registering the MCP server with a client** — `make mcp-clients` lists the
MCP clients detected on this machine; `make mcp-install` (or
`python tools/mcp_install.py --all`) registers 'Kalman-AI' with all of them,
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
- **Never call `expm()` inside `f()`, `F()` or any other per-step hook.** The
  EKF re-evaluates `f()` `2n+1` times per step for the central-difference
  Jacobian, so anything loop-invariant in there is paid thousands of times per
  second. Two cases were found and removed:
  - A **nilpotent** generator (`A² = 0` — a slope drives a coordinate, nothing
    drives the slope: constant-velocity kinematics, straight-line tracks) has
    `exp(A·s) = I + A·s` **exactly**. Write the closed form. It is not an
    approximation; it agrees with `scipy.linalg.expm` to the last bit at small
    `s` and is 1–2 ULP *more* accurate at large `s`, because the exact answer is
    representable and Padé scaling-and-squaring rounds. Removing it took
    `BallisticModel.f()` from 21.1 µs to 8.8 µs and its EKF forward pass from
    1.9 s to 0.63 s (**3×**).
  - Otherwise, hoist it: `dt = 1/fs` and the generator are constants of a run.
  Discretisation belongs in `__init__` or in a cache, never in the hot path.

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
- The ballistic, car, rocket and space models are `DYNAMICS = CONTINUOUS`. The
  storm, GPS and oscillator models still write a discrete `f()` and could follow.
- `KalmanSolver` carries no mesh contract: `n`/`shape`/`length`/`dt` and the
  physics hook are re-declared per concrete solver, and the hook is named
  `rhs_hat` (spectral) vs `rhs` (finite-difference). A new discretization family
  inherits nothing; subclass a shipped solver instead.
- `KalmanIntegrator.step()` returns no error estimate, so embedded pairs cannot
  adapt at the framework level; field solvers run Dormand–Prince as fixed-step
  RK5 off its Butcher tableau.
- Two Doxygen warnings come from source docstrings, not the docs:
  `framework/filters/ckf.py` (an inline comment read as a declaration) and
  `helpers.py:358` (a docstring bullet list read as a member).
- A `KalmanSpec`'s top-level `fs` only reaches models whose `__init__` names
  `fs`; for `**kwargs` models it has to go in `dynamics.params`.
- `KalmanSpec`'s documented `labels` / `control` fields are accepted but ignored
  by `make_model`.

## 9. Documentation

`docs/` is a numbered Markdown tree (`docs/README.md` is the index); `AGENTS.md`
is for changing the code, `docs/` for using it. When you change a public
surface — a verb, an endpoint, a registry, an error message — update the page
that documents it in the same commit. Pages 6, 9, 10 and 14 quote real behaviour
including failure modes, so a fixed bug means an edited page.

Those same files are **published**: `make docs` renders them, plus `modules/`
docstrings, through Doxygen + doxygen-awesome (the `share/doxygen` submodule,
which carries its own nested `cmake` submodule). The CI `docs` job runs on every
commit and fails the pipeline if the build breaks; `pages` publishes from the
default branch and version tags. See `docs/13_Deployment.md`.

```bash
git submodule update --init --recursive     # once per clone: share/doxygen + cmake/Modules
make docs        # needs cmake + doxygen + graphviz
```

Three things about that build that are not obvious:

- **Doxygen 1.18.0 crashes** (a bus error in its comment scanner, about one run
  in four on any Markdown page). Use 1.17 or 1.12:
  `cmake -S . -B var/build -DDOXYGEN_EXECUTABLE=/path/to/doxygen`.
- **The page layout is `cmake/DoxygenLayout.xml`**, not the theme's. The theme's
  file lists the navigation only; from Doxygen 1.13 on, a section the layout
  does not list is not rendered, and every class page came out empty. The
  top-level `CMakeLists.txt` appends `LAYOUT_FILE` to the generated Doxyfile.
- **Search has two halves.** Doxygen's box finds names. `tools/site_search.py`,
  run by the `sitesearch` target after Doxygen, adds the text of every page: it
  writes `search/fulltext-index.js` and appends its script to `search/search.js`.
  It changes no template, so the theme submodule stays as it is.
  `tests/test_site_search.py` covers the index builder.
- **The side navigation is grouped in `share/site/navigation.dox`**
  (Applications, User guide, API access, C++, For developers). A new guide page
  is listed there, or it appears alone at the top level. The same file is why
  page names carry no local path: `STRIP_FROM_PATH` is appended to the Doxyfile.
- **One page per application, in `docs/apps/`.** Each says what the system is,
  the state and the measurement, how the model is built, how it steps in time
  (an integrator, a solver, or its own step), and a snippet that runs. Do not
  put two applications on one page. A new application gets a page, a row in
  `docs/17_Applications.md`, and a line in `navigation.dox`.
- **Animations are precomputed** by `tools/make_animations.py` and committed as
  GIFs beside the page that shows them (`docs/images`, `docs/apps/images`). A
  GIF plays on the repository page as well as on the site. The build does not
  run the script.
- **Versions.** Each build has its own folder; `tools/site_versions.py` lists
  them for the menu, orders them by version number and, in CI, restores the
  earlier ones and prunes old development builds (`make pages-restore`,
  `make pages-prune`). docs/13 has the details.
- **The notebooks are pages of the site** (`notebooks/`, linked from the menu),
  written by `tools/site_notebooks.py` in the same target. A notebook viewed on
  GitLab is shown without JavaScript, so its interactive players are blank
  there; on the site they run. The step needs `nbconvert` in the Python that
  CMake found, and is skipped with a notice otherwise:
  `cmake -S . -B var/build -DKALMAN_SITE_PYTHON=/path/to/env/bin/python`.

Doxygen's Markdown parser is stricter than a web renderer, and a malformed
construct silently swallows the rest of the page rather than erroring. Three
rules, all learned from real breakage here:

- **Keep a `[text](link)` on one line.** A link split across a newline
  terminates the page — every table after it collapses into one cell.
- **No `'single quotes'` inside `` `inline code` `` in a table cell.** They
  become typographic quotes and unbalance the `<tt>`, merging the rest of the
  table.
- **Never write a `---` rule directly after a blockquote**, and avoid
  `<angle-bracket>` placeholders in prose — Doxygen reads them as HTML tags and
  drops them.

Page filenames use hyphens (`9_Python-API.md`), not spaces: Doxygen cannot
resolve `%20`-encoded cross-references. `make docs` must end with no warnings
outside the two known docstring ones above.

---

## 10. libKalman — the C++ port

`include/Kalman/` + `src/` build `libKalman`: the same Model / Engine / Filter
contracts in C++, as a **standalone library with no required dependency**. See
[docs/15](docs/15_Cxx-Library.md) for using it; this section is about changing it.

**The dependency direction is the rule everything else follows.** kalman-framework
does not depend on ROOT — ROOT+ depends on *it*. A ROOT/ROOT+ component links
libKalman, never the reverse. So: no `TObject`, no `ClassDef`, no LinkDef, no
dictionary, no `TMatrixD`, no `TGraph`, no `TFile`. `Kalman::Matrix` and
`Kalman::Vector` exist precisely so the core owes nothing to anyone.

**C++ is the implementation; Python is the interface.** The library does not
link CPython and is not generated from anything. The Python classes forward to
it: a model written in Python is run by the C++ filter loop, which calls back
its `f` and `h` (`Kalman::Models::Callback`), and a controller function closes
the loop the same way (`kalman_filter_run_closed_loop`). The Python
implementations that remain are the reference the tests compare against
(`native.python_reference()`), on their way out. Do not add a new feature to a
Python core; add it in C++ and forward to it.

**`include/` and `src/` mirror `modules/framework/`**, file for file and
subdirectory for subdirectory — `Engines/`, `Models/`, `Filters/`, `IO/`,
`Abstract/`. A new file on one side has an obvious home on the other. The mapping
table is in docs/15; keep it accurate when you add to either side.

**No applications in the core.** libKalman ships no physics. A pendulum,
a drift chamber, a suspension or a power grid subclasses `Models::Linear` /
`Models::Extended` and lives at apps level, exactly as `modules/apps/` does for
Python. The models in `tests/cxx/Fixtures.h` are test fixtures and must stay
there.

**The applications are `libKalmanApps`** (`apps/`), a second library that links
the first. Each model in it is the twin of one Python class in `modules/apps/`,
which mixes in `NativeTwin` and hands its filtering pass to the twin when the
library is present. Rules for a twin:

- It must give the Python answer. `tests/test_native_twins.py` compares the two
  on every ported configuration; add the new one there and to
  `tools/benchmark_twins.py`.
- Nothing is defined twice. Scalars cross as named parameters; `Q`, `R`, the
  sample rate and the initial state are pushed from the Python instance. Do not
  recompute a covariance in C++ that Python already built.
- `native_name()` returns `None` for every configuration that was not ported
  (a Python callable, a state-dependent `Q`, a control). Falling back is
  correct; guessing is not.
- Derive from `Kalman::Apps::Model`, which differences `f`/`h` as
  `helpers.numerical_jacobian` does. Do not use the library's central scheme in
  a twin: the two sides would linearise differently.

**Adding a member to `Kalman::Model` changes its layout.** Everything compiled
against the header must be rebuilt together: the library, `libKalmanApps`, the
example plugin, the tests, and downstream users such as ROOT+. A stale plugin
does not fail to load; it corrupts memory.

**Scope**: every `EngineKind` has a C++ engine (linear, extended, unscented,
particle, ensemble, combinatorial), and the integrators, solvers, controllers
and routers are ported too. Still Python-only: training of the learned routers
and controllers (Torch), which cross as a finished policy through ONNX or
TorchScript. docs/15 has the table.

Conventions, most of which the build or the tests will enforce:

- **Everything is in `namespace Kalman`**, with `Engines::`, `Models::`,
  `Filters::`, `IO::` and `Abstract::` nested to match the directories. No `T`
  prefix: these are not ROOT classes, and the prefix would invite confusion with
  root-plus' own `ROOT::IOPlus::TKalmanFilter`, which is a separate and much
  smaller standalone filter. Both can live in one process.
- **Three libraries, and the core links no AI runtime.** `libKalman` (`include/`,
  `src/`) is the core; `libKalmanApps` (`apps/`) the application models;
  `libKalmanLearn` (`learn/`) the learned components. ONNX Runtime and libTorch
  are found and linked by `learn/CMakeLists.txt` only (`KALMAN_USE_ONNXRUNTIME`,
  on when found; `KALMAN_USE_TORCH`, on when found: the C++ libTorch, from `Torch_DIR`
  or the conda environment's `libtorch` package, never through Python's `torch`). Never add
  `find_package(Torch)` or `find_package(ONNXRuntime)` to the core, and never
  include a runtime header outside `learn/src/`. The core's one optional backend
  is BLAS (`KALMAN_USE_BLAS`). Everything is complete and correct with all of
  them off.
- **The core was `libKalmanFramework` up to 1.0.** The build installs that name
  as a link, and `native/core.py` still looks for it and reads the old
  environment variables. Keep both until a release note says otherwise.
- **Do not route `Kalman::Matrix` through Torch.** A Torch op has a ~230 ns
  dispatch floor — 10-25x an entire small matrix product. BLAS is used instead,
  and only at or above `KALMAN_BLAS_MIN_DIM`, because below the crossover the
  built-in kernel is measurably faster. Both numbers are measurements; re-measure
  before changing them.
- **Covariance inverses go through `Utils::Inverse`**, which tries Cholesky and
  then falls back to an eigenvalue pseudo-inverse. There is deliberately **no**
  Tikhonov ladder there: ridging a singular innovation covariance makes the
  Cholesky succeed and returns an inverse of order 1e12, i.e. a colossal Kalman
  gain conjured from a direction the measurement says nothing about. The jitter
  ladder belongs in `Utils::CholeskyFactor` (sampling), where it is harmless.
- **C++20.** Do not set `CMAKE_CXX_STANDARD_REQUIRED`:
  `cmake/Modules/StandardCompilerCXX.cmake` repurposes it as a *minimum standard
  number*, so the usual `ON` is read as "require C++23" and configuration fails.
- **The C++ tests are `tests/cxx/*.cc`**, plain executables returning a failure
  count, run by `make ctests`. `make tests` stays the Python suite. They assert
  on filter *consistency* (NIS ~ m, NEES ~ n), not on golden numbers, so they
  catch a mis-scaled covariance rather than just a changed one.
- **The routing layer crosses the language boundary as a trained policy, not a
  binding.** Training stays in Python (`routers/nn/` is Torch and stays
  Python-only); `export_onnx()` / `export_torchscript()` write one
  self-contained artifact with the feature standardisation and the softmax head
  baked in. Do not add a second file carrying the normalisation: shipped apart
  from its weights it still yields a valid-looking probability vector. Keep
  `export_onnx`'s `ir_version=9` default — Torch emits IR 10 and ONNX Runtime
  1.16 rejects it outright. `tests/test_router_export.py` pins the equivalence,
  currently 2e-13 (ONNX) and 1.8e-13 (TorchScript).
- **Keep the router/model split, and keep the runtimes out of the router.**
  `Kalman::Routers::Neural` owns features and normalisation and holds a
  `NN::Model`; the backends own the forward pass. ONNX Runtime is reached only
  from `learn/src/Routers/NN/Onnx.cc` and libTorch only from `Torch.cc` beside it and
  `learn/src/Solvers/Vorticity.cc` (the batched field solver behind Python's
  `SpectralSolver(backend="torch")`, through `Kalman/Learn/CAPI.h` and `native/learn.py`), all behind a pImpl.
  `backend="torch"` means that C++ solver only: do not add a Python-`torch` implementation beside it.
  That is what makes them optional in fact and not just in name: `NN::MLP`
  needs nothing, so the routing layer works in a build with no optional
  dependency at all. Never `#include <onnxruntime...>` or `<torch/...>` in
  `Routers/Neural.*` or any public header.
- **A wrong-looking IKF is usually a wrong-looking *test*.** Two modes that
  differ only in Q, under a measurement noise that swamps both, genuinely cannot
  be told apart — measured: innovation covariances 8% apart and both with mean
  NIS ~ 1.09, at which point ~0.5/0.5 is the *correct* answer, not a bug. Before
  suspecting the filter, check that the regimes are distinguishable in the data
  at all.

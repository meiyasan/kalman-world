# Troubleshooting

Real error messages from this codebase, and what they actually mean.

---

## Import and packaging

**`ModuleNotFoundError: No module named 'kalman'`**
The package is not installed. `pip install -e .` from the repository root, or
run with `PYTHONPATH=.`. On disk it is `modules/`; installed it is `kalman`.

**`ModuleNotFoundError: No module named 'modules'`**
You imported the on-disk path from outside the repository root. Import
`kalman.…` instead — that is the public surface.

**`The MCP server needs the MCP SDK` / `The REST API needs FastAPI + uvicorn`**
An optional extra is missing. `pip install "kalman[mcp]"` / `"kalman[rest]"`, or
`make setup` for all of them. These entry points exit 0 by design rather than
crashing.

---

## Model construction

**`AssertionError: You must provide fs or sample_rate`**
The model needs a sampling rate. Through the API, note that a top-level `fs` in
a `KalmanSpec` is only forwarded when the class `__init__` names `fs`
explicitly. For a class taking `**kwargs` (e.g. `BallisticLinearModel`), put it
in the params instead:

```python
{"base": "BallisticLinearModel", "dynamics": {"params": {"fs": 50.0}}}
```

**`ValueError: Pass either F= or A=, not both.`**
The matrices and continuous construction paths are mutually exclusive, as are
`Q=` and `Qc=`. Pick one — see [Models](4_Models.md).

**`A= needs fs= (or sample_rate=) to discretize via expm(A/fs).`**
The continuous path discretizes at `dt = 1/fs`, so the rate is required.

**`Qc= requires A=`**
Van Loan discretization needs the continuous generator.

**Shape errors during `super().__init__()` in a custom model**
Construction runs `_check_shapes`, which evaluates `F(zeros)` and hence `f` at
`x = 0`. Set `self.n` (and anything `f` touches) **before** calling
`super().__init__()`.

**`TypeError: … disallows engine kind …`**
The filter you chose is not in the model's `ALLOWED_ENGINES`. Either use a
permitted filter or widen the model's declaration.

---

## Integrators and solvers

**`TypeError: integrator KIND not in ALLOWED_INTEGRATORS (['RK4', …])`**
The model restricts which *shipped* kinds it admits. Either choose a permitted
one or widen `ALLOWED_INTEGRATORS`. A **custom** class or instance is exempt from
this check — it is validated against `KalmanIntegrator`/`KalmanSolver` instead —
so if you are seeing this for your own scheme, you passed an enum member rather
than the class. See [Integrators and Solvers](6_Integrators-and-Solvers.md).

**`ValueError: unknown integrator kind 'huen'; choose from [...] or register_integrator(...)`**
A typo, or a scheme you have not registered. `register_integrator("huen", cls)`
first, or fix the name. Unknown kinds raise rather than falling back to the
default.

**`ValueError: 'rk4' is a shipped kind; pass KalmanIntegratorKind.RK4 to rebind it`**
`register_*` will not shadow a shipped kind by name. If you really mean to
replace it process-wide, pass the enum member.

**`TypeError: … has no explicit tableau; the spectral solver needs an explicit
Runge–Kutta method.`**
Field solvers run the stages themselves from `integrator.tableau`. An integrator
with `tableau = None` (implicit or adaptive-only) cannot drive one.

**`NotImplementedError: IMPLICIT_EULER is reserved` / `… solver is reserved`**
Reserved registry entries. Use `rk4`/`dormand_prince`, or
`spectral`/`finite_difference`.

**Changing the integrator made no difference.**
Expected on a linear model. The linear engine and the native C fast path use
`F()` and never call `f()`. The integrator applies when the model declares
`DYNAMICS = CONTINUOUS`, or when a solver drives a field.

**A mistyped kind used to run anyway.**
No longer: `resolve_integrator`, `resolve_solver`, `resolve_engine` and
`resolve_provider` all raise `ValueError` on an unrecognised kind. `None` still
selects the default.

---

## Filtering results

**NIS far above `m`** — the filter is overconfident: `R` and/or `Q` are too
small. **Far below `m`** — they are too large and the data is being ignored.
**Structured, non-white innovations** — the model is wrong in a way tuning will
not fix. See [Filters and Diagnostics](5_Filters-and-Diagnostics.md).

**`NotImplementedError: EnsembleKalmanFilter has no RTS smoother`**
EnKF and PF are forward-only. Use `forward()`.

**Comparing filter variants gives inconsistent results run to run.**
Give each variant its **own** RNG stream (`seed=`) and share only the
observation stream. Shared streams cross-contaminate — this was a real bug here
once.

**Reconstructed signal decays at the tail / spurious lines at `fc ± fs_dec`**
Resampling artifacts in the rotating-frame oscillator models, both fixed. The
mechanism (edge replication, whole-`down`-block padding, Kaiser β = 12) is
documented in [`../README.md`](../README.md) — and window your FFT before
concluding anything about out-of-band leakage.

---

## API and transports

**`ValueError: Cannot plot measurement error: no predicted measurement available`**
The default figure's error panel needs a truth trajectory. Pass one
(`model.plot(record, sample=truth)`) or drop the panel
(`model.plot(record, error=False)`). `render_plot` / `GET /plot/{run_id}` do the
latter automatically, since a stored run has no separate truth.

**`404` from `/diagnostics/{run_id}` or `/plot/{run_id}` right after a
successful run**
The run store is per-process. Under multiple uvicorn workers the ID belongs to a
different worker. Run one worker.

**`ValueError: unknown model base '…'; see catalog()['models']`**
The `base` is neither `matrices`/`continuous` nor a catalog class name. Specs
are pure data — arbitrary code is rejected by design.

---

## Authentication

**`AuthError: a bearer token with scope 'read' is required.`**
`KALMAN_JWT_SECRET` is set, so auth is on, and no token was presented. `401`
over REST.

**`AuthError: token lacks required scope 'write_models'.`**
Valid token, wrong scopes. Mint a new one — and remember `send_measurement` does
not imply `read`. `403` over REST.

**`InvalidToken: invalid or expired token`**
Wrong secret, tampered token, or past `exp`. `401` over REST. Log in again
(`POST /token`) or mint a new token.

**`InvalidToken: this token was revoked.`**
Someone revoked the token, or all tokens of its user (`POST /token/revoke`,
`make api-revoke`). Log in again: a token issued after the revocation works.

**`429 too many failed logins`**
5 failed logins in 5 minutes for this user from this address. Wait for the
time in `Retry-After`, or ask an admin to run `make api-unlock USER=…`. A
restart of the server does not clear the count: it is in the shared registry.

**`AuthError: auth is enabled but PyJWT is not installed`**
Deliberately fail-closed. `pip install "kalman[auth]"`.

**`AuthError: account 'NAME' uses a non-bcrypt hash`**
An htpasswd entry written by another tool with a different algorithm. Re-add it:
`make api-user USER=NAME`.

**`PermissionError: missing outbound credential 'NAME'`**
A *provider* credential, not a caller token. Set `$KALMAN_NAME_KEY` or add it
to `~/.kalman/credentials.json`. See [Providers](7_Providers.md).

---

## Build and test

**`make docs` / `make pages` fails with `share/doxygen is empty` or `cmake/Modules is empty`**
The theme and the shared CMake helpers are submodules. Fetch both:

```bash
$ git submodule update --init --recursive
```

`bootstrap.sh` does this for you when it can reach the remote; a clone made with
`--recurse-submodules` never hits it.

**`cmake is required to configure the documentation build`**
The docs build is CMake-driven (`bootstrap.sh` → `var/build`).
`apt-get install cmake` / `brew install cmake`. Doxygen and Graphviz are needed
too; both are in the CI image.

**Doxygen warns `unable to resolve reference to '…%20….md'`**
A cross-page link was written URL-encoded. Doxygen wants the literal filename —
which is why the pages under `docs/` are named with hyphens rather than spaces.

**Test suite is red**
The expected baseline is **453 passed, 2 skipped**; the skips are
optional-dependency gates. `python -m pytest tests/test_structure.py -q` alone
tells you whether an architectural invariant broke (a base class outside
`framework/`, a registry not covering its enum, a flat `apps/*.py` module).

**Native acceleration not used**
`make accel` builds it; failure is non-fatal and the pure-Python fallback gives
identical results. `tests/bench_native.py`, run at the end of `make tests`,
reports which path is live.

**Torch errors on Apple Silicon about float64**
MPS has no float64. Use float32 on `mps` and float64 elsewhere.

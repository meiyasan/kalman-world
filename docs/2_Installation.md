# Installation

The core is pure Python + NumPy/SciPy/Matplotlib. Everything else — the MCP
server, the REST API, authentication, remote providers, GPU solvers — is an
**optional extra**, lazily imported and guarded. The framework imports and the
test suite passes without any of them.

---

## One command

```bash
$ make setup
```

`make setup` installs the core requirements, then the optional extras, then
tries to build the C acceleration, then self-checks and prints what is live:

```
==> python: .venv/bin/python
==> core dependencies (requirements.txt)
==> optional extras (mcp, rest, auth, remote)
==> C acceleration (optional)
    built
==> self-check
    46 models, 6 engines, 4 integrators, 4 solvers, 6 providers
    MCP server   ready
    REST API     ready
    scoped auth  ready
    GPU/ML       ready
```

Anything reported as `not installed` is an optional extra you can add later;
nothing in the core stops working without it.

The Makefile picks its interpreter in this order: `.venv/bin/python`, then a
conda env named `$ENVNAME`, then `python3`. Override it explicitly:

```bash
$ make setup PYTHON=/usr/local/bin/python3.11
```

---

## Manual install

```bash
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
$ pip install -e .                     # installs `modules/` as the `kalman` package
```

### Optional extras

| Extra | Install | Unlocks |
|-------|---------|---------|
| `mcp` | `pip install "kalman[mcp]"` | the MCP server (`python -m kalman.mcp`) |
| `rest` | `pip install "kalman[rest]"` | FastAPI + uvicorn REST API (`python -m kalman.rest`) |
| `auth` | `pip install "kalman[auth]"` | bcrypt accounts + scoped JWT bearer tokens |
| `remote` | `pip install "kalman[remote]"` | `RemoteProvider` (httpx) |
| `test` | `pip install "kalman[test]"` | pytest, coverage, ruff |
| `all` | `pip install "kalman[all]"` | all of the above |

Extras not installed are never fatal. `python -m kalman.mcp` without the SDK
prints an install hint and exits 0 rather than crashing.

Deeper optional dependencies are not packaged as extras and are installed on
demand: `torch` (GPU solver backends, neural controllers/routers),
`cdsapi`/`xarray`/`h5py` (ERA5 and turbulence providers), `motmetrics`
(MOT tracking evaluation).

### Conda

```bash
$ make env          # builds from environment.yml
$ make conda
```

`environment.x86_64.yml` is the pinned x86 variant used by CI.

---

## Native acceleration (optional)

The Python engines evaluate the *model* in Python and hand the *step* — the
covariance prediction, the Joseph update, a whole linear pass, the RTS sweep,
the sigma-point moments, particle weights, NIS/NEES — to `libKalmanFramework`,
the same C++ classes a ROOT+ program runs (see [C++ Library](15_Cxx-Library.md)).

```bash
$ make accel        # = make build install: builds libKalmanFramework into var/install
```

Without the library (no cmake, or not built) the framework silently uses its
pure-Python path — identical results, slower. `make tests` ends with
`tests/bench_native.py`, which reports which path is actually in use.

---

## Verify the install

```bash
$ make tests                                   # full suite + coverage + native bench
$ python -m pytest tests/test_structure.py -q  # architecture invariants only
$ make linter                                  # ruff, correctness-focused
```

The expected baseline is **453 passed, 2 skipped** — the skips are
optional-dependency gates. A red suite is never "expected"; see
[Troubleshooting](14_Troubleshooting.md).

A five-second smoke check that the package resolves:

```bash
$ python -c "from kalman.api import catalog; print(len(catalog()['models']), 'models')"
46 models
```

---

## Docker

```bash
$ make image        # build
$ make serve        # docker compose up -d rest mcp
```

This brings up the REST API on **port 8000** and the MCP server (SSE transport)
on **port 8765**. Full configuration, environment variables and hardening notes
are in [Deployment](13_Deployment.md).

---

## Registering the MCP server with your AI client

```bash
$ make mcp-clients   # which MCP clients are detected on this machine
$ make mcp           # register 'Kalman-AI' with all of them, then verify
```

See [MCP Server](11_MCP-Server.md) for per-client detail, dry runs, tokens and
removal.

---

## Layout after install

On disk the package lives in `modules/`; installed, it is imported as `kalman`
(`[tool.setuptools] package-dir = {"kalman" = "modules"}`). So
`modules/apps/weather/flow.py` is `kalman.apps.weather.flow`. Always import the
`kalman.…` path — `modules.…` only works from the repository root and is not the
public surface.

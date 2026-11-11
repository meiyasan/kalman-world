# kalman-framework documentation

A modular Python framework for Kalman-family estimation: composable contracts in
`kalman.framework`, domain applications in `kalman.apps`, and one JSON facade
(`kalman.api`) exposed over REST and MCP so humans, scripts and agents all drive
the same code.

Pages are numbered in reading order. If you are new, read 1 → 3 → 5 and stop.

## For users

| # | Page | What it answers |
|---|------|-----------------|
| 1 | [Getting Started](1_Getting-Started.md) | Filter a signal in ten minutes |
| 2 | [Installation](2_Installation.md) | Environments, extras, native acceleration, Docker |
| 3 | [Overview](3_Overview.md) | Architecture: the seven roles and how they compose |
| 4 | [Models](4_Models.md) | The three ways to define a system; the model catalog |
| 5 | [Filters and Diagnostics](5_Filters-and-Diagnostics.md) | KF/EKF/UKF/PF/EnKF/CKF/IKF — which one, and is it healthy |
| 6 | [Integrators and Solvers](6_Integrators-and-Solvers.md) | Time stepping, spatial discretization, GPU backends, customization |
| 7 | [Providers](7_Providers.md) | Where the data comes from: synthetic, file, sensor, remote, ERA5 |
| 8 | [Control and Routing](8_Control-and-Routing.md) | Closed-loop controllers and IKF model routing |

## For API access

| # | Page | What it answers |
|---|------|-----------------|
| 9 | [Python API](9_Python-API.md) | The `kalman.api` facade and the `KalmanSpec` dict |
| 10 | [REST API](10_REST-API.md) | HTTP endpoints, ports, `curl`, OpenAPI, Unity/C# client |
| 11 | [MCP Server](11_MCP-Server.md) | Tools, one-command client registration |
| 12 | [Authentication](12_Authentication.md) | Accounts, scoped JWTs, what each scope unlocks |
| 13 | [Deployment](13_Deployment.md) | Docker Compose, environment variables, ports, hardening |
| 14 | [Troubleshooting](14_Troubleshooting.md) | Error messages and what they actually mean |

## For C++ / ROOT

| # | Page | What it answers |
|---|------|-----------------|
| 15 | [C++ Library](15_Cxx-Library.md) | `libKalmanFramework`: the core ported to C++, loadable in a ROOT+ analysis |
| 16 | [C++ Models from Python](16_Cxx-Models-from-Python.md) | Write a model once in C++, use it from both languages; a worked toy example |

## Also in this repository

- [`../README.md`](../README.md) — notation, filter pipeline, composite models.
- [`../AGENTS.md`](../AGENTS.md) — conventions for changing the code (humans and LLMs).
- [`../share/notebooks/`](../share/notebooks/) — 16 executed, output-bearing notebooks,
  from `00_Kalman101` to `14_StormTracking`. These are the narrative documentation;
  these pages are the reference. Trained artefacts a notebook would otherwise
  rebuild on every run (the neural controllers and routers of `12_KalmanAI`) are
  kept under `share/notebooks/cache/<notebook>/` by `kalman.helpers.cached_fit`;
  delete a file, or set `KALMAN_CACHE_REFRESH=1`, to retrain.

## Conventions used in these pages

- On disk the package is `modules/`; installed, it is **`kalman`**. Always import
  `kalman.…` (`package-dir = {"kalman" = "modules"}` in `pyproject.toml`).
- `$` marks a shell command; `>>>`-free code blocks are copy-pasteable Python.
- Every example in these pages was executed against the current tree.

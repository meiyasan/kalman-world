# libKalmanNet: the JSON API, REST and MCP servers in C++

`libKalmanNet` (`net/`) is the fourth C++ library. It answers the verbs of the
Python facade `kalman.api` ([Python API](9_Python-API.md)) for the models that
live in C++, and serves them over HTTP ([REST API](10_REST-API.md)) and over
the Model Context Protocol ([MCP Server](11_MCP-Server.md)) with the same
routes, tool names, payloads and status codes as the Python servers. A client
written for `python -m kalman.rest` or `python -m kalman.mcp` talks to
`kalman-serve` unchanged, for the verbs it serves.

It is decoupled from the core: `libKalman` gains no dependency, and nothing in
the build needs `libKalmanNet`.

| Class | What it is |
|---|---|
| `Kalman::Net::Api` | the verbs: a verb name, its arguments as a JSON object, a JSON answer |
| `Kalman::Net::RestServer` | the routes of `modules/rest/app.py`, on cpp-httplib |
| `Kalman::Net::McpServer` | the tools of `modules/mcp/server.py`, JSON-RPC 2.0 over stdio or `POST /mcp` |
| `Kalman::Net::Auth` | verifies the scoped JWTs that `kalman.api.auth` issues |
| `kalman-serve` | the program: `--rest --port N` or `--mcp` |

## What it serves, and what stays Python

It serves every model `Kalman::Factory` knows: `libKalmanApps` (loaded at start
when found) and every plugin loaded since, by their registered names
(`{"base": "pendulum"}`); and the specs whose base is `matrices` or
`continuous`, given inline or authored in the shared registry.

A model written in Python lives in the Python process: its `f` and `h` are
Python code. The C++ server cannot run it and says so (`400`, "unknown model
base ... A model written in Python is served by the Python server"). So does
every verb that needs a Python library, with `501` and the name of the Python
server:

| Verb (route) | Why it stays Python |
|---|---|
| `render_plot` (`GET /plot/{run_id}`) | matplotlib draws the figures |
| `run_osse` (`POST /osse`) | the twin experiment is `kalman.apps.weather` |
| `run_control` (`POST /control`) | the controllers' training (LQR fit, neural) is Python |
| `fetch`, `providers`, `describe_provider` (`POST /fetch`, `GET /providers`) | the providers (ERA5, files, sensors, remote APIs) are Python |
| `register_model`, `delete_model`, `register_source`, `delete_source`, `push_measurements` | the registry's writers validate with the Python framework and own the events |
| `save_spec`, `load_spec`, `scaffold` (`POST /scaffold`) | authoring reads and writes Python files |
| `login`, `revoke`, `unlock` (`POST /token...`) | accounts are bcrypt in the htpasswd file; the lockout is Python's |
| `GET /events/stream` | Server-Sent Events; poll `GET /events` instead |

Both servers share one registry (`$KALMAN_STATE_DIR/registry.sqlite`), so a
model registered through Python is a spec base here, a plugin loaded here is
in Python's `plugins()` and both see the same events.

## Build

`make build` builds it when its dependencies are found; they are private to the
library and come from the system or the conda environment:

```bash
conda install -n kalman-framework -c conda-forge nlohmann_json cpp-httplib openssl
```

| Dependency | For | Required |
|---|---|---|
| nlohmann_json | JSON | yes |
| cpp-httplib | the HTTP server | yes |
| OpenSSL (libcrypto) | HMAC-SHA256 of the JWTs | yes |
| SQLite3 | the shared registry: authored models, plugins, sources, events, revoked tokens | no |

Without one of the first three the library is skipped with a notice and the
rest of the build is unchanged. Without SQLite the registry reads as empty and,
with auth on, every token is refused: the revocation list cannot be read, and
a check that cannot be made fails closed, as in Python. `-DKALMAN_BUILD_NET=OFF`
switches the library off.

From another CMake project:

```cmake
find_package(Kalman REQUIRED COMPONENTS Net)
target_link_libraries(my_server PRIVATE Kalman::Net)
```

The public headers (`<Kalman/Net/Net.h>`) use the standard library only.

## Start it

```bash
var/install/bin/kalman-serve --rest --port 8000 --plugin KalmanPlugin_example
var/install/bin/kalman-serve --mcp                     # stdio, one JSON-RPC message per line
```

| Option | Effect |
|---|---|
| `--rest [--host H] [--port N]` | the REST API, default `127.0.0.1:8000`; `--port 0` takes a free port and prints it |
| `--mcp` | the MCP server on stdio |
| `--plugin NAME` | load a plugin at start, from the allowed directories (repeatable) |
| `--no-apps` | do not load `libKalmanApps` |
| `--registry FILE` | the registry file |

The environment is the Python servers': `KALMAN_JWT_SECRET` (auth on),
`KALMAN_JWT_ISSUER`, `KALMAN_JWT_AUDIENCE`, `KALMAN_STATE_DIR`,
`KALMAN_PLUGIN_PATH`, `KALMAN_MCP_TOKEN` (stdio), `KALMAN_MAX_RUNS`,
`KALMAN_ALLOW_ANONYMOUS`, `KALMAN_CORS_ORIGINS`. As in Python, the server
refuses to listen on anything but loopback without auth (unless
`KALMAN_ALLOW_ANONYMOUS=1`) and refuses a secret shorter than 32 bytes.

Register it with an MCP client like the Python one, with the binary as the
command:

```json
{"mcpServers": {"Kalman-AI": {"command": "/path/to/var/install/bin/kalman-serve",
                              "args": ["--mcp", "--plugin", "KalmanPlugin_example"]}}}
```

## Routes and tools

| Verb | Scope | REST | MCP tool |
|---|---|---|---|
| `catalog` | read | `GET /catalog` | `list_models` |
| `describe` | read | `GET /describe/{name}` | `describe_model` |
| `models` | read | `GET /models` | `list_registered_models` |
| `sources`, `describe_source` | read | `GET /sources`, `GET /sources/{name}` | `list_sources` |
| `events` | read | `GET /events?since=&limit=`, `GET /events/cursor` (no token) | `list_events` |
| `plugins` | read | `GET /plugins` | |
| `diagnostics` | read | `GET /diagnostics/{run_id}` | `get_diagnostics` |
| `whoami` | (a valid token) | `GET /whoami` | |
| `run_filter` | send_measurement | `POST /filter` `{spec, data \| source, query?, smooth?}` | `run_filter` |
| `forecast` | send_measurement | `POST /forecast` `{spec, x0, P0?, steps?}` | `forecast` |
| `sample` | send_measurement | `POST /sample` `{spec, duration, seed?, process_noise?, measurement_noise?}` | `sample` |
| `load_plugin` | load_plugins | `POST /plugins` `{name}` | `load_plugin` |
| | | `GET /healthz` | |
| | | `POST /mcp` (MCP over HTTP, JSON-RPC) | |

`POST /sample` and the `sample` tool exist on this server only (the Python
facade has the verb but no route or tool). `diagnostics` answers what Python
answers (`nis`, `innovation`, `innovation_covariance`) plus `mean_nis`, and for a
sampled run `nees` and `mean_nees` against its truth. `run_filter` with
`source=` reads a push source of the registry; other sources are providers,
which run in Python (`501`).

The MCP server implements `initialize`, `ping`, `tools/list`, `tools/call`,
`resources/list` and `resources/read` (`catalog://models`, `catalog://sources`,
`registry://events`). A tool result carries the answer as `structuredContent`
and as JSON text; a refusal is a result with `isError: true`. The token is,
first match wins, the tool's `token` argument, the `Authorization: Bearer`
header (`POST /mcp`), then `KALMAN_MCP_TOKEN` on stdio.

## Tokens

`Kalman::Net::Auth` verifies the tokens of [Authentication](12_Authentication.md):
HS256 with `KALMAN_JWT_SECRET` (OpenSSL HMAC, compared in constant time),
`exp` and `sub` required, `nbf`/`iat` not in the future, `iss` and `aud` when
configured (and a token naming an audience refused by a server that expects
none, PyJWT's rule), then the revocation list in the shared registry. A token
minted with `python -m kalman.api.auth token` or `POST /token` on the Python
server is accepted here, and a token revoked there is refused here from the
next request. Then the scope: `401` for a missing or bad token, `403` for a
good one without the scope (`admin` covers all).

Issuing tokens stays in Python: accounts are bcrypt hashes in the htpasswd file,
logins are rate-limited in the registry, and none of that is needed to verify.

## From Python, in process

`modules/framework/native/net.py` loads `libKalmanNet` next to the core (one
model factory for both) and calls a verb without a socket:

```python
from kalman.framework.native import net
net.call("describe", {"name": "pendulum"})
net.call("run_filter", {"spec": {"base": "pendulum", "fs": 50.0}, "data": zk})
```

A refusal raises `net.NetError` with the HTTP status in `.status`.

## Agreement with Python, and speed

`tests/test_net_server.py` starts `kalman-serve` and compares it with
`kalman.api` on the same request, on the example plugin's `constant-velocity`
and `pendulum` and on a `matrices` spec. `run_filter`, `diagnostics`, `forecast`
and a noise-free `sample` agree exactly (largest difference 0): the Python
engines run the same C++ kernels.

One exception, found here: smoothing a nonlinear model. The Python RTS pass
gains with `transition[k]` (the Jacobian at the posterior of step k-1),
`Kalman::Filter::Backward` with `transition[k+1]` (at the posterior of step k,
the RTS gain); on the pendulum the smoothed states differ by 1.7e-4. Linear
models are unaffected (one F for every step).

`run_filter`, 2000 steps (Apple Silicon, warm):

| Model | Python facade | `kalman-serve` REST round trip | in process (`net.call`) |
|---|---|---|---|
| `pendulum` (EKF, plugin) | 66.8 ms | 10.6 ms | 11.5 ms |
| `matrices` spec (KF) | 32.0 ms | 8.6 ms | 9.3 ms |
| `constant-velocity` (KF, plugin) | 12.3 ms | 8.8 ms | 9.1 ms |

Most of the C++ time is JSON: 2000 covariances are written and parsed as text.
At 200 steps the REST call costs 1.6-1.8 ms against 4-8 ms in Python; `catalog`
and `describe` take about 1 ms on either server.

## Rules for changing it

See `AGENTS.md`, section 10: the dependencies stay private to `net/`, the core
gains none, and a verb keeps the name, arguments and answer of its Python
namesake, checked by `tests/test_net_server.py`.

# Providers

A provider answers **WHERE the data comes from**. It is the sibling of
`kalman.apps`: apps describe systems, providers feed them.

Every provider returns a plain dict payload — some of
`{"zk", "xk", "truth", "time", "meta"}` — so any transport can carry it as JSON,
and the arrays drop straight into `run_filter` / `run_osse` with no adaptation.

---

## The contract

```python
class KalmanProvider:
    KIND = None                       # a KalmanProviderKind member

    def fetch(self, **query): ...     # pull a dataset  → payload dict
    def stream(self, **query): ...    # yield live items (streaming sources)

    def measurements(self, **query): ...   # convenience: the zk array
    def truth(self, **query): ...          # convenience: the truth trajectory
```

The base offers a small query-keyed cache, so repeating a query costs nothing.
Providers that are not streaming raise a clear `NotImplementedError` from
`stream()`, and vice versa.

## Shipped providers

| Kind | Class | Source |
|---|---|---|
| `synthetic` | `SyntheticProvider` | generated from a model or a `KalmanSpec`, via `KalmanSampler` |
| `file` | `FileProvider` | `.npz` / `.npy` / `.json` / `.csv` on disk |
| `sensor` | `SensorProvider` | live iterable, generator, or zero-arg callback |
| `remote` | `RemoteProvider` | HTTP endpoint (cached, rate-limited, authenticated) |
| `era5` | `ERA5Provider` | ERA5 reanalysis fields (needs `cdsapi`/`xarray`) |
| `turbulence` | `TurbulenceProvider` | Kraichnan / NSKT turbulence datasets |

The two weather providers live with their domain
(`kalman.apps.weather.providers`) and are registered into `kalman.providers`
alongside the generic ones.

## Resolving one

```python
from kalman.providers import resolve_provider

prov = resolve_provider("file", path="run.npz")
zk = prov.measurements()
```

`resolve_provider` raises `ValueError` on an unknown kind — it does not fall
back silently the way the integrator/solver resolvers do.

---

## By kind

### Synthetic

```python
from kalman.providers import SyntheticProvider

prov = SyntheticProvider(model)                    # or SyntheticProvider(spec=my_spec)
payload = prov.fetch(duration=10.0)                # {"zk", "xk", "time", "meta"}
```

The bridge between "I have a model" and "I have data" — the same sampler that
backs `model.sample()` and `kalman.api.sample()`.

### File

```python
from kalman.providers import FileProvider

FileProvider.save("run.npz", zk, xk=xk, time=t)    # write
prov = FileProvider("run.npz")
zk = prov.measurements()
```

- `.npz` — key `zk` required; `xk` / `time` optional.
- `.npy` — a single `(T, m)` measurement array.
- `.json` — `{"zk": [...], "xk": [...]?, "time": [...]?}`.
- `.csv` — a numeric table, taken as `zk`.

### Sensor (streaming)

```python
from kalman.providers import SensorProvider

prov = SensorProvider(my_generator, max_items=1000)
for row in prov.stream():
    ...
```

The source may be any iterable, a generator, or a zero-arg callback returning a
row (or `None` to stop). Compose it with the filter's streaming append
(`forward(reset=False, buffer=…)`) to run a filter live.

### Remote

```python
from kalman.providers import RemoteProvider

prov = RemoteProvider("https://data.example.org/api",
                      credential_name="mylab",     # outbound secret, resolved server-side
                      min_interval=1.0,            # rate limit, seconds
                      field="zk")
payload = prov.fetch("series/42", params={"since": "2026-01-01"})
```

Responses are cached, a minimum inter-request interval is enforced, and the
bearer credential is resolved **server-side** — never from a caller's token.
Needs `httpx` (`pip install "kalman[remote]"`).

### ERA5 and turbulence

```python
from kalman.providers import ERA5Provider, TurbulenceProvider

ERA5Provider().fetch(model, dates=[...], area=[...], level=500)
TurbulenceProvider().fetch(model, variable="vorticity", stride=1)
```

These need their own heavy optional dependencies (`cdsapi`, `xarray`, `h5py`)
and, for ERA5, a CDS API key — see credentials below.

---

## Credentials

Two credential systems exist and must not be confused:

| | Outbound | Inbound |
|---|---|---|
| Module | `kalman.providers.credentials` | `kalman.api.auth` |
| Purpose | *our* secrets for reaching remote services (CDS key, lab API tokens) | *callers'* tokens for reaching us |
| Storage | `$KALMAN_<NAME>_KEY` or `~/.kalman/credentials.json` | bcrypt `.htpasswd` + signed JWTs |
| Page | this one | [Authentication](12_Authentication.md) |

```bash
$ export KALMAN_MYLAB_KEY="…"           # resolved as credential_name="mylab"
# or
$ cat ~/.kalman/credentials.json
{"mylab": "…"}
```

Resolution is **fail-closed**: a provider that needs a credential and cannot
find one raises `PermissionError` naming both the env var and the store path.
Secrets are returned to the calling code only and are never logged. Credential
files are gitignored and must stay that way.

---

## Through the API

`fetch` is exposed on the facade, and therefore over MCP and REST, gated by the
`send_measurement` scope. A raw `file`, `remote` or `sensor` provider also
needs `admin`: these kinds read server files or use the server's outbound
credentials. See [Authentication](12_Authentication.md).

```python
from kalman.api import fetch, run_filter

payload = fetch({"kind": "file", "init": {"path": "run.npz"}})
result  = run_filter(spec, payload["zk"])
```

```bash
$ curl -X POST localhost:8000/fetch -H 'Content-Type: application/json' \
       -d '{"provider": {"kind": "synthetic", "init": {}}, "query": {"duration": 5.0}}'
```

Note the shape: `{"kind": …, "init": {…}}` is the constructor, and `query` is
what goes to `fetch()`.

### Named sources

A source is a provider with a name, kept in the shared registry. You set it up
one time, and then all clients use it by name:

```python
from kalman.api import register_source, push_measurements, fetch, run_filter

register_source("lab-1", {"kind": "remote",
                          "init": {"base_url": "https://lab.example/api",
                                   "credential_name": "lab"}})       # admin
register_source("scene-1", {"kind": "push", "m": 1})                # write_models

push_measurements("scene-1", [[0.01], [0.02]], time=[0.0, 0.01])    # a device writes
fetch("scene-1", {"since": 0})                                      # anyone reads
run_filter({"base": "demo_cv"}, source="lab-1")                     # measure + filter
```

A `push` source is a buffer in the registry, not a provider class. Devices,
simulations and game engines send rows into it over REST or MCP. Each push
gives subscribers a `measurement.received` event. The query of a named source
can only narrow it. It cannot change the `path` of a file source or the
`base_url` of a remote source. The REST routes are in
[REST API](10_REST-API.md#measurement-sources).

## Writing your own

Subclass `KalmanProvider`, tag a `KIND`, implement `fetch` and/or `stream`, and
register it in `PROVIDER_BY_KIND`. Domain-specific providers belong with their
domain (`apps/<domain>/providers/`) and are registered from
`kalman/providers/__init__.py` — that is how the weather ones work.

The same closed-enum limitation described in
[Integrators and Solvers](6_Integrators-and-Solvers.md) applies: a new kind
means adding a `KalmanProviderKind` member in-tree.

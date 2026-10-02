# REST API

A FastAPI app over the [`kalman.api` facade](9_Python-API.md) — one thin
adapter, no logic of its own. Every route declares a required **scope**, checked
by the same `authorize()` the MCP server uses.

```bash
$ pip install "kalman[rest]"
$ make rest                       # python -m kalman.rest
```

---

## Ports and addresses

| Service | Default port | Environment variables |
|---|---|---|
| **REST API** | **8000** | `KALMAN_REST_HOST` (default `127.0.0.1`), `KALMAN_REST_PORT` |
| MCP server (SSE) | 8765 | `KALMAN_MCP_HOST` (default `127.0.0.1`), `KALMAN_MCP_PORT`, `KALMAN_MCP_TRANSPORT` |

On any address other than loopback, the server does not start without
`KALMAN_JWT_SECRET` (at least 32 bytes). Set `KALMAN_ALLOW_ANONYMOUS=1` to
override this, but only on a network you trust. See
[Authentication](12_Authentication.md).

```bash
$ KALMAN_REST_PORT=9000 python -m kalman.rest
$ make serve                      # docker compose: REST on 8000, MCP on 8765
```

`docker-compose.yml` publishes `8000:8000` and `8765:8765`. Full deployment
detail is in [Deployment](13_Deployment.md).

### Self-documenting

| URL | What |
|---|---|
| `http://localhost:8000/docs` | interactive Swagger UI |
| `http://localhost:8000/openapi.json` | the OpenAPI schema — a machine-discoverable contract for any non-MCP agent |
| `http://localhost:8000/healthz` | `{"ok": true, "auth": false}`, unauthenticated; `auth` tells if tokens are required |

---

## Endpoints

| Method | Path | Scope | Body / params |
|---|---|---|---|
| `GET` | `/healthz` | none | — |
| `POST` | `/token` | none (user + password) | `{user, password, scopes?, exp_hours?}` → `{access_token, …}` |
| `POST` | `/token/revoke` | any valid token (`admin` for another user) | `{}` this token, or `{user}` all tokens of a user |
| `GET` | `/whoami` | any valid token | — returns `{auth, user, scopes, exp}` |
| `GET` | `/catalog` | `read` | — |
| `GET` | `/describe/{name}` | `read` | — |
| `GET` | `/diagnostics/{run_id}` | `read` | — |
| `GET` | `/plot/{run_id}` | `read` | returns `image/png` |
| `GET` | `/providers`, `/providers/{name}` | `read` | — |
| `GET` | `/models` | `read` | — the registry, with owners |
| `GET` | `/plugins` | `read` | — plugins recorded in the registry |
| `GET` | `/sources`, `/sources/{name}` | `read` | — |
| `GET` | `/events` | `read` | `?since=&limit=` |
| `GET` | `/events/stream` | `read` | `?since=&timeout=`, Server-Sent Events |
| `POST` | `/filter` | `send_measurement` | `{spec, data, smooth?}` or `{spec, source, query?}` |
| `POST` | `/forecast` | `send_measurement` | `{spec, x0, P0?, steps}` |
| `POST` | `/osse` | `send_measurement` | `{spec, options?}` |
| `POST` | `/control` | `send_measurement` | `{model, controller, duration}` |
| `POST` | `/fetch` | `send_measurement` | `{source, query?}` or `{provider, query?}` |
| `POST` | `/sources/{name}/measurements` | `send_measurement` | `{rows, time?}` |
| `GET` | `/sources/{name}/measurements` | `send_measurement` | `?since=&limit=` |
| `POST` | `/models` | `write_models` | `{spec, name?, replace?}` |
| `DELETE` | `/models/{name}` | `write_models` + owner | — |
| `POST` | `/sources` | `write_models` (`admin` for `file`/`remote`/`sensor`) | `{name, provider, replace?}` |
| `DELETE` | `/sources/{name}` | `write_models` + owner | — |
| `POST` | `/plugins` | `load_plugins` | `{name}` |
| `POST` | `/scaffold` | `write_models` | `{spec, class_name?}` |

Status codes:

| Code | When |
|---|---|
| `400` | the request is wrong: a missing field, a bad spec, rows of the wrong width |
| `401` | no token, or a forged, expired or revoked token (with `WWW-Authenticate: Bearer`); a wrong password at `/token` |
| `403` | a valid token without the scope, or a change to an entry the caller does not own |
| `404` | an unknown model, source or `run_id` |
| `409` | the name is already used by a different definition |
| `429` | too many failed logins (with `Retry-After`) |

---

## Examples

Authentication is off until you set `KALMAN_JWT_SECRET`, so a local development
server needs no header. Add `-H "Authorization: Bearer $TOKEN"` once it is on.

```bash
$ curl -s localhost:8000/healthz
{"ok":true,"auth":false}

$ curl -s localhost:8000/catalog | jq '.models | length'
43

$ curl -s localhost:8000/describe/StormModel | jq '{name, default_engine, params}'
```

### Filtering a measurement set

```bash
$ curl -s -X POST localhost:8000/filter \
    -H 'Content-Type: application/json' \
    -d '{
      "spec": {
        "base": "matrices", "fs": 100.0,
        "dynamics": {"F": [[1, 0.01], [0, 1]], "H": [[1, 0]]},
        "noise":    {"Q": [[1e-6, 0], [0, 1e-4]], "R": [[0.05]]},
        "initial":  {"x0": [[0], [1]]},
        "bindings": {"engine": "linear"}
      },
      "data": [[0.01], [0.02], [0.031], [0.039]]
    }'
{"run_id":"run-3f9c2a1b7d4e",
 "xk":[[0.01,1.0],[0.02,1.0],[0.030000082,1.00000012],[0.039999978,0.99999992]],
 "Pk":[[[2.0099e-06,9.9996e-07],[9.9996e-07,1.99999e-04]], …],
 "time":[…]}
```

`data` is `(T, m)` nested lists. The spec is the same `KalmanSpec` documented in
[Python API](9_Python-API.md).

### Follow-ups on a run

```bash
$ curl -s localhost:8000/diagnostics/run-3f9c2a1b7d4e | jq '.nis | length'
$ curl -s localhost:8000/plot/run-3f9c2a1b7d4e -o run.png
```

`/plot` returns `image/png`. A stored run carries no separate ground truth, so
the measurement-error panel is omitted; pass your own truth and plot locally if
you need it.

### Forecasting

```bash
$ curl -s -X POST localhost:8000/forecast -H 'Content-Type: application/json' \
    -d '{"spec": {…}, "x0": [[0.0], [1.0]], "steps": 4}'
{"xk": [...], "Pk": [...]}
```

Models with a domain forecast — the storm family's `forecast_cone` — return that
instead (`{lon, lat, I, I_sigma, ellipses}`). See
[Python API](9_Python-API.md).

Runs hold live model objects, so they stay in the memory of the REST process.
They do not survive a restart, and the oldest go first after
`KALMAN_MAX_RUNS` (default 1000). With more than one uvicorn worker, a `run_id`
from one worker is not known to the others, so run a single worker. When the
MCP server runs with `KALMAN_API_URL`, it sends its runs here too. Thus a
`run_id` that an agent gets can be used from Unity, and the other way round.

### Fetching from a provider

```bash
$ curl -s -X POST localhost:8000/fetch -H 'Content-Type: application/json' \
    -d '{"source": "lab-1", "query": {}}'
$ curl -s -X POST localhost:8000/fetch -H 'Content-Type: application/json' \
    -d '{"provider": {"kind": "synthetic", "init": {"spec": {…}}}, "query": {"duration": 5}}'
```

A raw `file`, `remote` or `sensor` provider needs the `admin` scope. A raw
`file` provider can read any file on the server. A raw `remote` provider makes
the server call a URL with the server's own outbound credentials. Other users
fetch from a **named source** that an admin set up (see below).

### Authoring

```bash
$ curl -s -X POST localhost:8000/models -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' -d '{"spec": {…}, "name": "demo_cv"}'
{"name":"demo_cv","n":2,"m":1,"l":0,"owner":"alice","action":"created"}

$ curl -s -X POST localhost:8000/scaffold -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' -d '{"spec": {…}, "class_name": "DemoCV"}'
{"source":"\"\"\"Scaffolded from a KalmanSpec …"}
```

`/scaffold` returns source text for a human to review and commit. Nothing is
written to the repository and nothing is executed.

---

## The shared registry

Models, plugins and measurement sources are kept in one registry. All clients
use it: curl, Unity, C++ programs, and the MCP server. It is a SQLite file in
`KALMAN_STATE_DIR` (default `~/.kalman/state`; `/var/lib/kalman` in the
container). The registry stays after a restart. Each process that uses the
same directory sees the same entries.

- **Register once, use everywhere.** After `POST /models`, the name is a spec
  base for all clients: `{"spec": {"base": "demo_cv"}, "data": …}`.
- **The same spec again does nothing** (`"action": "unchanged"`). Thus a
  client can register its models each time it starts.
- **A different spec under a used name gives `409`.** To change the entry,
  send `"replace": true`. Only the owner (the token's `sub`) or an admin can
  replace or delete an entry.
- **A name cannot hide a catalog model.** You cannot register `pendulum` or
  `OscillatorModel` for your own spec.
- **Plugins are loaded again in all processes.** A plugin that is loaded with
  `POST /plugins` is recorded. Each process that uses the registry loads it
  again by name from its own allow-listed directories. This includes a
  restarted server.

```bash
$ curl -s localhost:8000/models -H "Authorization: Bearer $TOKEN" | jq '.models[] | {name, owner, n, m}'
$ curl -s -X DELETE localhost:8000/models/demo_cv -H "Authorization: Bearer $TOKEN"
```

### Measurement sources

A **source** has a name. Clients fetch measurements from it, or filter them in
one step. There are two kinds:

- **`push`**: a buffer. A device, a simulation or a Unity scene sends rows to
  it, and any client reads them.
- **A provider that the server calls**, for example `remote` for a lab
  instrument behind HTTP. An admin sets it up one time, with the server's own
  outbound credential (only its *name*), and users fetch it by name.

```bash
# set up (write_models; admin for file/remote/sensor)
$ curl -s -X POST localhost:8000/sources -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"name": "scene-1", "provider": {"kind": "push", "m": 1, "description": "Unity scene 1"}}'
$ curl -s -X POST localhost:8000/sources -H "Authorization: Bearer $ADMIN" \
    -H 'Content-Type: application/json' \
    -d '{"name": "lab-1", "provider": {"kind": "remote",
         "init": {"base_url": "https://lab.example/api", "credential_name": "lab"}}}'

# send rows into a push source (send_measurement); time is optional, one per row
$ curl -s -X POST localhost:8000/sources/scene-1/measurements -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' -d '{"rows": [[0.01], [0.02]], "time": [0.0, 0.01]}'
{"source":"scene-1","first":41,"last":42,"count":2}

# measure and filter in one call
$ curl -s -X POST localhost:8000/filter -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"spec": {"base": "demo_cv"}, "source": "scene-1", "query": {"since": 0}}'

# or read the rows: meta.next is the cursor for the next read
$ curl -s "localhost:8000/sources/scene-1/measurements?since=40" -H "Authorization: Bearer $TOKEN"
```

A push source keeps the last `KALMAN_SOURCE_MAX_ROWS` rows (default 100000).
One push can have at most 10000 rows. If you set `m`, rows of a different
width are refused. A query can only narrow a source. It cannot change the
`path` of a file source or the `base_url` of a remote source.

### Subscribing to changes

Each change to the registry is an event: `model.created`, `model.replaced`,
`model.deleted`, `plugin.loaded`, `source.created`, `source.replaced`,
`source.deleted`, `measurement.received`. You can poll the events, or keep a
stream open:

```bash
$ curl -s "localhost:8000/events?since=0" -H "Authorization: Bearer $TOKEN"
{"events":[{"id":1,"type":"model.created","payload":{"name":"demo_cv","owner":"alice","n":2,"m":1,"l":0},"ts":…}],"last":1}

$ curl -N localhost:8000/events/stream -H "Authorization: Bearer $TOKEN"
retry: 3000

id: 2
event: measurement.received
data: {"id": 2, "type": "measurement.received", "payload": {"source": "scene-1", …}}
```

The stream starts from now. It starts from `?since=` if you give it, or from
the `Last-Event-ID` header that a reconnecting client sends. A comment line is
sent every 15 s, so that proxies do not close the stream.
`?timeout=<seconds>` closes the stream after that time. The log keeps the last
`KALMAN_MAX_EVENTS` events (default 10000).
---

## Known rough edges

- **Top-level `fs` is only forwarded when the model's `__init__` names it.**
  For a class that takes `**kwargs` instead (e.g. `BallisticLinearModel`), put
  it in `dynamics.params`: `{"dynamics": {"params": {"fs": 50.0}}}`.
- **CORS is off by default.** Browser clients, including Unity WebGL builds,
  need it. Set `KALMAN_CORS_ORIGINS` to the allowed origins, separated by
  commas, for example `https://game.example,http://localhost:8080`, or `*`.
  The token goes in the `Authorization` header, never in a cookie, so
  credentials mode stays off.
- **A browser `EventSource` cannot send headers.** Thus it cannot open
  `/events/stream` when auth is on. In a browser, poll `GET /events` with
  `fetch()`, or use an SSE client that sets headers.

---

## Connecting from Unity

Use the **REST API on port 8000**. Unity cannot speak the MCP stdio transport,
and while the MCP server can run over SSE on 8765, plain HTTP + JSON is the
right interface for a game engine.

Start the server so that the device can reach it. On a network interface,
the server needs a secret:

```bash
$ export KALMAN_JWT_SECRET=$(python -c 'import secrets; print(secrets.token_urlsafe(48))')
$ KALMAN_REST_HOST=0.0.0.0 KALMAN_REST_PORT=8000 python -m kalman.rest
$ make api-token USER=unity SCOPES=read,send_measurement      # paste into the client
```

Then in the Editor use `http://localhost:8000`; from a device on the same LAN
use the host machine's IP (`http://192.168.x.x:8000`).

```csharp
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public class KalmanClient : MonoBehaviour
{
    public string baseUrl = "http://localhost:8000";
    public string bearerToken = "";        // leave empty when auth is disabled

    [System.Serializable] public class FilterResult { public string run_id; public float[][] xk; }

    IEnumerator Start() { yield return Filter(); }

    IEnumerator Filter()
    {
        string body = @"{
          ""spec"": {
            ""base"": ""matrices"", ""fs"": 100.0,
            ""dynamics"": {""F"": [[1, 0.01], [0, 1]], ""H"": [[1, 0]]},
            ""noise"":    {""Q"": [[1e-6, 0], [0, 1e-4]], ""R"": [[0.05]]},
            ""initial"":  {""x0"": [[0], [1]]},
            ""bindings"": {""engine"": ""linear""}
          },
          ""data"": [[0.01], [0.02], [0.031], [0.039]]
        }";

        using var req = new UnityWebRequest($"{baseUrl}/filter", "POST");
        req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");
        if (!string.IsNullOrEmpty(bearerToken))
            req.SetRequestHeader("Authorization", $"Bearer {bearerToken}");

        yield return req.SendWebRequest();

        if (req.result != UnityWebRequest.Result.Success)
            Debug.LogError($"Kalman API: {req.responseCode} {req.error}");
        else
            Debug.Log(req.downloadHandler.text);      // {"run_id":"run-…","xk":[[…]],…}
    }
}
```

Notes for a real integration:

- **Nested arrays.** Unity's built-in `JsonUtility` cannot deserialize jagged
  arrays (`float[][]`) or top-level arrays. Use `Newtonsoft.Json`
  (`com.unity.nuget.newtonsoft-json`) for the response.
- **Per-frame filtering.** Do not POST the whole history every frame. Either
  batch the measurements you have accumulated, or keep the recursive update
  in C# and use the API for setup, tuning and diagnostics.
- **Latency.** Each call is a round trip plus model construction; expect
  milliseconds on localhost, more over a network. Nothing in the API is
  real-time-safe.
- **Tokens.** If you enable auth, mint a `read`+`send_measurement` token
  (`make api-token USER=unity SCOPES=read,send_measurement`) and never ship an
  `admin` token in a build — see [Authentication](12_Authentication.md). For
  a build that people use with their own accounts, do not put a token in the
  build: ask for the user and password, call `POST /token`, keep the
  `access_token` in memory, and call `POST /token/revoke` when the user logs
  out.
- **WebGL builds** also need CORS: set `KALMAN_CORS_ORIGINS` (see above).

### Sending measurements from a scene and getting the estimate back

Use a `push` source when the scene is the instrument. The scene sends what it
measures, and any client (this scene, an agent through MCP, a ROOT analysis)
can filter those rows by the source name:

```csharp
// once, by someone with write_models:  POST /sources {"name":"scene-1","provider":{"kind":"push","m":1}}
IEnumerator Push(float z, float t)
{
    string body = $"{{\"rows\": [[{z}]], \"time\": [{t}]}}";
    using var req = new UnityWebRequest($"{baseUrl}/sources/scene-1/measurements", "POST");
    req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
    req.downloadHandler = new DownloadHandlerBuffer();
    req.SetRequestHeader("Content-Type", "application/json");
    req.SetRequestHeader("Authorization", $"Bearer {bearerToken}");
    yield return req.SendWebRequest();
}

IEnumerator Estimate()
{
    // "demo_cv" is a model someone registered; any registered name or KalmanSpec works
    string body = @"{""spec"": {""base"": ""demo_cv""}, ""source"": ""scene-1""}";
    using var req = new UnityWebRequest($"{baseUrl}/filter", "POST");
    req.uploadHandler   = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
    req.downloadHandler = new DownloadHandlerBuffer();
    req.SetRequestHeader("Content-Type", "application/json");
    req.SetRequestHeader("Authorization", $"Bearer {bearerToken}");
    yield return req.SendWebRequest();
    Debug.Log(req.downloadHandler.text);          // {"run_id": "...", "xk": [[...]], ...}
}
```

Use a `remote` source when the server must ask an instrument for a measurement
on request. Then `POST /filter {"spec": …, "source": "lab-1"}` makes the
measurement and gives the estimate in one call.

Send the measurements in batches, for example every 100 ms, not one request
per frame. To find new models without a restart, poll `GET /events?since=<last>`
from a coroutine, and update the model list when you get a `model.created`
event.

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
| **REST API** | **8000** | `KALMAN_REST_HOST` (default `0.0.0.0`), `KALMAN_REST_PORT` |
| MCP server (SSE) | 8765 | `KALMAN_MCP_HOST`, `KALMAN_MCP_PORT`, `KALMAN_MCP_TRANSPORT` |

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
| `http://localhost:8000/healthz` | `{"ok": true}`, unauthenticated |

---

## Endpoints

| Method | Path | Scope | Body / params |
|---|---|---|---|
| `GET` | `/healthz` | none | — |
| `GET` | `/catalog` | `read` | — |
| `GET` | `/describe/{name}` | `read` | — |
| `GET` | `/diagnostics/{run_id}` | `read` | — |
| `GET` | `/plot/{run_id}` | `read` | returns `image/png` |
| `POST` | `/filter` | `send_measurement` | `{spec, data, smooth?}` |
| `POST` | `/forecast` | `send_measurement` | `{spec, x0, P0?, steps}` |
| `POST` | `/osse` | `send_measurement` | `{spec, options?}` |
| `POST` | `/control` | `send_measurement` | `{model, controller, duration}` |
| `POST` | `/fetch` | `send_measurement` | `{provider, query?}` |
| `POST` | `/models` | `write_models` | `{spec, name?}` |
| `POST` | `/scaffold` | `write_models` | `{spec, class_name?}` |

Status codes: `401` when no token was presented, `403` when the token lacks the
scope, `404` for an unknown model or `run_id`.

---

## Examples

Authentication is off until you set `KALMAN_JWT_SECRET`, so a local development
server needs no header. Add `-H "Authorization: Bearer $TOKEN"` once it is on.

```bash
$ curl -s localhost:8000/healthz
{"ok":true}

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
{"run_id":"run0",
 "xk":[[0.01,1.0],[0.02,1.0],[0.030000082,1.00000012],[0.039999978,0.99999992]],
 "Pk":[[[2.0099e-06,9.9996e-07],[9.9996e-07,1.99999e-04]], …],
 "time":[…]}
```

`data` is `(T, m)` nested lists. The spec is the same `KalmanSpec` documented in
[Python API](9_Python-API.md).

### Follow-ups on a run

```bash
$ curl -s localhost:8000/diagnostics/run0 | jq '.nis | length'
$ curl -s localhost:8000/plot/run0 -o run0.png
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

`run_id`s live in an **in-process dict**. They do not survive a restart, and
with more than one uvicorn worker a `run_id` minted by one worker is invisible
to the others. Run a single worker, or treat every request as self-contained.

### Fetching from a provider

```bash
$ curl -s -X POST localhost:8000/fetch -H 'Content-Type: application/json' \
    -d '{"provider": {"kind": "file", "init": {"path": "run.npz"}}, "query": {}}'
```

### Authoring

```bash
$ curl -s -X POST localhost:8000/models -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' -d '{"spec": {…}, "name": "demo_cv"}'
{"name":"demo_cv","n":2,"m":1,"l":0}

$ curl -s -X POST localhost:8000/scaffold -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' -d '{"spec": {…}, "class_name": "DemoCV"}'
{"source":"\"\"\"Scaffolded from a KalmanSpec …"}
```

`/scaffold` returns source text for a human to review and commit. Nothing is
written to the repository and nothing is executed.

---

## Known rough edges

- **Top-level `fs` is only forwarded when the model's `__init__` names it.**
  For a class that takes `**kwargs` instead (e.g. `BallisticLinearModel`), put
  it in `dynamics.params`: `{"dynamics": {"params": {"fs": 50.0}}}`.
- **No CORS middleware is configured.** Browser-based callers (including Unity
  WebGL builds) will be blocked by the same-origin policy. Put the API behind a
  reverse proxy that adds the headers, or add `CORSMiddleware` in
  `modules/rest/app.py`.

---

## Connecting from Unity

Use the **REST API on port 8000**. Unity cannot speak the MCP stdio transport,
and while the MCP server can run over SSE on 8765, plain HTTP + JSON is the
right interface for a game engine.

Start the server so it is reachable from the device:

```bash
$ KALMAN_REST_HOST=0.0.0.0 KALMAN_REST_PORT=8000 python -m kalman.rest
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
            Debug.Log(req.downloadHandler.text);      // {"run_id":"run0","xk":[[…]],…}
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
  `admin` token in a build — see [Authentication](12_Authentication.md).
- **WebGL builds** additionally need CORS headers, which are not configured (see
  above).

# MCP Server

The Model Context Protocol server exposes the
[`kalman.api` facade](9_Python-API.md) to AI clients — Claude Code, Claude
Desktop, Cursor, VS Code, Codex, Gemini CLI and others. It is a thin adapter:
every tool is one facade call plus a scope check.

Server identifier: **`Kalman-AI`** — the config key, the tool-name prefix, and
the name most clients show in their server list.

> Renamed from `kalman-ai`. `make mcp` (or `tools/mcp_install.py`) migrates an
> existing registration: the old key is removed and the new one written in the
> same pass, reported as `would-rename` / `updated`. Re-run it once per machine,
> then restart the client.

```bash
$ pip install "kalman[mcp]"
$ make mcp                        # register with every detected client, then verify
```

---

## Tools

| Tool | Scope | Arguments |
|---|---|---|
| `list_models` | `read` | `token?` |
| `describe_model` | `read` | `name`, `token?` |
| `list_registered_models` | `read` | `token?` — the shared registry, with owners |
| `get_diagnostics` | `read` | `run_id`, `token?` |
| `list_providers` / `describe_provider` | `read` | `name`, `token?` |
| `list_sources` | `read` | `token?` — named measurement sources |
| `list_events` | `read` | `since=0`, `limit=100`, `token?` — what changed in the registry |
| `run_filter` | `send_measurement` | `spec`, `data` or `source` + `query?`, `smooth?`, `token?` |
| `forecast` | `send_measurement` | `spec`, `x0`, `P0?`, `steps=1`, `token?` |
| `run_osse` | `send_measurement` | `spec`, `n_cycles=40`, `n_members=16`, `token?` |
| `run_control` | `send_measurement` | `model`, `controller`, `duration`, `token?` |
| `fetch_data` | `send_measurement` | `provider` (a source name, or `{kind, init}`), `query?`, `token?` |
| `push_measurements` | `send_measurement` | `source`, `rows`, `time?`, `token?` |
| `register_model` | `write_models` | `spec`, `name?`, `replace?`, `token?` |
| `delete_model` | `write_models` + owner | `name`, `token?` |
| `register_source` | `write_models` (`admin` for `file`/`remote`/`sensor`) | `name`, `provider`, `replace?`, `token?` |
| `scaffold_app` | `write_models` | `spec`, `class_name?`, `token?` |
| `load_plugin` | `load_plugins` | `name`, `token?` — a compiled (C++) model library, by name from the allow-listed dirs |

Three resources are published:

| Resource | Content |
|---|---|
| `catalog://models` | every model that can be a spec base, and the authored ones |
| `catalog://sources` | the named measurement sources |
| `registry://events` | the 100 newest registry events |

Apps need no per-app MCP code — they are auto-cataloged, and a registered spec
is indistinguishable from a built-in model.

`token` is the scoped JWT. You can omit it when authentication is off, which
is the normal local setup. You can also omit it when the token comes from the
HTTP `Authorization` header or from `KALMAN_MCP_TOKEN`. See
[Authentication](12_Authentication.md#presenting-a-token).

---

## One registry for agents and HTTP clients

Models, plugins and sources that you register through MCP are in the shared
registry (see [REST API](10_REST-API.md#the-shared-registry)). Thus REST
clients such as Unity, other agents and other users see them, and the MCP
server sees what they register. Tool results stay current: `list_models`
reads the registry each time it is called.

There are two ways to run the server:

| | In-process (default) | Client of the REST API (`KALMAN_API_URL`) |
|---|---|---|
| Registry | shared through `KALMAN_STATE_DIR` | the REST API's |
| Runs (`run_id`) | in this process only | in the REST process: an agent's `run_id` works from Unity |
| Token check | here, needs `KALMAN_JWT_SECRET` | by the REST API; this process needs no secret |
| Native code | loads plugins here | none; the REST process loads them |

Use the REST client mode when several people or programs work together:

```bash
$ KALMAN_API_URL=https://kalman.example python -m kalman.mcp          # stdio, for a local client
$ KALMAN_MCP_TRANSPORT=streamable-http KALMAN_MCP_HOST=0.0.0.0 \
  KALMAN_API_URL=http://rest:8000 python -m kalman.mcp                # shared, on :8765/mcp
```

### Change notifications

A client that subscribes to a resource gets `notifications/resources/updated`
when the registry changes. It does not matter who made the change: this
server, the REST API, another agent or a Unity scene. The server supports both
ways to subscribe:

- `resources/subscribe`, for MCP versions up to 2025-11-25. Most clients use
  this today.
- `subscriptions/listen`, for MCP 2026-07-28.

| Registry event | Resource reported as updated |
|---|---|
| `model.*`, `plugin.loaded` | `catalog://models` |
| `source.*`, `measurement.received` | `catalog://sources` |
| every event | `registry://events` |

```python
# MCP up to 2025-11-25: resources/subscribe, notifications to the message handler
async def on_message(message):
    if getattr(getattr(message, "params", None), "uri", None) == "catalog://models":
        ...                                           # a model or plugin was added: read it again

async with ClientSession(read, write, message_handler=on_message) as session:
    await session.initialize()
    await session.subscribe_resource("catalog://models")

# MCP 2026-07-28: subscriptions/listen
from mcp.client import Client

async with Client("https://kalman.example/mcp") as client:
    async with client.listen(resource_subscriptions=["catalog://models"]) as sub:
        async for event in sub:
            models = await client.read_resource("catalog://models")
```

The notification has no registry content. It tells the client to read the
resource again, and that read is checked against the client's token as usual.
The server looks for changes every `KALMAN_EVENTS_POLL` seconds (default 1).

In REST client mode with auth on, the server has no token of its own to read
the events. Notifications work in both cases:

- **With `KALMAN_MCP_TOKEN`** (a token with the `read` scope): the server
  reads the events, and reports only the resource that changed.
- **Without it**: the server reads `GET /events/cursor`, which needs no token
  and gives only the id of the newest event. On each change it reports all
  three resources, and the clients read again the ones they use.

A client that does not subscribe (or a server with SDK 1.x, which has
neither way) calls `list_events(since=<last>)`, which gives
`model.created`, `plugin.loaded`, `measurement.received` and the other events.
HTTP clients can get the same events as a stream from `GET /events/stream`.

There is no MCP tool for figures — a PNG is not a useful protocol payload, so
plotting stays on `GET /plot/{run_id}` over [REST](10_REST-API.md).

---

## Registering with a client

```bash
$ make mcp-clients                # what is detected on this machine
platform: mac   repo: /path/to/kalman-framework
  claude-code        installed  ~/.claude.json  <- kalman registered
  claude-desktop     installed  ~/Library/…/claude_desktop_config.json  <- kalman registered
  codex              installed  ~/.codex/config.toml  <- kalman registered
  cursor             not found  ~/.cursor/mcp.json
  …

$ make mcp                        # register everywhere detected + verify the server builds
$ make mcp-uninstall              # remove our entry from every client
```

Under the hood this is `tools/mcp_install.py`:

```bash
$ python tools/mcp_install.py --list
$ python tools/mcp_install.py --all --dry-run          # show the diff first
$ python tools/mcp_install.py claude-code cursor       # only these
$ python tools/mcp_install.py --all --remove
$ python tools/mcp_install.py --all --token "$TOKEN" --jwt-secret "$SECRET"
$ python tools/mcp_install.py --project --portable     # repo-local, committable config
```

The script knows each client's config location and container key (`mcpServers`,
VS Code's `servers`, Codex's TOML table), **edits only our own entry**, and backs
the file up first (`<file>.bak-<timestamp>`). Restart the client to pick up the
change.

Supported clients: `claude-code`, `claude-desktop`, `cursor`, `windsurf`,
`vscode`, `vscode-insiders`, `cline`, `gemini-cli`, `codex`, `opencode`, `kiro`,
`trae`. To add one, append it to `CLIENTS` in that script.

### Per-user vs committed config

The default entry embeds absolute paths — correct for a per-user config.
`--portable` emits machine-independent values (`python` from `PATH`,
`PYTHONPATH="."`) for a config committed to the repository; `--project` writes
`.vscode/mcp.json` and `.mcp.json` alongside the code. The repository already
ships that portable form:

```json
{
  "mcpServers": {
    "Kalman-AI": {
      "command": "python",
      "args": ["-m", "kalman.mcp"],
      "env": {"PYTHONPATH": "."}
    }
  }
}
```

Anyone who clones the repository and opens it in a client that reads `.mcp.json`
gets the server with no setup.

---

## Running it manually

```bash
$ make mcp-serve                  # python -m kalman.mcp — stdio, waits for a client
```

| Variable | Default | Meaning |
|---|---|---|
| `KALMAN_MCP_TRANSPORT` | `stdio` | `stdio`, `sse`, `streamable-http` |
| `KALMAN_MCP_HOST` | `127.0.0.1` | bind address for network transports |
| `KALMAN_MCP_PORT` | `8765` | port for network transports |
| `KALMAN_MCP_PUBLIC_URL` | `http://<host>:<port>` | the URL clients use (OAuth resource metadata) |
| `KALMAN_API_URL` | unset | send all work to this REST API |
| `KALMAN_JWT_SECRET` | unset | enables scope enforcement (in-process mode) |
| `KALMAN_MCP_TOKEN` | unset | stdio: the token used when a call gives none. REST client mode: an optional `read` token, so that a notification names the resource that changed |

Over `sse` and `streamable-http`, clients send `Authorization: Bearer <JWT>`.
The server requires it when it can check tokens: with `KALMAN_JWT_SECRET`, or
with a REST API that has auth on. A connection without a valid token gets
`401`. A server on an address other than loopback, with no way to check
tokens, does not start (unless `KALMAN_ALLOW_ANONYMOUS=1`).

stdio is right locally; SSE on **8765** is what `make serve` runs in the
container. See [Deployment](13_Deployment.md).

Without the SDK installed, `python -m kalman.mcp` prints an install hint and
exits 0 rather than crashing — the core is never blocked by a missing optional
dependency.

### SDK compatibility

The server supports both SDK generations: MCP SDK ≥ 2.0 exposes the
decorator-style class as `mcp.server.MCPServer`, while 1.x called the same thing
`mcp.server.fastmcp.FastMCP`. `_server_class()` tries both. This is covered by a
**live stdio session test** that spawns `python -m kalman.mcp` and drives it as a
real client — it caught the SDK 2.0 rename, which a mocked test would have
missed.

---

## What an agent can do with it

A useful session shape, all through the tools above:

1. `list_models` → see what exists;
2. `describe_model` → parameters and bindings for a candidate;
3. `run_filter` with a `KalmanSpec` → estimates plus a `run_id`;
4. `get_diagnostics` → NIS and innovations, to judge whether the tuning holds;
5. `register_model` → publish a new spec, validated and smoke-tested, to the
   shared registry, where Unity, REST clients and other agents can use it;
6. `scaffold_app` → emit reviewable Python for a human to commit.
7. `load_plugin` → bring a compiled C++ model library into the catalogue (see
   [C++ Library](15_Cxx-Library.md)); its models are then ordinary spec bases,
   in every process that shares the registry;
8. `run_filter(source="scene-1")` → filter the measurements a device or a Unity
   scene pushed into a source, and `list_events` → see what changed since the
   last look.

The authoring path is safe to expose because **specs are pure data**: there is
no `eval`/`exec` anywhere in `kalman.api`, only catalog names and the
`matrices`/`continuous` declarative bases are accepted, and `scaffold_app`
returns source text without writing, importing or executing anything.

If you want scopes enforced (an agent that may read and run, but not author),
enable authentication and register the client with a scoped token:

```bash
$ export KALMAN_JWT_SECRET="…"
$ TOKEN=$(make -s api-token USER=agent SCOPES=read,send_measurement)
$ python tools/mcp_install.py claude-code --token "$TOKEN" --jwt-secret "$KALMAN_JWT_SECRET"
```

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
| `get_diagnostics` | `read` | `run_id`, `token?` |
| `run_filter` | `send_measurement` | `spec`, `data`, `token?` |
| `forecast` | `send_measurement` | `spec`, `x0`, `P0?`, `steps=1`, `token?` |
| `run_osse` | `send_measurement` | `spec`, `n_cycles=40`, `n_members=16`, `token?` |
| `run_control` | `send_measurement` | `model`, `controller`, `duration`, `token?` |
| `fetch_data` | `send_measurement` | `provider`, `query?`, `token?` |
| `register_model` | `write_models` | `spec`, `name?`, `token?` |
| `scaffold_app` | `write_models` | `spec`, `class_name?`, `token?` |
| `load_plugin` | `load_plugins` | `name`, `token?` — a compiled (C++) model library, by name from the allow-listed dirs |

One resource is published: **`catalog://models`**, the model-name list.

Apps need no per-app MCP code — they are auto-cataloged, and a spec authored in
the session is indistinguishable from a built-in model.

`token` is the scoped JWT and may be omitted entirely when authentication is
disabled, which is the normal local setup. See
[Authentication](12_Authentication.md).

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
| `KALMAN_MCP_HOST` | `0.0.0.0` | bind address for network transports |
| `KALMAN_MCP_PORT` | `8765` | port for network transports |
| `KALMAN_JWT_SECRET` | unset | enables scope enforcement |
| `KALMAN_MCP_TOKEN` | unset | token injected into the launch environment |

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
5. `register_model` → author a new spec, validated and smoke-tested;
6. `scaffold_app` → emit reviewable Python for a human to commit.
7. `load_plugin` → bring a compiled C++ model library into the catalogue (see
   [C++ Library](15_Cxx-Library.md)); its models are then ordinary spec bases.

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

# Deployment

Two services front the framework: the [REST API](10_REST-API.md) and the
[MCP server](11_MCP-Server.md). Both are stateless apart from the in-process
run store, and both read the same environment.

---

## Ports

| Service | Port | Bind variable | Notes |
|---|---|---|---|
| REST API | **8000** | `KALMAN_REST_HOST` / `KALMAN_REST_PORT` | HTTP + JSON, OpenAPI at `/docs` |
| MCP (SSE) | **8765** | `KALMAN_MCP_HOST` / `KALMAN_MCP_PORT` | only for network transports; stdio locally |

Both are published by `docker-compose.yml` as `8000:8000` and `8765:8765`.

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `KALMAN_REST_HOST` | `127.0.0.1` | REST |
| `KALMAN_REST_PORT` | `8000` | REST |
| `KALMAN_MCP_TRANSPORT` | `stdio` | MCP — `stdio`, `sse`, `streamable-http` |
| `KALMAN_MCP_HOST` | `127.0.0.1` | MCP (network transports) |
| `KALMAN_MCP_PORT` | `8765` | MCP (network transports) |
| `KALMAN_MCP_PUBLIC_URL` | `http://<host>:<port>` | MCP: the URL clients use, given in its OAuth resource metadata |
| `KALMAN_MCP_TOKEN` | unset | MCP over stdio: the token used when a call gives none. MCP as a REST client: an optional `read` token, so that a change notification names the resource that changed |
| `KALMAN_API_URL` | unset | MCP: send all work to this REST API (one registry, one set of runs) |
| `KALMAN_API_TIMEOUT` | `300` | MCP: seconds to wait for the REST API |
| `KALMAN_JWT_SECRET` | unset | **enables authentication** everywhere; at least 32 bytes |
| `KALMAN_JWT_ISSUER` / `KALMAN_JWT_AUDIENCE` | unset | stamp and require `iss` / `aud` in tokens |
| `KALMAN_ALLOW_ANONYMOUS` | `0` | `1` lets a server without auth listen on a network address |
| `KALMAN_DEFAULT_SCOPES` | `read` | login scopes of an account without a grant |
| `KALMAN_TOKEN_MAX_HOURS` | `24` | longest lifetime of a token from `POST /token` |
| `KALMAN_CORS_ORIGINS` | unset | REST: browser origins allowed, comma-separated, or `*` |
| `KALMAN_STATE_DIR` | `~/.kalman/state` | the shared registry (`registry.sqlite`) |
| `KALMAN_MAX_RUNS` | `1000` | runs kept in memory before the oldest go |
| `KALMAN_SOURCE_MAX_ROWS` | `100000` | rows kept per push source |
| `KALMAN_MAX_EVENTS` | `10000` | events kept in the log |
| `KALMAN_EVENTS_POLL` | `0.5` | seconds between registry checks in `/events/stream` |
| `KALMAN_HTPASSWD` | `~/.kalman/htpasswd` | account store location |
| `KALMAN_BACKEND` | `numpy` | solver tensor backend (`numpy` / `torch`) |
| `KALMAN_DEVICE` | `cpu` | solver device (`cpu` / `cuda` / `mps`) |
| `KALMAN_<NAME>_KEY` | — | outbound provider credentials |

`.env` at the repository root is loaded by both the Makefile and Compose, and is
gitignored. That is where the secret belongs.

---

## Docker Compose

```bash
$ make image                # build
$ make serve                # docker compose up -d rest mcp
$ make stop                 # down
```

Three services are defined:

- **`rest`** — `python -m kalman.rest` on 8000, with a `/healthz` healthcheck.
  The repository is mounted read-only. The htpasswd file is mounted at
  `/run/secrets/htpasswd`, and `KALMAN_HTPASSWD` points at it. The shared
  registry is in the named volume `state` (`/var/lib/kalman`), so it stays
  after `make stop`.
- **`mcp`** — `python -m kalman.mcp` with `KALMAN_MCP_TRANSPORT=sse` on 8765,
  as a client of `rest` (`KALMAN_API_URL=http://rest:8000`). It starts after
  `rest` is healthy. It does not get the JWT secret, because `rest` checks the
  tokens.
- **`container`** — the interactive development container (`make shell`), with
  conda cached in a named volume.

```bash
$ export KALMAN_JWT_SECRET="$(openssl rand -hex 32)"
$ export KALMAN_HTPASSWD="$HOME/.kalman/htpasswd"
$ make serve
$ curl -s localhost:8000/healthz
{"ok":true,"auth":true}
```

Without `KALMAN_JWT_SECRET`, both services do not start, because they listen on
`0.0.0.0` in the container. On a closed development network, set
`KALMAN_ALLOW_ANONYMOUS=1` before `make serve`.

Both service containers mount the source read-only (`./:/mnt/local:ro`), so a
running service cannot modify the repository — including via `write_scaffold`.

---

## Running without Docker

```bash
$ python -m kalman.rest                                         # 127.0.0.1:8000
$ KALMAN_MCP_TRANSPORT=streamable-http KALMAN_API_URL=http://127.0.0.1:8000 \
      python -m kalman.mcp                                      # 127.0.0.1:8765/mcp
```

To serve other machines, set `KALMAN_REST_HOST=0.0.0.0` (and
`KALMAN_MCP_HOST=0.0.0.0`) together with `KALMAN_JWT_SECRET`.

Or via the console scripts installed by `pip install -e .`: `kalman-rest`,
`kalman-mcp`.

### One worker, please

The registry (models, plugins, sources, events) is on disk, so all workers
share it. Runs are different: they hold live model objects, so they stay in the
memory of each process. Under multiple uvicorn workers, a `run_id` from one
worker is not known to the others, and `/diagnostics/{run_id}` and
`/plot/{run_id}` sometimes give 404. Run a single worker. To share runs between
agents and HTTP clients, give the MCP server `KALMAN_API_URL`, so that all runs
are made in the REST process.

---

## Hardening checklist

- **Enable auth.** `KALMAN_JWT_SECRET` ≥ 32 bytes. The servers do not start on
  a network address without it (unless `KALMAN_ALLOW_ANONYMOUS=1`), and they
  do not start with a shorter secret. See [Authentication](12_Authentication.md).
- **Terminate TLS in front.** Tokens are bearer credentials in a header; the
  app speaks plain HTTP.
- **Do not expose 8000/8765 to the internet directly.** Put them behind a
  reverse proxy that also supplies rate limiting. For browser clients, set
  `KALMAN_CORS_ORIGINS` to the exact origins, not `*`.
- **Back up the `state` volume.** It holds the shared registry. It has no
  secrets: a remote source records only the *name* of its outbound credential.
- **Scope tokens narrowly and expire them quickly.** Give each account only the
  grant it needs (`make api-user USER=… SCOPES=…`), and revoke a token that
  leaks (`POST /token/revoke`).
- **Rate-limit `POST /token` at the proxy.** The server locks out a user and
  client address after 5 failed logins, on all workers and across restarts.
  Behind a proxy, all clients have the address of the proxy, so the lockout is
  then per user. A rate limit at the proxy also stops a client that tries many
  different user names.
- **Keep the source mount read-only.** `write_scaffold` writes to a path the
  caller chooses.
- **Watch resource cost.** `run_osse` runs a full ensemble twin experiment and
  `run_control` trains LQR/neural controllers; both are unbounded in CPU time
  from the caller's point of view. Enforce timeouts at the proxy.
- **Mount `~/.kalman/` read-only** if the container only needs to verify
  accounts, not create them.

---

## CI and the documentation site

`.gitlab-ci.yml` builds the image, installs the package, runs the suite, and
then builds the documentation.

| Job | Stage | When | What |
|---|---|---|---|
| `docs` | `post-build` | **every commit** | `make docs` — builds the site, fails the pipeline if Doxygen cannot, and keeps `public/` as an artifact |
| `pages` | `deploy` | default branch + version tags | `make pages` — publishes to GitLab Pages |

So every push compiles the documentation and every branch gets a downloadable
`public/` artifact; only the default branch and tags actually publish, because
GitLab Pages serves one deployment per project.

### Versions

Each build lands in its own folder, named after the version: `1.0.2` for a
release tag, `1.0.2-11-g64fefe7` for a build of the default branch between
releases. The menu in the side bar switches between them and keeps you on the
same page when the other version has it.

Because Pages serves a single deployment, the `pages` job carries the earlier
versions along: `make pages-restore` takes them from the artifacts of the
previous `pages` jobs (the default branch and each release tag), `make pages`
builds the current one beside them, and `make pages-prune` keeps every release
and only the newest development build. A release is therefore listed from the
day its tag is pushed, and stays.

`tools/site_versions.py` does the listing, the ordering (by version number, so
1.0.10 comes after 1.0.9) and the pruning. Locally, `make docs` lists whatever
is in `var/build/public/docs`; serve that folder with
`python -m http.server` to try the menu, since a page opened from `file://`
cannot read the list.

The site is Doxygen + [doxygen-awesome](https://git.ligo.org/kagra/libraries-addons/doxygen-awesome),
driven by CMake, exactly as the sibling projects on this template do it. Two
submodules, both fetched by CI through `GIT_SUBMODULE_PATHS`:

| Submodule | Provides |
|---|---|
| `share/doxygen` | the theme: `Doxyfile.in` (the root `Doxyfile.in` symlinks to it), layout, CSS/JS, and the `doxygen` / `metadata` / `pages` targets |
| `cmake/Modules` | the shared KAGRA CMake helpers — `BuildOptions`, `DoxygenVariables`, `ProjectArchitecture`, `Message`, … |

The root `CMakeLists.txt` only builds documentation: `BUILD_LIBRARY`,
`BUILD_TOOLS`, `BUILD_EXAMPLES` and `BUILD_TESTS` are forced off (the package is
installed with pip, the C fast path by CFFI, and the suite runs under pytest),
and so is `BUILD_PYPROJECT` — `pyproject.toml` is hand-maintained here and
`pyproject.py` would regenerate it without the extras, console scripts or tool
configuration.

### What ends up in the site

`DoxygenVariables.cmake` collects `README.md`, `LICENSE`, `CONTRIBUTING.md` and
everything under `docs/`, `examples/` and `tests/`; the root `CMakeLists.txt`
adds `modules/` (the Python package — Doxygen reads its docstrings, with
`PYTHON_DOCSTRING = YES`) and `AGENTS.md`. `README.md` is the landing page.

Versioning follows the theme's convention: output goes to
`public/docs/<version>/`, where `<version>` is the tag if there is one, else the
short commit hash, with `public/docs/latest` pointing at the newest build and
`metadata.txt` listing every version present.

### Building it yourself

```bash
$ git submodule update --init --recursive    # once: theme + cmake helpers
$ make docs        # var/build/public/docs/<version>/
$ make pages       # + public/ and the latest alias, then opens it
```

Both configure the CMake tree on demand (`bootstrap.sh` → `var/build`); they
need `cmake`, `doxygen` and `graphviz`.

The theme links `public/` into the build tree, and GitLab artifacts do not
follow symlinks, so CI finishes with `make pages-materialize`, which replaces
the link with a real copy and fails if nothing was generated. (Setting
`PROJECT_SYMLINK=OFF` does *not* work: the theme re-defaults any falsey value
back to `ON`.)

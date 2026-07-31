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
| `KALMAN_REST_HOST` | `0.0.0.0` | REST |
| `KALMAN_REST_PORT` | `8000` | REST |
| `KALMAN_MCP_TRANSPORT` | `stdio` | MCP — `stdio`, `sse`, `streamable-http` |
| `KALMAN_MCP_HOST` | `0.0.0.0` | MCP (network transports) |
| `KALMAN_MCP_PORT` | `8765` | MCP (network transports) |
| `KALMAN_MCP_TOKEN` | unset | token injected into a registered client's launch env |
| `KALMAN_JWT_SECRET` | unset | **enables authentication** everywhere |
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

- **`rest`** — `python -m kalman.rest` on 8000, with a `/healthz` healthcheck
  every 30 s. The repository is mounted read-only; the htpasswd file is mounted
  at `/run/secrets/htpasswd` and `KALMAN_HTPASSWD` points at it.
- **`mcp`** — `python -m kalman.mcp` with `KALMAN_MCP_TRANSPORT=sse` on 8765.
- **`container`** — the interactive development container (`make shell`), with
  conda cached in a named volume.

```bash
$ export KALMAN_JWT_SECRET="$(openssl rand -hex 32)"
$ export KALMAN_HTPASSWD="$HOME/.kalman/htpasswd"
$ make serve
$ curl -s localhost:8000/healthz
{"ok":true}
```

Both service containers mount the source read-only (`./:/mnt/local:ro`), so a
running service cannot modify the repository — including via `write_scaffold`.

---

## Running without Docker

```bash
$ KALMAN_REST_HOST=0.0.0.0 KALMAN_REST_PORT=8000 python -m kalman.rest
$ KALMAN_MCP_TRANSPORT=sse KALMAN_MCP_PORT=8765  python -m kalman.mcp
```

Or via the console scripts installed by `pip install -e .`: `kalman-rest`,
`kalman-mcp`.

### One worker, please

`run_id`s live in a per-process dictionary. Under multiple uvicorn workers, a
`run_id` returned by one worker is invisible to the others, so
`/diagnostics/{run_id}` and `/plot/{run_id}` will intermittently 404. Run a
single worker until the run store is externalised, or design clients to treat
each request as self-contained.

---

## Hardening checklist

- **Enable auth.** `KALMAN_JWT_SECRET` ≥ 32 bytes. Without it every endpoint is
  open. See [Authentication](12_Authentication.md).
- **Terminate TLS in front.** Tokens are bearer credentials in a header; the
  app speaks plain HTTP.
- **Do not expose 8000/8765 to the internet directly.** Put them behind a
  reverse proxy that also supplies rate limiting and, if browsers are involved,
  CORS headers (none are configured in the app).
- **Scope tokens narrowly and expire them quickly.** There is no revocation
  list.
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

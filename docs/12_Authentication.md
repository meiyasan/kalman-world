# Authentication

One scoped-token system guards the [facade](9_Python-API.md) and therefore
both transports. It is **fail-closed**: when a secret is configured, anything
missing, invalid or expired is denied.

```bash
$ pip install "kalman[auth]"        # bcrypt + pyjwt
```

> Do not confuse this with **outbound** credentials
> (`kalman.providers.credentials`), which are *our* secrets for reaching remote
> services. Those are covered in [Providers](7_Providers.md).

## Two layers

| Layer | What it is | Where it lives |
|---|---|---|
| **Account store** | Apache-format `.htpasswd`, `name:bcrypt` | `~/.kalman/htpasswd`, or `$KALMAN_HTPASSWD` |
| **Bearer tokens** | signed JWTs (HS256) carrying scopes | minted on demand, never stored |

Accounts exist only to authenticate a person *minting* a token. Callers present
tokens, never passwords.

## Scopes

| Scope | Unlocks |
|---|---|
| `read` | introspection: `catalog`, `describe`, `diagnostics`, plots |
| `send_measurement` | execution: `run_filter`, `forecast`, `run_osse`, `run_control`, `fetch` |
| `write_models` | authoring: `register_model`, `save_spec`, `scaffold` |
| `load_plugins` | loading compiled model libraries: `load_plugin` — this one runs code, hence its own scope |
| `admin` | everything, plus account management |

`admin` is a superscope — a token carrying it satisfies every check. Scopes are
not hierarchical otherwise: `send_measurement` does **not** imply `read`. Mint
`read,send_measurement` when a client needs both.

---

## Enabling it

Authentication is **off** until `KALMAN_JWT_SECRET` is set. Off means in-process
Python use is unrestricted — the local user already has the machine — and every
transport accepts unauthenticated calls. That is the right default for a laptop
and the wrong one for anything reachable by others.

```bash
$ export KALMAN_JWT_SECRET="$(openssl rand -hex 32)"
```

Use at least 32 bytes; PyJWT warns below that. Keep the secret out of the
repository and out of shell history — put it in `.env`, which is gitignored.

```python
from kalman.api.auth import enabled
enabled()        # True once the secret is set
```

## Creating an account

```bash
$ make api-user USER=marco
password for marco: ␣
account 'marco' written to /Users/marco/.kalman/htpasswd
```

The file is written `0600` in the Apache `user:$2b$…` format that httpd, nginx
and every htpasswd tool understand, using the maintained `bcrypt` package
directly (`passlib` is unmaintained and breaks against bcrypt ≥ 5). Apache's
`$2y$` prefix is accepted on verification; a non-bcrypt hash is rejected with an
explicit message telling you to re-add the account.

## Minting a token

```bash
$ make api-token USER=marco SCOPES=read,send_measurement EXP=24
password for marco: ␣
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9…

$ make api-admin-token USER=marco EXP=1
```

`EXP` is in hours (default 24). Underneath:

```bash
$ python -m kalman.api.auth adduser --user marco [--htpasswd PATH]
$ python -m kalman.api.auth token   --user marco --scopes read --exp 24 [--no-verify]
$ python -m kalman.api.auth verify  --token "$TOKEN" --scope read
OK
```

`--no-verify` mints without checking the account — for CI and automation where
no password can be typed. Guard the secret accordingly.

## Presenting a token

**REST** — an `Authorization` header:

```bash
$ curl -s localhost:8000/catalog -H "Authorization: Bearer $TOKEN"
```

`401` when no token was presented, or the token is forged or expired. The
response has `WWW-Authenticate: Bearer`. `403` when the token does not have
the scope. `GET /whoami` shows what a token is:

```bash
$ curl -s localhost:8000/whoami -H "Authorization: Bearer $TOKEN"
{"auth":true,"user":"alice","scopes":["read","send_measurement"],"exp":1790000000}
```

**MCP over HTTP** (`sse`, `streamable-http`): the same `Authorization: Bearer`
header, on the connection. The server checks the token before a session opens.
Thus a client without a valid token gets `401` and cannot call any tool. Each
tool then checks its own scope.

```json
{"mcpServers": {"kalman": {"type": "http", "url": "https://kalman.example/mcp",
                           "headers": {"Authorization": "Bearer eyJ…"}}}}
```

**MCP over stdio**: there is no header. The server uses `KALMAN_MCP_TOKEN` from
its launch environment, so the agent never has to handle the token:

```bash
$ python tools/mcp_install.py claude-code --token "$TOKEN" --jwt-secret "$KALMAN_JWT_SECRET"
```

A client that cannot set either one can still give the `token` argument on
each tool call:

```json
{"name": "run_filter", "arguments": {"spec": {…}, "data": [...], "token": "eyJ…"}}
```

The order is: the `token` argument, then the `Authorization` header, then
`KALMAN_MCP_TOKEN` (stdio only).

When the MCP server runs with `KALMAN_API_URL`, it does not have the secret.
It sends the caller's token to the REST API. The REST API checks the token for
the connection (`/whoami`, cached 30 s) and for each call. Thus there is only
one place that checks tokens.

**Python** — pass `credential=` to the verbs that take it:

```python
from kalman.api import fetch, register_model
fetch({"kind": "file", "init": {"path": "run.npz"}}, credential=TOKEN)
register_model(spec, "demo_cv", credential=TOKEN)
```

---

## Behaviour in detail

```python
from kalman.api.auth import authorize, AuthError

authorize(token, "read")            # True, or raises AuthError
```

- **No secret configured** → returns `True` immediately. Auth is off.
- **Secret configured, no token** → `AuthError: a bearer token with scope 'read' is required.`
- **Bad signature / expired / malformed** → `AuthError: invalid or expired token: ExceptionType.`
  The underlying exception type is named; its message is not echoed.
- **Valid token, wrong scope** → `AuthError: token lacks required scope 'write_models'.`
- **Secret configured but PyJWT missing** → denied, not bypassed:
  `AuthError: auth is enabled but PyJWT is not installed`.

That last one is the fail-closed property that matters most. A missing optional
dependency degrades every other feature gracefully; here it denies instead.

Token payload:

```json
{"sub": "marco", "scopes": ["read", "send_measurement"], "iat": …, "exp": …}
```

`exp` and `sub` are required. A token without them is refused.

**Issuer and audience.** Set `KALMAN_JWT_ISSUER` and/or `KALMAN_JWT_AUDIENCE`
on the machine that mints and on the servers. New tokens then contain
`iss`/`aud`, and the servers refuse tokens without the correct values. Use this
when two deployments (for example staging and production) must not accept each
other's tokens.

**A server does not start open on a network.** `python -m kalman.rest` and
`python -m kalman.mcp` (HTTP transports) do not start in these conditions:

- auth is off and the server listens on an address other than loopback;
- the secret is shorter than 32 bytes (HS256 needs at least that).

Set `KALMAN_ALLOW_ANONYMOUS=1` to override the first rule, but only on a
network you trust. A server on `127.0.0.1` with auth off is the local user's
own server, and it starts as before.

**Ownership.** A model or source in the shared registry records the `sub` of
the token that created it. Only that user or an `admin` can replace or delete
it. An entry created while auth was off has no owner. Only an admin can change
it.

**Providers that reach outside the server.** A raw `file`, `remote` or `sensor`
provider in `fetch` needs the `admin` scope, and a source of these kinds also
needs `admin` to set up. Without this rule, a `send_measurement` token could
read any file on the server, or make the server send its outbound credential to
a host that the caller chooses. Other users fetch these sources by name.

There is no revocation list. A leaked token is valid until it expires — mint
short-lived tokens for anything you hand out.

---

## Practical setups

**Laptop, single user.** Leave auth off. Nothing to configure.

**Shared server or container.** Set `KALMAN_JWT_SECRET`, create accounts, mint
per-consumer tokens with the narrowest scopes that work. Terminate TLS in front
of the API — the tokens are bearer credentials and travel in a header.

**An agent that may run but not author.** `SCOPES=read,send_measurement`. It can
introspect, filter, forecast and fetch, but `register_model` and `scaffold_app`
are denied.

**A model author** (a person who publishes models from C++/ROOT or Python):
`SCOPES=read,send_measurement,write_models`. Their models are shared with all
users, and only they (or an admin) can change them. Loading a compiled plugin
also needs `load_plugins`, because a plugin runs code on the server.

**A game client or embedded consumer** (see [REST API](10_REST-API.md)).
`SCOPES=read,send_measurement`: it can push rows to a source and filter them.
Use a short expiry, and never an `admin` token. Anything in a build is public.

**CI.** `--no-verify` with the secret from the runner's own secret store.

---

## Rules

- Never commit secrets. `.env`, `.cdsapirc` and credential files are gitignored
  and must stay that way.
- Secrets are never logged, and no code path prints a token.
- Inbound auth (`api/auth.py`) and outbound credentials
  (`providers/credentials.py`) stay separate — a caller's token is never used to
  reach a third-party service.

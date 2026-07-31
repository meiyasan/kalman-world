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

`401` when no token was presented, `403` when the token lacks the scope.

**MCP** — a `token` argument on the tool call:

```json
{"name": "run_filter", "arguments": {"spec": {…}, "data": [...], "token": "eyJ…"}}
```

Register the client with the token baked into its launch environment so the
agent never has to handle it:

```bash
$ python tools/mcp_install.py claude-code --token "$TOKEN" --jwt-secret "$KALMAN_JWT_SECRET"
```

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

**A game client or embedded consumer** (see [REST API](10_REST-API.md)).
`SCOPES=send_measurement`, short expiry, and never an `admin` token — anything
shipped in a build is public.

**CI.** `--no-verify` with the secret from the runner's own secret store.

---

## Rules

- Never commit secrets. `.env`, `.cdsapirc` and credential files are gitignored
  and must stay that way.
- Secrets are never logged, and no code path prints a token.
- Inbound auth (`api/auth.py`) and outbound credentials
  (`providers/credentials.py`) stay separate — a caller's token is never used to
  reach a third-party service.

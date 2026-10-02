"""
Scoped-token auth for the API facade / MCP / REST — fail-closed.

Two layers: a ``.htpasswd`` **account store** (Apache ``name:bcrypt``, with the
scopes an account may hold in the optional third field that Apache and nginx
treat as a comment) that authenticates users to *mint* tokens — on the admin's
machine with the CLI, or by the user over HTTP with :func:`login` — and
**signed JWT bearer tokens** callers present at runtime carrying fine-grained
**scopes**:

  - ``read``            — introspection (catalog/describe/diagnostics/plots)
  - ``send_measurement``— execution (run_filter/forecast/osse/control, fetch)
  - ``write_models``    — authoring (register_model/save_spec/scaffold)
  - ``load_plugins``    — loading compiled model libraries (load_plugin); this
                          one runs code, which is why it is not ``write_models``
  - ``admin``           — everything + account management

Each verb declares its required scope; :func:`authorize` verifies the JWT
(HS256, ``KALMAN_JWT_SECRET``) and the scope.  **Auth disabled** (no secret
configured) allows in-process Python use (the local user already has the box);
when a secret *is* set, a valid token is required and anything missing/invalid/
expired is denied.  Optional deps ``pyjwt`` + ``bcrypt`` (``kalman[auth]``),
lazily imported; secrets are never logged.

Tokens carry ``sub`` (the user), ``scopes``, ``iat``, ``exp`` and ``jti`` (the
token's own id).  A token can be revoked before it expires — one token by its
``jti``, or every token of a user issued so far (:func:`revoke`); the list is
in the shared registry, so every server on it refuses the token at once.  Set
``KALMAN_JWT_ISSUER`` and/or ``KALMAN_JWT_AUDIENCE`` to also stamp and require
``iss``/``aud``, so a token minted for one deployment is refused by another
that shares the secret.  A server that listens on a network interface refuses
to start without auth (:func:`check_server`).

CLI (``python -m kalman.api.auth``): ``adduser`` / ``token`` / ``verify`` /
``revoke``.
"""
from __future__ import annotations
import os
import time

SCOPES = ("read", "send_measurement", "write_models", "load_plugins", "admin")


class AuthError(PermissionError):
    """Raised on missing/invalid/expired token or insufficient scope."""


class InvalidToken(AuthError):
    """No token, or one that is forged, expired, malformed or revoked (HTTP
    401) — as opposed to a good token that lacks the scope (HTTP 403)."""


def _secret() -> str | None:
    return os.environ.get("KALMAN_JWT_SECRET") or None


def _htpasswd_path() -> str:
    return os.environ.get("KALMAN_HTPASSWD", os.path.expanduser("~/.kalman/htpasswd"))


def enabled() -> bool:
    """True when auth is configured (a JWT secret is present)."""
    return _secret() is not None


#: HS256 needs a key at least as long as its output (RFC 7518 §3.2).
MIN_SECRET_BYTES = 32


def _decode(credential: str) -> dict:
    """Verify signature, expiry and (when configured) issuer/audience."""
    try:
        import jwt
    except ImportError as e:                     # fail-closed
        raise AuthError("auth is enabled but PyJWT is not installed "
                        "(pip install 'kalman[auth]').") from e
    iss = os.environ.get("KALMAN_JWT_ISSUER") or None
    aud = os.environ.get("KALMAN_JWT_AUDIENCE") or None
    try:
        payload = jwt.decode(credential, _secret(), algorithms=["HS256"],
                             issuer=iss, audience=aud,
                             options={"require": ["exp", "sub"]})
    except Exception as e:                       # invalid signature / expired / malformed
        raise InvalidToken(f"invalid or expired token: {type(e).__name__}.") from e
    try:
        from . import store
        revoked = store.is_revoked(payload.get("jti"), payload.get("sub"), payload.get("iat"))
    except Exception as e:                       # fail-closed: no list, no entry
        raise InvalidToken(f"cannot read the revocation list: {type(e).__name__}.") from e
    if revoked:
        raise InvalidToken("this token was revoked.")
    return payload


def claims(credential: str | None) -> dict | None:
    """The verified claims of ``credential`` with no scope check, or ``None``
    when auth is disabled. Raises :class:`AuthError` on a bad token."""
    if not enabled():
        return None
    if not credential:
        raise InvalidToken("a bearer token is required.")
    return _decode(credential)


def identity(credential: str | None) -> str | None:
    """Who is calling: the token's ``sub``, or ``None`` when auth is disabled."""
    c = claims(credential)
    return None if c is None else c.get("sub")


def has_scope(credential: str | None, scope: str) -> bool:
    try:
        return authorize(credential, scope)
    except AuthError:
        return False


def require_owner(record: dict, credential: str | None, what: str):
    """Allow a change to a shared record only for its owner or an admin.

    With auth disabled everybody is the local user, so anything goes. A record
    created while auth was off has no owner; only an admin may then change it."""
    if not enabled():
        return
    who = identity(credential)
    if has_scope(credential, "admin"):
        return
    if record.get("owner") is not None and record.get("owner") == who:
        return
    raise AuthError(f"{what} belongs to {record.get('owner') or 'nobody'}; "
                    "only its owner or an admin may change it.")


_LOOPBACK = ("127.0.0.1", "localhost", "::1")


def check_server(host: str, *, service: str, verified_elsewhere: bool = False):
    """Refuse to serve an open API on a network interface.

    A server bound to loopback is the local user's own; one bound to anything
    else (``0.0.0.0``, a LAN address, a container port) must authenticate,
    unless ``KALMAN_ALLOW_ANONYMOUS=1`` says the network itself is trusted.
    ``verified_elsewhere`` is for the MCP server in front of the REST API: the
    REST API checks the tokens, so the MCP server needs no secret of its own."""
    if enabled():
        if len(_secret().encode()) < MIN_SECRET_BYTES:
            raise AuthError(f"{service}: KALMAN_JWT_SECRET is shorter than "
                            f"{MIN_SECRET_BYTES} bytes; HS256 needs at least that "
                            "(python -c 'import secrets; print(secrets.token_urlsafe(48))').")
        return
    if verified_elsewhere or host in _LOOPBACK:
        return
    if os.environ.get("KALMAN_ALLOW_ANONYMOUS") == "1":
        import sys
        print(f"WARNING: {service} listens on {host} WITHOUT authentication "
              "(KALMAN_ALLOW_ANONYMOUS=1). Anybody who can reach it can register "
              "models and load plugins.", file=sys.stderr)
        return
    raise AuthError(f"{service}: refusing to listen on {host} without authentication. "
                    "Set KALMAN_JWT_SECRET (>= 32 bytes) and give clients a token from "
                    "`make api-token`, bind to 127.0.0.1, or set KALMAN_ALLOW_ANONYMOUS=1 "
                    "on a network you trust.")


# ---------------------------------------------------------------------------
# Runtime authorization (inbound)
# ---------------------------------------------------------------------------
def authorize(credential: str | None, scope: str) -> bool:
    """Authorize ``credential`` (a JWT) for ``scope``; raise :class:`AuthError`
    if denied.  No-op (returns True) when auth is disabled."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r} (choose from {SCOPES}).")
    secret = _secret()
    if secret is None:
        return True                              # auth disabled — local dev
    if not credential:
        raise InvalidToken(f"a bearer token with scope '{scope}' is required.")
    payload = _decode(credential)
    scopes = set(payload.get("scopes", []))
    if scope in scopes or "admin" in scopes:
        return True
    raise AuthError(f"token lacks required scope '{scope}'.")


# ---------------------------------------------------------------------------
# Token minting (outbound, admin-side) + account store
# ---------------------------------------------------------------------------
def mint_token(user: str, scopes, *, secret: str | None = None, exp_hours: float = 24.0) -> str:
    """Sign a scoped JWT for ``user``.  ``secret`` defaults to ``KALMAN_JWT_SECRET``."""
    import jwt
    secret = secret or _secret()
    if not secret:
        raise AuthError("set KALMAN_JWT_SECRET to mint tokens.")
    scopes = [s.strip() for s in (scopes.split(",") if isinstance(scopes, str) else scopes) if s]
    bad = [s for s in scopes if s not in SCOPES]
    if bad:
        raise ValueError(f"unknown scopes {bad}; choose from {SCOPES}.")
    import math, uuid
    t = time.time()
    now = int(t)
    # iat to the millisecond (rounded down, so it is never in the future): a
    # "revoke everything until now" must not catch the token of the login that
    # follows it in the same second.
    payload = {"sub": user, "scopes": scopes, "iat": math.floor(t * 1000) / 1000,
               "exp": now + int(exp_hours * 3600), "jti": uuid.uuid4().hex}
    if os.environ.get("KALMAN_JWT_ISSUER"):
        payload["iss"] = os.environ["KALMAN_JWT_ISSUER"]
    if os.environ.get("KALMAN_JWT_AUDIENCE"):
        payload["aud"] = os.environ["KALMAN_JWT_AUDIENCE"]
    return jwt.encode(payload, secret, algorithm="HS256")


# bcrypt caps the password at 72 bytes and (since 5.0) refuses longer input
# outright, so truncate explicitly rather than let it raise.
_BCRYPT_MAX = 72


def _bcrypt():
    try:
        import bcrypt
        return bcrypt
    except ImportError as e:
        raise AuthError("account management needs bcrypt "
                        "(pip install 'kalman[auth]').") from e


def _clean_scopes(scopes) -> list[str]:
    scopes = [s.strip() for s in (scopes.split(",") if isinstance(scopes, str) else scopes)
              if s and s.strip()]
    bad = [s for s in scopes if s not in SCOPES]
    if bad:
        raise ValueError(f"unknown scopes {bad}; choose from {SCOPES}.")
    return scopes


def _accounts(path: str | None = None) -> dict:
    """``{user: (digest, granted scopes or None)}`` from the account store."""
    path = path or _htpasswd_path()
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh.read().splitlines():
                if ":" not in line or line.lstrip().startswith("#"):
                    continue
                name, digest, *rest = line.split(":", 2)
                granted = [s for s in rest[0].split(",") if s] if rest and rest[0].strip() else None
                out[name] = (digest, granted)
    return out


def add_user(user: str, password: str, path: str | None = None, scopes=None) -> str:
    """Add/update a bcrypt ``.htpasswd`` account; returns the file path.

    Written in the Apache ``user:$2b$…`` format that httpd, nginx and every
    htpasswd tool understand.  Uses the maintained ``bcrypt`` package directly
    — ``passlib`` is unmaintained and breaks against bcrypt >= 5.

    ``scopes`` is what the account may ask for when it logs in
    (:func:`login`), kept in the third field, which those tools read as a
    comment.  ``None`` keeps the account's present grant.
    """
    bc = _bcrypt()
    if ":" in user or "\n" in user or not user:
        raise ValueError("username must be non-empty and must not contain ':' or a newline")
    path = path or _htpasswd_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    digest = bc.hashpw(password.encode()[:_BCRYPT_MAX], bc.gensalt()).decode()
    granted = _clean_scopes(scopes) if scopes is not None else _accounts(path).get(user, (None, None))[1]

    lines = []
    if os.path.exists(path):
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines()
                     if ln.strip() and ln.split(":", 1)[0] != user]
    lines.append(f"{user}:{digest}" + (f":{','.join(granted)}" if granted else ""))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)                       # credentials file: owner-only
    return path


def verify_user(user: str, password: str, path: str | None = None) -> bool:
    """Check ``password`` against the account store (constant-time via bcrypt)."""
    bc = _bcrypt()
    digest = _accounts(path).get(user, (None, None))[0]
    if not digest:
        return False
    # Apache writes $2y$; it is bit-identical to $2b$ for verification.
    if digest.startswith("$2y$"):
        digest = "$2b$" + digest[4:]
    if not digest.startswith("$2"):
        raise AuthError(f"account {user!r} uses a non-bcrypt hash; "
                        f"re-add it with `make api-user USER={user}`.")
    try:
        return bc.checkpw(password.encode()[:_BCRYPT_MAX], digest.encode())
    except ValueError:
        return False


def user_scopes(user: str, path: str | None = None) -> list[str]:
    """What ``user`` may ask for at login: the account's grant, or
    ``KALMAN_DEFAULT_SCOPES`` (default ``read``) for an account without one."""
    granted = _accounts(path).get(user, (None, None))[1]
    if granted is None:
        granted = _clean_scopes(os.environ.get("KALMAN_DEFAULT_SCOPES", "read"))
    return [s for s in granted if s in SCOPES]


# ---------------------------------------------------------------------------
# Login over the API, and revocation
# ---------------------------------------------------------------------------
class BadCredentials(AuthError):
    """Wrong user or password (HTTP 401; the two are not told apart)."""


class RateLimited(AuthError):
    """Too many failed logins; ``retry_after`` seconds until the next try."""

    def __init__(self, retry_after: float):
        super().__init__(f"too many failed logins; try again in {int(retry_after) + 1} s.")
        self.retry_after = retry_after


#: Failed logins allowed per (user, client) inside the window, in seconds.
LOGIN_ATTEMPTS, LOGIN_WINDOW = 5, 300.0
_FAILS: dict = {}
_DUMMY: list = []


def _throttle(key) -> list:
    now = time.monotonic()
    fails = [t for t in _FAILS.get(key, []) if now - t < LOGIN_WINDOW]
    if len(fails) >= LOGIN_ATTEMPTS:
        raise RateLimited(LOGIN_WINDOW - (now - fails[0]))
    return fails


def login(user: str, password: str, scopes=None, *, exp_hours: float | None = None,
          client: str = "") -> dict:
    """Exchange an account's password for a scoped JWT.

    The token carries the scopes asked for, which must all be in the account's
    grant (an ``admin`` grant covers any); with ``scopes=None`` it carries the
    whole grant.  Its lifetime is at most ``KALMAN_TOKEN_MAX_HOURS`` (default
    24).  Five failed tries per user and client lock that pair out for five
    minutes, and an unknown user costs the same bcrypt check as a wrong
    password, so the answer's timing does not say which accounts exist."""
    if not enabled():
        raise ValueError("authentication is off (no KALMAN_JWT_SECRET); no token is needed.")
    if not isinstance(user, str) or not isinstance(password, str) or not user:
        raise ValueError("login needs a user and a password.")
    key = (user, client)
    fails = _throttle(key)
    ok = verify_user(user, password)
    if not ok and user not in _accounts():
        bc = _bcrypt()
        if not _DUMMY:
            _DUMMY.append(bc.hashpw(b"no-such-account", bc.gensalt()))
        bc.checkpw(password.encode()[:_BCRYPT_MAX], _DUMMY[0])
    if not ok:
        _FAILS[key] = fails + [time.monotonic()]
        if len(_FAILS) > 10000:                  # a bound on what guessing can make us keep
            _FAILS.clear()
        raise BadCredentials("wrong user or password.")
    _FAILS.pop(key, None)

    granted = user_scopes(user)
    asked = _clean_scopes(scopes) if scopes else granted
    if "admin" not in granted:
        denied = [s for s in asked if s not in granted]
        if denied:
            raise AuthError(f"account {user!r} is not granted {denied}; it has {granted}.")
    if not asked:
        raise AuthError(f"account {user!r} has no scopes.")
    cap = float(os.environ.get("KALMAN_TOKEN_MAX_HOURS", "24"))
    hours = cap if exp_hours is None else max(min(float(exp_hours), cap), 1.0 / 60.0)
    token = mint_token(user, asked, exp_hours=hours)
    return {"access_token": token, "token_type": "bearer", "expires_in": int(hours * 3600),
            "scopes": asked, "user": user}


def revoke(credential: str | None = None, *, user: str | None = None) -> dict:
    """Stop a token from working before it expires.

    With no ``user``: the presented token itself (a logout; any valid token may
    do this).  With ``user``: every token of that user issued until now — the
    user's own, or anybody's for an ``admin``.  The list is in the shared
    registry, so every server on it refuses the token from the next request."""
    from . import store
    c = claims(credential)
    if c is None:
        raise ValueError("authentication is off; there are no tokens to revoke.")
    if user is None:
        if not c.get("jti"):
            raise ValueError("this token has no id (jti); revoke by user instead.")
        store.revoke_token(c["jti"], c.get("sub"), c.get("exp"), c.get("sub"))
        return {"revoked": "token", "user": c.get("sub")}
    if user != c.get("sub") and "admin" not in c.get("scopes", []):
        raise AuthError("only an admin may revoke another user's tokens.")
    cutoff = store.revoke_user(user, c.get("sub"))
    return {"revoked": "user", "user": user, "before": cutoff}


# ---------------------------------------------------------------------------
# CLI  (python -m kalman.api.auth ...)
# ---------------------------------------------------------------------------
def _main(argv=None):
    import argparse, getpass, sys
    p = argparse.ArgumentParser(prog="kalman.api.auth", description="Kalman API auth")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adduser"); a.add_argument("--user", required=True); a.add_argument("--htpasswd")
    a.add_argument("--scopes", help="what the account may ask for at login, e.g. read,send_measurement")
    r = sub.add_parser("revoke", help="revoke one token, or every token of a user issued so far")
    r.add_argument("--token"); r.add_argument("--user")
    t = sub.add_parser("token"); t.add_argument("--user", required=True)
    t.add_argument("--scopes", default="read"); t.add_argument("--exp", type=float, default=24.0)
    t.add_argument("--no-verify", action="store_true", help="mint without checking the htpasswd account")
    v = sub.add_parser("verify"); v.add_argument("--token", required=True); v.add_argument("--scope", default="read")
    args = p.parse_args(argv)

    if args.cmd == "adduser":
        pw = getpass.getpass(f"password for {args.user}: ")
        path = add_user(args.user, pw, args.htpasswd, scopes=args.scopes or None)
        print(f"account '{args.user}' written to {path} "
              f"(login scopes: {','.join(user_scopes(args.user, path))})")
    elif args.cmd == "revoke":
        # Run where the servers' KALMAN_STATE_DIR is: the list is in the registry.
        from . import store
        if args.user:
            store.revoke_user(args.user, "cli")
            print(f"every token of '{args.user}' issued until now is revoked")
        elif args.token:
            c = claims(args.token)
            if not c or not c.get("jti"):
                print("this token has no id (jti); use --user", file=sys.stderr); return 1
            store.revoke_token(c["jti"], c.get("sub"), c.get("exp"), "cli")
            print(f"token {c['jti']} of '{c.get('sub')}' is revoked")
        else:
            print("give --token or --user", file=sys.stderr); return 1
    elif args.cmd == "token":
        if not args.no_verify:
            pw = getpass.getpass(f"password for {args.user}: ")
            if not verify_user(args.user, pw):
                print("authentication failed", file=sys.stderr); return 1
        print(mint_token(args.user, args.scopes, exp_hours=args.exp))
    elif args.cmd == "verify":
        try:
            authorize(args.token, args.scope); print("OK")
        except AuthError as e:
            print(f"DENIED: {e}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

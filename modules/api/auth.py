"""
Scoped-token auth for the API facade / MCP / REST — fail-closed.

Two layers: a ``.htpasswd`` **account store** (Apache ``name:bcrypt``) that
authenticates users to *mint* tokens, and **signed JWT bearer tokens** callers
present at runtime carrying fine-grained **scopes**:

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

CLI (``python -m kalman.api.auth``): ``adduser`` / ``token`` / ``verify``.
"""
from __future__ import annotations
import os
import time

SCOPES = ("read", "send_measurement", "write_models", "load_plugins", "admin")


class AuthError(PermissionError):
    """Raised on missing/invalid/expired token or insufficient scope."""


def _secret() -> str | None:
    return os.environ.get("KALMAN_JWT_SECRET") or None


def _htpasswd_path() -> str:
    return os.environ.get("KALMAN_HTPASSWD", os.path.expanduser("~/.kalman/htpasswd"))


def enabled() -> bool:
    """True when auth is configured (a JWT secret is present)."""
    return _secret() is not None


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
        raise AuthError(f"a bearer token with scope '{scope}' is required.")
    try:
        import jwt
    except ImportError as e:                     # fail-closed
        raise AuthError("auth is enabled but PyJWT is not installed "
                        "(pip install 'kalman[auth]').") from e
    try:
        payload = jwt.decode(credential, secret, algorithms=["HS256"])
    except Exception as e:                       # invalid signature / expired / malformed
        raise AuthError(f"invalid or expired token: {type(e).__name__}.") from e
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
    now = int(time.time())
    payload = {"sub": user, "scopes": scopes, "iat": now, "exp": now + int(exp_hours * 3600)}
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


def add_user(user: str, password: str, path: str | None = None) -> str:
    """Add/update a bcrypt ``.htpasswd`` account; returns the file path.

    Written in the Apache ``user:$2b$…`` format that httpd, nginx and every
    htpasswd tool understand.  Uses the maintained ``bcrypt`` package directly
    — ``passlib`` is unmaintained and breaks against bcrypt >= 5.
    """
    bc = _bcrypt()
    if ":" in user:
        raise ValueError("username must not contain ':'")
    path = path or _htpasswd_path()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    digest = bc.hashpw(password.encode()[:_BCRYPT_MAX], bc.gensalt()).decode()

    lines = []
    if os.path.exists(path):
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines()
                     if ln.strip() and ln.split(":", 1)[0] != user]
    lines.append(f"{user}:{digest}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(path, 0o600)                       # credentials file: owner-only
    return path


def verify_user(user: str, password: str, path: str | None = None) -> bool:
    """Check ``password`` against the account store (constant-time via bcrypt)."""
    bc = _bcrypt()
    path = path or _htpasswd_path()
    if not os.path.exists(path):
        return False
    for line in open(path).read().splitlines():
        name, _, digest = line.partition(":")
        if name != user or not digest:
            continue
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
    return False


# ---------------------------------------------------------------------------
# CLI  (python -m kalman.api.auth ...)
# ---------------------------------------------------------------------------
def _main(argv=None):
    import argparse, getpass, sys
    p = argparse.ArgumentParser(prog="kalman.api.auth", description="Kalman API auth")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adduser"); a.add_argument("--user", required=True); a.add_argument("--htpasswd")
    t = sub.add_parser("token"); t.add_argument("--user", required=True)
    t.add_argument("--scopes", default="read"); t.add_argument("--exp", type=float, default=24.0)
    t.add_argument("--no-verify", action="store_true", help="mint without checking the htpasswd account")
    v = sub.add_parser("verify"); v.add_argument("--token", required=True); v.add_argument("--scope", default="read")
    args = p.parse_args(argv)

    if args.cmd == "adduser":
        pw = getpass.getpass(f"password for {args.user}: ")
        path = add_user(args.user, pw, args.htpasswd)
        print(f"account '{args.user}' written to {path}")
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

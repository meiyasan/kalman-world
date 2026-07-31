"""Outbound credentials for remote providers — **distinct from the inbound JWT
auth** (:mod:`kalman.api.auth`).  These are the framework's own secrets for
reaching remote services (CDS key, API tokens), resolved server-side from the
environment or ``~/.kalman/credentials.json`` — never from a caller's token.

Fail-closed: a remote provider that needs a credential and can't find one raises
``PermissionError``.  Secrets are returned to the caller code only, never logged.
"""
from __future__ import annotations
import json
import os

_STORE = os.path.expanduser("~/.kalman/credentials.json")


def get_credential(name: str, *, env: str | None = None, required: bool = True) -> str | None:
    """Resolve outbound secret ``name`` from ``$env`` (default
    ``KALMAN_<NAME>_KEY``) then ``~/.kalman/credentials.json``."""
    env = env or f"KALMAN_{name.upper()}_KEY"
    val = os.environ.get(env)
    if val is None and os.path.exists(_STORE):
        try:
            val = json.load(open(_STORE)).get(name)
        except Exception:
            val = None
    if val is None and required:
        raise PermissionError(f"missing outbound credential '{name}' (set ${env} "
                              f"or add it to {_STORE}).")
    return val

"""
``KalmanSpec`` authoring — the validated pipeline behind :func:`register_model`.

An authored spec goes through: **(1)** structural validation → **(2)** framework
validation (shapes / symmetric-PSD Q,R / dim-consistency, via instantiation +
``_check_shapes``) → **(3)** a one-step smoke gate ("does it actually run") →
**(4)** registration in the shared store (:mod:`.store`), so ``catalog()`` and
``make_model`` see it in every process on the same state directory, and
subscribers get a ``model.created`` event → **(5)** optional :func:`save_spec`
to a ``specs/`` dir (reloadable).  Authoring is gated
by the ``write_models`` scope (:mod:`kalman.api.auth`).

**No code execution**: specs are pure data; only names already in the catalog
(or the ``matrices``/``continuous`` declarative bases) are accepted; anything
else raises ``ValueError``.
"""
from __future__ import annotations
import json
import os
import numpy as np

_DECLARATIVE = ("matrices", "continuous")


def _psd_ok(M, name):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square; got shape {M.shape}.")
    if not np.allclose(M, M.T, atol=1e-8):
        raise ValueError(f"{name} must be symmetric.")
    if np.linalg.eigvalsh(0.5 * (M + M.T)).min() < -1e-8:
        raise ValueError(f"{name} must be positive semi-definite.")


def _validate_structure(spec: dict):
    if not isinstance(spec, dict):
        raise ValueError("spec must be a dict.")
    base = spec.get("base", "matrices")
    from . import _model_catalog, _MODELS
    if base not in _DECLARATIVE and base not in _model_catalog() and base not in _MODELS:
        raise ValueError(f"unknown base {base!r}; use one of {_DECLARATIVE} or a catalog model.")
    noise = spec.get("noise", {}) or {}
    for k in ("Q", "R"):
        if noise.get(k) is not None:
            _psd_ok(noise[k], k)
    # reject obviously non-spec / code-bearing values in the declarative dynamics
    if base in _DECLARATIVE:
        for k, v in (spec.get("dynamics", {}) or {}).items():
            if v is not None:
                try:
                    np.asarray(v, dtype=float)
                except (TypeError, ValueError):
                    raise ValueError(f"dynamics[{k!r}] must be numeric arrays, not {type(v).__name__}.")


def register_model(spec: dict, name: str | None = None, *, replace: bool = False,
                   credential: str | None = None) -> dict:
    """Validate, smoke-test, and register an authored model spec in the shared
    registry.

    Returns ``{"name", "n", "m", "l", "owner", "action"}``; ``action`` is
    ``created``, ``replaced`` or ``unchanged`` (the same spec again is a no-op).
    A different spec under a taken name needs ``replace=True``, and then the
    caller must own the entry or be an admin.  A name may not shadow a catalog
    model: everybody who asks for ``constant-velocity`` must get the same one.
    Requires the ``write_models`` scope when auth is enabled."""
    from .auth import authorize, identity, require_owner
    authorize(credential, "write_models")

    _validate_structure(spec)
    from . import make_model, _model_catalog, _check_name
    # An unnamed spec is named after its content: the same spec always lands
    # on the same entry, where a counter would differ from process to process.
    import hashlib
    name = name or spec.get("name") or "authored-" + hashlib.sha1(
        json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:10]
    _check_name(name, "model")
    if name in _DECLARATIVE or name in _model_catalog():
        raise ValueError(f"{name!r} is a catalog model; choose another name for yours.")
    model = make_model(spec)                      # framework validation via _check_shapes
    # smoke gate: the dynamics and observation must actually evaluate
    x0 = np.asarray(model.x0, dtype=float)
    _ = model.f(x0)
    _ = model.h(x0)
    if model.fs:
        _ = model.sample(2.0 / model.fs)          # a couple of steps

    from . import store
    info = {"n": int(model.n), "m": int(model.m), "l": int(model.l)}
    rec, action = store.put_model(
        name, dict(spec), info, identity(credential), replace=replace,
        may_replace=lambda r: require_owner(r, credential, f"model {name!r}"))
    return {"name": name, **info, "owner": rec["owner"], "action": action}


def save_spec(name: str, directory: str | None = None, *, credential: str | None = None) -> str:
    """Persist a registered spec as JSON under ``directory`` (default ``./specs``).
    Reloadable via :func:`load_spec`.  Requires ``write_models``."""
    from .auth import authorize
    authorize(credential, "write_models")
    from . import _MODELS
    if name not in _MODELS:
        raise KeyError(f"no registered spec named {name!r}; register_model first.")
    directory = directory or os.path.join(os.getcwd(), "specs")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w") as fh:
        json.dump(_MODELS[name], fh, indent=2)
    return path


def load_spec(path: str, *, register: bool = True, credential: str | None = None) -> dict:
    """Load a spec JSON from disk; optionally re-register it (``write_models``)."""
    with open(path) as fh:
        spec = json.load(fh)
    if register:
        register_model(spec, name=os.path.splitext(os.path.basename(path))[0], credential=credential)
    return spec

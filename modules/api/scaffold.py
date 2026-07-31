"""
Scaffold a reviewable app module from a ``KalmanSpec``.

:func:`scaffold` returns **Python source** for a thin ``modules/apps/<name>.py``
— the path from an LLM-authored spec to a real app, with a *human* as the commit
gate.  It only *emits* source (never ``exec``s it); :func:`write_scaffold` writes
it to a human-chosen path, and neither auto-imports nor installs it.  Gated by
the ``write_models`` scope.
"""
from __future__ import annotations
import os
import numpy as np


def _fmt(v):
    return "None" if v is None else repr(np.asarray(v, dtype=float).tolist())


def scaffold(spec: dict, class_name: str | None = None, *, credential: str | None = None) -> str:
    """Return Python source for a ``KalmanModel``/``LinearKalmanModel`` subclass
    built from ``spec`` (matrices or continuous-linear bases)."""
    from .auth import authorize
    authorize(credential, "write_models")

    spec = dict(spec or {})
    base = spec.get("base", "matrices")
    if base not in ("matrices", "continuous"):
        raise ValueError("scaffold supports the 'matrices'/'continuous' declarative bases; "
                         "parameterized catalog models are already importable.")
    cls = class_name or spec.get("name") or "AuthoredModel"
    cls = "".join(w[:1].upper() + w[1:] for w in str(cls).replace("-", "_").split("_") if w)
    dyn = spec.get("dynamics", {}) or {}
    noise = spec.get("noise", {}) or {}
    init = spec.get("initial", {}) or {}
    fs = spec.get("fs", 1.0)
    sensor = spec.get("sensor_id", "S")

    kwargs = []
    if base == "matrices":
        for k in ("F", "H", "B", "L", "M"):
            if dyn.get(k) is not None:
                kwargs.append(f"{k}=np.array({_fmt(dyn[k])})")
    else:
        for k in ("A", "Qc", "Lc"):
            if dyn.get(k) is not None:
                kwargs.append(f"{k}=np.array({_fmt(dyn[k])})")
    for k in ("Q", "R"):
        if noise.get(k) is not None:
            kwargs.append(f"{k}=np.array({_fmt(noise[k])})")
    for k in ("x0", "P0"):
        if init.get(k) is not None:
            kwargs.append(f"{k}=np.array({_fmt(init[k])})")
    kwargs.append(f"fs={fs}")
    args = ",\n            ".join(kwargs)
    parent = "LinearKalmanModel" if base == "matrices" else "KalmanModel"

    return f'''"""Scaffolded from a KalmanSpec by kalman.api.scaffold — review before committing."""
import numpy as np
from kalman.framework import {parent}


class {cls}({parent}):
    """Authored model (base={base!r})."""
    NAME = {cls!r}
    DESCRIPTION = "Scaffolded from a KalmanSpec."
    VERSION = "0.1"

    def __init__(self, sensor_id={sensor!r}, **kw):
        super().__init__(
            sensor_id,
            {args},
            **kw,
        )
'''


def write_scaffold(spec: dict, path: str, class_name: str | None = None,
                   *, credential: str | None = None) -> str:
    """Write :func:`scaffold` source to ``path`` (a human-chosen staging location,
    NOT auto-imported).  Requires ``write_models``."""
    src = scaffold(spec, class_name, credential=credential)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(src)
    return path

"""kalman.native - the C++ side of the framework, in one place.

C++ is the implementation and Python the interface: the classes of
``kalman.framework`` forward to ``libKalman`` on their own, so most code never
imports this module. It is for the cases where the C++ side is addressed
directly::

    import kalman.native as native

    native.available()                    # is libKalman loaded?
    native.status()                       # which libraries, where from
    native.load_plugin("KalmanPlugin_toy")   # register a C++ model library
    model = native.NativeModel("toy")     # a model that lives in C++
    engine = native.NativeEngine("extended")

    with native.python_reference():       # the Python reference path, for comparisons
        ...

Nothing is implemented here; every name comes from ``kalman.framework``.
"""
from __future__ import annotations

import os

from .framework.native import core as _core
from .framework.models.native import (NativeModel, NativeTwin, python_reference, native_model_classes,
                                      run_callback_filter)
from .framework.engines.native import NativeEngine

__all__ = ["available", "status", "version", "names", "describe", "plugins", "load_plugin",
           "NativeModel", "NativeTwin", "NativeEngine", "native_model_classes", "run_callback_filter",
           "python_reference", "export_model", "read_model_file"]


def available() -> bool:
    """True when libKalman is loaded."""
    return bool(_core.HAS_CXX_CORE)


def version() -> str:
    """The version of the loaded libKalman."""
    return _core.version()


def names() -> list:
    """The models registered with the C++ factory."""
    return _core.names()


def describe(name: str) -> dict:
    """What the C++ factory knows about the model ``name``."""
    return _core.describe(name)


def plugins() -> list:
    """The plugin libraries loaded so far."""
    return _core.plugins()


def load_plugin(name_or_path: str) -> list:
    """Load a C++ model library, by name from the allowed plugin directories or by
    a path inside them, and return the model names it registered."""
    return _core.load_plugin(_core.resolve_plugin(name_or_path))


def export_model(name: str, path: str = None, *, linearize: bool = False, steps: int = 0,
                 parameters: dict = None) -> dict:
    """Write the registered C++ model ``name`` out and return its ``model.json`` as a dict.

    ``path`` ending in ``.kw`` gives one archive (``model.json`` and the tables
    it refers to), another name ending in ``.json`` plain JSON; with no path nothing
    is written. (``name.kw.json`` is the settings file of ``name.kw``, not a model.)
    A model of matrices is written whole (``kind`` "complete"). A model given by
    code is written as a "reference" to its name and settings, or, with
    ``linearize``, as its Jacobians at the initial state. A model whose matrices
    change with time needs ``steps``, the number of samples to tabulate, and a
    ``.kw`` path. ``parameters`` sets named parameters first.

    ``result["spec"]`` is what :mod:`kalman.api` takes.
    """
    import json
    L = _core.require()
    ffi = _core.ffi
    handle = L.kalman_model_create(name.encode())
    if handle == ffi.NULL:
        raise ValueError(f"export_model({name!r}): {_core.last_error()}")
    try:
        for key, value in (parameters or {}).items():
            _core.check(L.kalman_model_set_parameter(handle, str(key).encode(), float(value)),
                        f"export_model({name!r}): parameter {key!r}")
        library = os.path.basename(_core.describe(name)["plugin"] or "")
        plugin = (library[3:] if library.startswith("lib") else library).split(".")[0]
        args = (name.encode(), plugin.encode() if plugin else ffi.NULL, int(bool(linearize)), int(steps))
        if path is not None:
            _core.check(L.kalman_model_save(handle, os.fspath(path).encode(), *args), f"export_model({name!r})")
            return read_model_file(path)["document"]
        text = L.kalman_model_export_json(handle, *args)
        if text == ffi.NULL:
            raise ValueError(f"export_model({name!r}): {_core.last_error()}")
        return json.loads(ffi.string(text).decode())
    finally:
        L.kalman_model_destroy(handle)


def read_model_file(path: str) -> dict:
    """Read a model file: ``{"document": <model.json as a dict>, "tables": {name: array},
    "settings": <the settings file beside it as a dict, or None>}``.

    Takes the archive (``.kw``) or plain JSON, told apart by content. The settings
    file is ``path + ".json"``: what is changed from the model file (``fs``,
    ``parameters``, ``noise``, ``initial``). It is returned, not applied: the C++
    side applies it when it opens the file (``kalman model``, ``kalman_model_load``). A table has the shape ``(steps, rows, columns)``: one matrix per
    sample for a matrix that changes with time. Needs no C++ library.
    """
    import json
    import zipfile
    import numpy as np
    tables = {}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            document = json.loads(archive.read("model.json"))
            for key, entry in ((document.get("tables") or {}).get("entries") or {}).items():
                tables[key] = np.frombuffer(archive.read(entry["file"]), dtype="<f8").reshape(entry["shape"])
    else:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    if document.get("format") != "kalman-model":
        raise ValueError(f"{path}: not a model file (\"format\" is not \"kalman-model\").")
    settings = None
    beside = os.fspath(path) + ".json"
    if os.path.isfile(beside):
        with open(beside, encoding="utf-8") as handle:
            settings = json.load(handle)
        if settings.get("format") != "kalman-settings":
            raise ValueError(f"{beside}: not a settings file (\"format\" is not \"kalman-settings\").")
    return {"document": document, "tables": tables, "settings": settings}


def status() -> dict:
    """Which C++ libraries this process uses: the core, the application models and
    the learned components, each with its path or the reason it is absent."""
    from .framework.native import learn
    from .framework.models import native as _native
    out = {"core": {"loaded": available(), "path": _core.LIBRARY_PATH, "error": _core.LOAD_ERROR,
                    "version": _core.version() if available() else None}}
    apps = sorted(_native.apps_library_names())
    out["apps"] = {"loaded": bool(apps), "models": apps}
    out["learn"] = {"loaded": learn.available(), "path": learn.LIBRARY_PATH, "error": learn.LOAD_ERROR,
                    "torch": learn.has_torch()}
    return out

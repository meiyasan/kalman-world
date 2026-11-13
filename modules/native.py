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
           "python_reference", "export_model", "read_model_file", "export", "open_model"]


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
        plugin = _plugin_of(name)
        args = (name.encode(), plugin.encode() if plugin else ffi.NULL, int(bool(linearize)), int(steps), ffi.NULL)
        if path is not None:
            _core.check(L.kalman_model_save(handle, os.fspath(path).encode(), *args), f"export_model({name!r})")
            return read_model_file(path)["document"]
        text = L.kalman_model_export_json(handle, *args)
        if text == ffi.NULL:
            raise ValueError(f"export_model({name!r}): {_core.last_error()}")
        return json.loads(ffi.string(text).decode())
    finally:
        L.kalman_model_destroy(handle)


def _plugin_of(name: str) -> str:
    library = os.path.basename(_core.describe(name)["plugin"] or "")
    return (library[3:] if library.startswith("lib") else library).split(".")[0]


def _tabulate(model, steps):
    """F, H, Q and R of a model whose matrices are functions of time, over
    ``steps`` samples: ``{name: array (steps, rows, columns)}``.

    Each function is called once per sample, sample after sample, in the order
    the filter's loop asks for them (F, Q, H, R) and at the times it reaches by
    adding dt. A function is meant to depend on t alone; one that keeps a count
    of its calls still sees the sequence it would in the loop."""
    import numpy as np
    n, m = int(model.n), int(model.m)
    t, dt = float(getattr(model, "t0", 0.0) or 0.0), 1.0 / float(model.fs or 1.0)
    out = {k: [] for k in "FQHR"}
    for _ in range(int(steps)):
        out["F"].append(np.asarray(model.F(t=t), dtype=float).reshape(n, n))
        out["Q"].append(np.asarray(model.Q(t=t), dtype=float).reshape(n, n))
        out["H"].append(np.asarray(model.H(t=t), dtype=float).reshape(m, n))
        out["R"].append(np.asarray(model.R(t=t), dtype=float).reshape(m, m))
        t += dt
    return {k: np.ascontiguousarray(np.stack(v)) for k, v in out.items()}


def _stand_in(model, structure, linearize, steps=0):
    """A C++ model of matrices (Kalman::Models::Tabulated) holding what the
    Python model ``model`` states: what the exporter writes, and what the filter
    runs for a model whose matrices change with time (``steps`` samples of them
    are then tabulated)."""
    import numpy as np
    from .framework.constants import KalmanStructureKind as S
    L, ffi = _core.require(), _core.ffi

    def flat(a, shape):
        return np.ascontiguousarray(np.asarray(a, dtype=float).reshape(shape))

    n, m, l = int(model.n), int(model.m), int(getattr(model, "l", 0) or 0)
    name = type(model).__name__
    raw = L.kalman_model_create_matrices(name.encode(), n, m, l)
    if raw == ffi.NULL:
        raise ValueError(f"export: {_core.last_error()}")
    handle = ffi.gc(raw, L.kalman_model_destroy)

    def put(fn, array, what, *args):
        _core.check(fn(handle, *args, _core.cdptr(array)), f"export: {what}")

    t0 = float(getattr(model, "t0", 0.0) or 0.0)
    tables = _tabulate(model, steps) if (structure is S.TIME_VARYING and steps and not linearize) else None
    first = (lambda k, call: tables[k][0]) if tables is not None else (lambda k, call: call())
    if getattr(model, "fs", None):
        _core.check(L.kalman_model_set_sample_rate(handle, float(model.fs)), "export: fs")
    _core.check(L.kalman_model_set_t0(handle, t0), "export: t0")
    x0 = flat(model.x0, (n,))
    put(L.kalman_model_set_x0, x0, "x0")
    if getattr(model, "P0", None) is not None:
        put(L.kalman_model_set_P0, flat(model.P0, (n, n)), "P0")
    put(L.kalman_model_set_Q, flat(first("Q", lambda: model.Q(t=t0)), (n, n)), "Q")
    put(L.kalman_model_set_R, flat(first("R", lambda: model.R(t=t0)), (m, m)), "R")

    for state, enum, count in ((1, model.states, n), (0, model.observables, m)):
        members = list(enum)
        if len(members) == count:       # one name per component, or none at all
            for i, member in enumerate(members):
                unit = member._value_ if isinstance(member._value_, str) else ""
                _core.check(L.kalman_model_set_component(handle, state, i, member._name_.lower().encode(),
                                                         unit.encode("ascii", "replace")), "export: components")

    if structure is S.MAPS and not linearize:
        _core.check(L.kalman_model_set_structure(handle, 0), "export: structure")
        return handle

    # The matrices: the model's own, or for a model given by code its Jacobians at x0.
    put(L.kalman_model_set_matrix, flat(model.F(x=model.x0, t=t0) if structure is S.MAPS
                                        else first("F", lambda: model.F(t=t0)), (n, n)), "F", b"F")
    put(L.kalman_model_set_matrix, flat(model.H(x=model.x0, t=t0) if structure is S.MAPS
                                        else first("H", lambda: model.H(t=t0)), (m, n)), "H", b"H")
    if structure is S.MAPS:
        return handle
    if tables is not None:
        # Only what the model gives as a function goes in as a table; the rest is constant.
        for which, stored in (("F", model._F), ("H", model._H), ("Q", model._Q), ("R", model._R)):
            if callable(stored):
                _core.check(L.kalman_model_set_table(handle, which.encode(), int(steps), _core.cdptr(tables[which])),
                            f"table of {which}")
    if l > 0 and model.B(t=t0) is not None:
        put(L.kalman_model_set_matrix, flat(model.B(t=t0), (n, l)), "B", b"B")
    if model.L(t=t0) is not None:
        put(L.kalman_model_set_matrix, flat(model.L(t=t0), (n, n)), "L", b"L")
    if getattr(model, "_Ac", None) is not None and getattr(model, "_Qc", None) is not None:
        # Stated in continuous time: the file keeps the generator and the intensity.
        put(L.kalman_model_set_generator, flat(model._Ac, (n, n)), "A")
        Qc = np.atleast_2d(np.asarray(model._Qc, dtype=float))
        Lc = flat(model._Lc, (n, Qc.shape[0]))
        _core.check(L.kalman_model_set_noise_intensity(handle, Qc.shape[0], _core.cdptr(np.ascontiguousarray(Qc)),
                                                       _core.cdptr(Lc)), "export: Qc")
    return handle


def _constructor_values(model):
    """The arguments ``model`` was constructed with, as a file can hold them:
    ``(found, not_written)``. Numbers, strings, booleans and arrays of numbers
    are written. An argument that is code or an object (a callable, a
    controller) cannot be, and is listed in ``not_written`` when it was given.
    The sample rate, noise and initial state are left out: the file has its own
    places for them."""
    import enum
    import numpy as np
    found, missing = {}, []
    arguments = getattr(model, "_init_arguments", None) or {}
    skipped = {"fs", "sample_rate", "x0", "P0", "Q", "R", getattr(model, "_init_first", None)}
    for name, value in arguments.items():
        if name in skipped or value is None:
            continue
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            value = value.item()
        if isinstance(value, enum.Enum):
            value = value._name_            # the framework's constructors take a kind by its name
        if isinstance(value, (bool, int, float, str)):
            found[name] = value
            continue
        if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
            found[name] = list(value)
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            try:
                array = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                array = None
            if array is not None and np.all(np.isfinite(array)):
                found[name] = array.tolist()
                continue
        missing.append(name)
    return found, missing


def export(model, path: str = None, *, linearize: bool = False, steps: int = 0) -> dict:
    """Write ``model`` out and return its ``model.json`` as a dict (``model.export`` calls this).

    Any model: one written in C++ (a :class:`NativeModel`) or in Python. What the
    file holds follows ``model.structure``:

    - matrices: all of it (``kind`` "complete");
    - matrices that change with time: all of it over ``steps`` samples, as tables
      (needs ``steps`` and a ``.kw`` path);
    - code (``f``, ``h``): a "reference" to the model by name with its settings,
      or, with ``linearize``, its Jacobians at the initial state as matrices.

    A reference to a Python class carries the arguments the instance was
    constructed with. An argument that is code or an object cannot be written;
    ``origin["parameters_not_written"]`` lists those, and such a file gives the
    class's default for them. The class must be in ``kalman.api``'s catalogue.

    ``path`` ending in ``.kw`` gives one archive, another name ending in
    ``.json`` plain JSON; with no path nothing is written.
    """
    import json
    from .framework.constants import KalmanStructureKind as S
    L, ffi = _core.require(), _core.ffi
    path = None if path is None else os.fspath(path)

    params = origin = None
    if isinstance(model, NativeModel):
        handle = model._handle
        registered = model.native_name if _core.has(model.native_name) else None
        plugin = _plugin_of(registered) if registered else ""
    else:
        structure = model.structure
        if structure is S.TIME_VARYING and not linearize and steps <= 0:
            raise ValueError("export: this model's matrices change with time; say over how many samples to "
                             "tabulate them (steps=), or ask for the matrices at the first sample (linearize=True).")
        registered, plugin = None, ""
        origin = {"model": type(model).__name__,
                  "python": f"{type(model).__module__}.{type(model).__qualname__}",
                  "title": str(getattr(model, "NAME", "") or type(model).__name__),
                  "description": str(getattr(model, "DESCRIPTION", "") or "")}
        if structure is S.MAPS and not linearize:
            from . import api
            registered = type(model).__name__
            if registered not in api._model_catalog():
                raise ValueError(f"export: {registered} is given by code and is not in kalman.api's catalogue, so a "
                                 "file cannot refer to it by name. Use linearize=True for its Jacobians at x0.")
            params, missing = _constructor_values(model)
            if missing:
                origin["parameters_not_written"] = missing
        handle = _stand_in(model, structure, linearize, int(steps))

    args = (registered.encode() if registered else ffi.NULL, plugin.encode() if plugin else ffi.NULL,
            int(bool(linearize)), int(steps),
            json.dumps(origin, ensure_ascii=True).encode() if origin is not None else ffi.NULL)
    if params is None and path is not None:
        _core.check(L.kalman_model_save(handle, path.encode(), *args), "export")
        return read_model_file(path)["document"]
    text = L.kalman_model_export_json(handle, *args)
    if text == ffi.NULL:
        raise ValueError(f"export: {_core.last_error()}")
    document = json.loads(ffi.string(text).decode())
    if params is not None:
        # A reference to a Python class: its settings are the constructor's arguments.
        document["spec"]["dynamics"]["params"] = params
        if path is not None:
            _core.check(L.kalman_model_document_save(json.dumps(document).encode(), path.encode()), "export")
    return document


def open_model(path: str, *, settings: bool = True):
    """Open a model file and return the model, ready to filter with.

    Everything the file holds is used: the tables of a model whose matrices
    change with time, and the settings file beside it (``path + ".json"``) unless
    ``settings`` is false. A reference needs what it refers to: the plugin of a
    C++ model loaded (:func:`load_plugin`), or the Python class in
    ``kalman.api``'s catalogue.

    A file of matrices or a reference to a C++ model runs in C++ (a
    :class:`NativeModel`); a reference to a Python class gives that class.
    """
    path = os.fspath(path)
    read = read_model_file(path)
    document = read["document"]
    if (document.get("origin") or {}).get("python") and document.get("kind") == "reference":
        from . import api
        spec = dict(document["spec"])
        chosen = read["settings"] if settings else None
        if chosen:
            # The same rules as Kalman::IO::ModelFile::Apply.
            if chosen.get("for") not in (None, spec["base"]):
                raise ValueError(f"{path}.json: these settings were written for the model {chosen['for']!r}, "
                                 f"not for {spec['base']!r}.")
            allowed = {"format", "format_version", "for", "fs", "parameters", "noise", "initial"}
            unknown = sorted(set(chosen) - allowed)
            if unknown:
                raise ValueError(f"{path}.json: {unknown[0]!r} is not a setting (settings are: fs, parameters, noise, "
                                 "initial). A settings file cannot change what the model is.")
            if "fs" in chosen:
                spec["fs"] = chosen["fs"]
            dynamics = dict(spec.get("dynamics") or {})
            dynamics["params"] = {**(dynamics.get("params") or {}), **(chosen.get("parameters") or {})}
            spec["dynamics"] = dynamics
            spec["noise"] = {**(spec.get("noise") or {}), **(chosen.get("noise") or {})}
            spec["initial"] = {**(spec.get("initial") or {}), **(chosen.get("initial") or {})}
        return api.make_model(spec)

    L, ffi = _core.require(), _core.ffi
    handle = L.kalman_model_open(path.encode(), int(bool(settings)))
    if handle == ffi.NULL:
        raise ValueError(f"open_model({path!r}): {_core.last_error()}")
    return NativeModel(_handle=handle)


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

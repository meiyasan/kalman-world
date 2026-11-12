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

from .framework.native import core as _core
from .framework.models.native import (NativeModel, NativeTwin, python_reference, native_model_classes,
                                      run_callback_filter)
from .framework.engines.native import NativeEngine

__all__ = ["available", "status", "version", "names", "describe", "plugins", "load_plugin",
           "NativeModel", "NativeTwin", "NativeEngine", "native_model_classes", "run_callback_filter",
           "python_reference"]


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

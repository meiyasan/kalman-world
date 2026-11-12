"""
kalman.framework.native.net -- libKalmanNet from Python.

``libKalmanNet`` is the C++ JSON API (``Kalman::Net::Api``) behind the C++
REST and MCP servers (``kalman-serve``). From Python it is one function,
:func:`call`: a verb of :mod:`kalman.api`, its arguments as a dict, and the
C++ answer as a dict -- the same answer ``kalman-serve`` gives over HTTP. It
serves the models compiled into libKalman and its plugins; Python-written
models stay with the Python facade.

The library is looked for like libKalmanLearn (see :mod:`.learn`), under the
name ``libKalmanNet``; ``KALMAN_NET_LIBRARY`` names the file itself. It is
loaded on first use. Loaded next to the core, it shares the core's model
factory, so a plugin loaded through :mod:`.core` is known to it too.
"""
from __future__ import annotations

import ctypes.util
import json
import os

from . import core

#: Mirror of Kalman/Net/CAPI.h (tests/test_net_server.py checks they agree).
_CDEF = """
const char *kalman_net_last_error(void);
int kalman_net_last_status(void);
const char *kalman_net_version(void);
const char *kalman_net_call(const char *verb, const char *arguments, const char *token);
"""

LIBRARY_NAME = "KalmanNet"
ffi = None
lib = None
LIBRARY_PATH = None
LOAD_ERROR = None
_tried = False


class NetError(RuntimeError):
    """A refused call: ``status`` is the HTTP status the servers would answer."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def candidate_paths() -> list[str]:
    """Where libKalmanNet is looked for, in order."""
    out = []
    explicit = os.environ.get("KALMAN_NET_LIBRARY")
    if explicit:
        out.append(explicit)
    if core.LIBRARY_PATH:                                  # next to the core that is loaded
        out.extend(os.path.join(os.path.dirname(core.LIBRARY_PATH), "lib" + LIBRARY_NAME + ext)
                   for ext in core._EXTS)
    for prefix in (os.environ.get("KALMAN_ROOT"), os.environ.get("KALMANFRAMEWORK_ROOT"),
                   str(core._REPO / "var" / "install"), str(core._REPO / "var" / "build")):
        if prefix:
            out.extend(os.path.join(prefix, "lib", "lib" + LIBRARY_NAME + ext) for ext in core._EXTS)
    system = ctypes.util.find_library(LIBRARY_NAME)
    if system:
        out.append(system)
    return out


def load():
    """dlopen libKalmanNet once. Returns ``lib``, or ``None`` (see ``LOAD_ERROR``)."""
    global ffi, lib, LIBRARY_PATH, LOAD_ERROR, _tried
    if lib is not None or _tried:
        return lib
    _tried = True
    try:
        from cffi import FFI
        core.load()                                        # the core first: one factory
        path = next((p for p in candidate_paths() if os.path.isfile(p)), None)
        if path is None:
            raise OSError("libKalmanNet not found; build it (make build install) or set "
                          "KALMAN_NET_LIBRARY.")
        _ffi = FFI()
        _ffi.cdef(_CDEF)
        lib = _ffi.dlopen(path)
        ffi, LIBRARY_PATH = _ffi, path
    except Exception as e:                    # missing file, no cffi ...
        LOAD_ERROR = f"{type(e).__name__}: {e}"
    return lib


def available() -> bool:
    """True when libKalmanNet is loaded (loading it if needed)."""
    return load() is not None


def version() -> str:
    if not available():
        raise RuntimeError(LOAD_ERROR or "libKalmanNet is not available")
    return ffi.string(lib.kalman_net_version()).decode()


def call(verb: str, arguments: dict | None = None, token: str | None = None) -> dict:
    """Run ``verb`` of the C++ API with ``arguments``; the answer as a dict.

    Raises :class:`NetError` with the HTTP status of the refusal (400, 401,
    403, 404, 501)."""
    if not available():
        raise RuntimeError(LOAD_ERROR or "libKalmanNet is not available")
    out = lib.kalman_net_call(verb.encode(), json.dumps(arguments or {}).encode(),
                              token.encode() if token else ffi.NULL)
    if out == ffi.NULL:
        raise NetError(int(lib.kalman_net_last_status()),
                       ffi.string(lib.kalman_net_last_error()).decode())
    return json.loads(ffi.string(out).decode())

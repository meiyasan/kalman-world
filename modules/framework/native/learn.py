"""
kalman.framework.native.learn — libKalmanLearn from Python.

``libKalmanLearn`` holds what needs an AI runtime. From Python it serves the
libTorch field solver, ``Kalman::Learn::Solvers::Vorticity``: the batched
pseudo-spectral step behind ``SpectralSolver(backend="torch")``.

The library is looked for like the core (see :mod:`.core`), under the name
``libKalmanLearn``; ``KALMAN_LEARN_LIBRARY`` names the file itself. It is loaded
on first use, not at import: it brings libTorch with it.

libTorch here is the C++ library; Python's torch package is not used. A process
can hold one libTorch only, so where torch is also imported in Python it must be
the same installation (in a conda environment, ``libtorch`` and ``pytorch`` are).
"""
from __future__ import annotations

import ctypes.util
import os

import numpy as np

from . import core

#: Mirror of Kalman/Learn/CAPI.h (tests/test_native_learn.py checks they agree).
_CDEF = """
const char *kalman_learn_last_error(void);
int kalman_learn_has_torch(void);
typedef struct kalman_vorticity kalman_vorticity_t;
kalman_vorticity_t *kalman_vorticity_create(int ny, int nx, const double *kx, const double *ky,
                                            const double *k2, const double *k2inv, const double *dealias,
                                            double nu, double beta,
                                            const double *forcing_re, const double *forcing_im,
                                            const char *device);
void kalman_vorticity_destroy(kalman_vorticity_t *solver);
const char *kalman_vorticity_device(const kalman_vorticity_t *solver);
int kalman_vorticity_advance(const kalman_vorticity_t *solver, const double *w, int batch, int nsteps,
                             double dt, int stages, const double *c, const double *A, const double *b,
                             double *out);
"""

LIBRARY_NAME = "KalmanLearn"
ffi = None
lib = None
LIBRARY_PATH = None
LOAD_ERROR = None
_tried = False


def candidate_paths() -> list[str]:
    """Where libKalmanLearn is looked for, in order."""
    out = []
    explicit = os.environ.get("KALMAN_LEARN_LIBRARY")
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
    """dlopen libKalmanLearn once. Returns ``lib``, or ``None`` (see ``LOAD_ERROR``)."""
    global ffi, lib, LIBRARY_PATH, LOAD_ERROR, _tried
    if lib is not None or _tried:
        return lib
    _tried = True
    try:
        from cffi import FFI
        path = next((p for p in candidate_paths() if os.path.isfile(p)), None)
        if path is None:
            raise OSError("libKalmanLearn not found; build it (make build install) or set "
                          "KALMAN_LEARN_LIBRARY.")
        _ffi = FFI()
        _ffi.cdef(_CDEF)
        lib = _ffi.dlopen(path)
        ffi, LIBRARY_PATH = _ffi, path
    except Exception as e:                    # missing file, a second libTorch, no cffi ...
        LOAD_ERROR = f"{type(e).__name__}: {e}"
    return lib


def available() -> bool:
    """True when libKalmanLearn is loaded (loading it if needed)."""
    return load() is not None


def has_torch() -> bool:
    """True when libKalmanLearn is loaded and was built with libTorch."""
    if not available():
        return False
    try:
        return bool(lib.kalman_learn_has_torch())
    except AttributeError:                    # a libKalmanLearn older than this interface
        return False


def last_error() -> str:
    return ffi.string(lib.kalman_learn_last_error()).decode() if lib is not None else (LOAD_ERROR or "")


def _d(a):
    """A C-contiguous float64 copy and its pointer."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    return a, ffi.cast("const double *", ffi.from_buffer(a))


class Vorticity:
    """``Kalman::Learn::Solvers::Vorticity``: batched 2-D vorticity steps on libTorch.

    ``dw/dt = -u.grad(w) - beta v + nu lap(w) + F`` on a periodic grid; ``forcing_hat`` is
    the 2-D transform of ``F`` (complex, ``(ny, nx)``) or ``None``.
    """

    def __init__(self, kx, ky, k2, k2inv, dealias, *, nu, beta=0.0, forcing_hat=None, device="cpu"):
        if not has_torch():
            raise RuntimeError("libKalmanLearn with libTorch is not available — "
                               + (LOAD_ERROR or "it was built without libTorch"))
        ny, nx = np.shape(kx)
        self.shape = (int(ny), int(nx))
        keep = [_d(a) for a in (kx, ky, k2, k2inv, np.asarray(dealias, dtype=float))]
        fre = fim = ffi.NULL
        if forcing_hat is not None:
            f = np.asarray(forcing_hat, dtype=complex)
            (_r, fre), (_i, fim) = _d(f.real), _d(f.imag)
        handle = lib.kalman_vorticity_create(ny, nx, *(p for _, p in keep), float(nu), float(beta),
                                             fre, fim, str(device or "cpu").encode())
        if handle == ffi.NULL:
            raise RuntimeError("kalman_vorticity_create: " + last_error())
        self._handle = ffi.gc(handle, lib.kalman_vorticity_destroy)
        self.device = ffi.string(lib.kalman_vorticity_device(self._handle)).decode()

    def advance(self, w, nsteps, dt, tableau):
        """Advance ``w`` of shape ``(batch, ny, nx)`` by ``nsteps`` steps of ``dt``.
        ``tableau`` is ``(c, A, b)`` of an explicit Runge-Kutta scheme."""
        ny, nx = self.shape
        x, px = _d(w)
        if x.ndim != 3 or x.shape[1:] != (ny, nx):
            raise ValueError(f"expected (batch, {ny}, {nx}), got {x.shape}")
        c, A, b = tableau
        s = len(b)
        full = np.zeros((s, s))
        for i, row in enumerate(A):                        # rows may be ragged (lower triangle)
            full[i, :len(row)] = row
        (_c, pc), (_A, pA), (_b, pb) = _d(np.asarray(c, dtype=float)), _d(full), _d(np.asarray(b, dtype=float))
        out = np.empty_like(x)
        rc = lib.kalman_vorticity_advance(self._handle, px, x.shape[0], int(nsteps), float(dt), s,
                                          pc, pA, pb, ffi.cast("double *", ffi.from_buffer(out)))
        if rc != 0:
            raise RuntimeError("kalman_vorticity_advance: " + last_error())
        return out

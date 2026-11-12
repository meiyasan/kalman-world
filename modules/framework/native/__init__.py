"""kalman/framework/native — the C++ core's kernels under the Python engines.

The Python engines evaluate the *model* themselves — ``f`` at the sigma points,
``F/H/Q/R`` at the current point, ``h`` at every particle — and hand the arrays to
``libKalman`` for the *step*: the covariance prediction, the NaN-masked
Joseph update, a whole linear pass, the RTS sweep, the sigma-point moments,
particle weights and resampling, NIS and NEES. One crossing per step with
arrays, never a callback per point. Those kernels are the same code as the C++
engines a ROOT+ program runs (``Kalman::Filter``, ``Engine::Smooth``,
``Utils::Correct`` ...), so a Python model here and a C++ model there run one
implementation of the arithmetic.

Exposes ``HAS_C_ACCEL`` and, when the library was found (see ``core.py`` for
where it is looked for), ``ffi``/``lib`` plus ``dptr``/``iptr`` helpers to get raw
pointers from contiguous numpy arrays. Every call site in the engines guards on
``HAS_C_ACCEL`` and falls back to the pure-Python path when the library is not
built -- the C++ path is faster and shared, never load-bearing for correctness.

This module used to compile its own C copy of these kernels (``kalman.c``); a
second implementation of a Kalman step was a second place for it to be wrong.
"""
from __future__ import annotations
import numpy as np

from . import core

ffi = core.ffi
lib = core.lib
HAS_C_ACCEL = core.HAS_CXX_CORE
LOAD_ERROR = core.LOAD_ERROR


def as_contiguous(arr, dtype=np.float64):
    """C-contiguous array of ``dtype`` (copies only if the input isn't already)."""
    return np.ascontiguousarray(arr, dtype=dtype)


def dptr(arr: np.ndarray):
    """``double *`` cffi pointer into a C-contiguous float64 numpy array."""
    return ffi.cast("double *", arr.ctypes.data)


def iptr(arr: np.ndarray):
    """``int *`` cffi pointer into a C-contiguous ``np.intc`` numpy array."""
    return ffi.cast("int *", arr.ctypes.data)

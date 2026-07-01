"""kalman/framework/native — optional C acceleration of small-matrix Kalman loops.

Exposes ``HAS_C_ACCEL`` and, when available, ``ffi``/``lib`` (the cffi handles for
``kalman.c``) plus ``dptr``/``iptr`` helpers to get raw pointers from contiguous
numpy arrays. Every call site in the engines guards on ``HAS_C_ACCEL`` and falls
back to the existing pure-Python path when the extension is missing or fails to
build (e.g. no C compiler available) -- the C path is purely an accelerator, never
load-bearing for correctness.
"""
from __future__ import annotations
import numpy as np

try:
    from ._kalman_native import ffi, lib
    HAS_C_ACCEL = True
except ImportError:
    ffi = None
    lib = None
    HAS_C_ACCEL = False


def as_contiguous(arr, dtype=np.float64):
    """C-contiguous array of ``dtype`` (copies only if the input isn't already)."""
    return np.ascontiguousarray(arr, dtype=dtype)


def dptr(arr: np.ndarray):
    """``double *`` cffi pointer into a C-contiguous float64 numpy array."""
    return ffi.cast("double *", arr.ctypes.data)


def iptr(arr: np.ndarray):
    """``int *`` cffi pointer into a C-contiguous ``np.intc`` numpy array."""
    return ffi.cast("int *", arr.ctypes.data)

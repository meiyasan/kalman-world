"""kalman/framework/native/core — the Python side of libKalmanFramework's C API.

This module ``dlopen``s the standalone C++ library — the same ``libKalmanFramework`` that
root-plus links — through the flat C surface declared in ``include/Kalman/CAPI.h``,
using cffi in **ABI mode**, so no compiler is involved on the Python side.

What comes through is the **factory**: every model compiled into the library or
into a plugin it has loaded, discoverable by name and constructible without the
Python side knowing its type. :class:`~kalman.framework.models.native.NativeModel`
wraps such an instance as an ordinary :class:`~kalman.framework.model.KalmanModel`,
which is how a C++ model reaches the catalogue, the API and the MCP server.

Exposes ``HAS_CXX_CORE`` and, when the library was found, ``ffi``/``lib`` plus the
thin helpers below. Everything is optional: an import never fails, it only reports.

Locating the library, in order:

  1. ``KALMAN_FRAMEWORK_LIBRARY`` — the file itself;
  2. ``KALMANFRAMEWORK_ROOT`` / ``KALMAN_ROOT`` — an install prefix (``<prefix>/lib``);
  3. this checkout's ``var/install/lib`` and ``var/build/lib``;
  4. the system loader (``ctypes.util.find_library``).

The ``cdef`` below is a copy of ``CAPI.h``; ``tests/test_native_core.py`` compares
the two, so a prototype added to the header must be added here as well.
"""
from __future__ import annotations
import ctypes.util
import os
import pathlib
import re
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parents[2]                      # modules/framework/native -> repo root
_EXT = ".dylib" if sys.platform == "darwin" else ".so"
_EXTS = (".dylib", ".so") if sys.platform == "darwin" else (".so", ".dylib")

# --- the contract ----------------------------------------------------------
# Verbatim prototypes from include/Kalman/CAPI.h (comments and the extern "C"
# wrapper stripped, which is all cffi's parser wants).
_CDEF = """
typedef struct kalman_model kalman_model_t;

typedef struct kalman_model_info
{
    const char *name;
    const char *title;
    const char *description;
    const char *version;
    const char *plugin;
    int n;
    int m;
    int l;
    double fs;
    int engine;
    int analytic;
} kalman_model_info_t;

const char *kalman_version(void);
const char *kalman_last_error(void);

int kalman_factory_count(void);
const char *kalman_factory_name(int i);
int kalman_factory_has(const char *name);
int kalman_factory_describe(const char *name, kalman_model_info_t *info);
int kalman_factory_load(const char *path);
int kalman_factory_plugin_count(void);
const char *kalman_factory_plugin(int i);

kalman_model_t *kalman_model_create(const char *name);
void kalman_model_destroy(kalman_model_t *model);
const char *kalman_model_name(const kalman_model_t *model);
const char *kalman_model_title(const kalman_model_t *model);
const char *kalman_model_description(const kalman_model_t *model);
const char *kalman_model_model_version(const kalman_model_t *model);
const char *kalman_model_describe(const kalman_model_t *model);

int kalman_model_dims(const kalman_model_t *model, int *n, int *m, int *l);
double kalman_model_sample_rate(const kalman_model_t *model);
int    kalman_model_set_sample_rate(kalman_model_t *model, double fs);
double kalman_model_t0(const kalman_model_t *model);
int    kalman_model_set_t0(kalman_model_t *model, double t0);
int kalman_model_engine(const kalman_model_t *model);
int kalman_model_analytic_jacobian(const kalman_model_t *model);
int kalman_model_check_shapes(const kalman_model_t *model);

int kalman_model_x0(const kalman_model_t *model, double *x0);
int kalman_model_P0(const kalman_model_t *model, double *P0);
int kalman_model_set_x0(kalman_model_t *model, const double *x0);
int kalman_model_set_P0(kalman_model_t *model, const double *P0);

int kalman_model_F(const kalman_model_t *model, double t, double *F);
int kalman_model_H(const kalman_model_t *model, double t, double *H);
int kalman_model_Q(const kalman_model_t *model, double t, double *Q);
int kalman_model_R(const kalman_model_t *model, double t, double *R);
int kalman_model_B(const kalman_model_t *model, double t, double *B);
int kalman_model_L(const kalman_model_t *model, double t, double *L);

int kalman_model_f(const kalman_model_t *model, const double *x, const double *u, double t, double *x_next);
int kalman_model_h(const kalman_model_t *model, const double *x, double t, double *z);
int kalman_model_jacobian(const kalman_model_t *model, const double *x, const double *u, double t, double *J);
int kalman_model_jacobian_h(const kalman_model_t *model, const double *x, double t, double *J);

typedef struct kalman_engine kalman_engine_t;

kalman_engine_t *kalman_engine_create(int kind);
kalman_engine_t *kalman_engine_create_unscented(double alpha, double beta, double kappa);
kalman_engine_t *kalman_engine_create_particle(int n_particles, double threshold, unsigned long long seed);
kalman_engine_t *kalman_engine_create_ensemble(int n_members, double inflation, unsigned long long seed);
void kalman_engine_destroy(kalman_engine_t *engine);
int kalman_engine_kind(const kalman_engine_t *engine);
int kalman_engine_has_smoother(const kalman_engine_t *engine);
int kalman_engine_reset(kalman_engine_t *engine, const kalman_model_t *model);
int kalman_engine_predict(const kalman_engine_t *engine, const kalman_model_t *model,
                          const double *x, const double *P, const double *u, double t,
                          double *x_out, double *P_out, double *F_out);
int kalman_engine_update(const kalman_engine_t *engine, const kalman_model_t *model,
                         const double *x, const double *P, const double *z, double t,
                         double *x_out, double *P_out);
int kalman_engine_innovation(const kalman_engine_t *engine, const kalman_model_t *model,
                             const double *x, const double *P, const double *z, double t,
                             double *nu, double *S);
int kalman_engine_smooth(const kalman_engine_t *engine, int n,
                         const double *x_post, const double *P_post,
                         const double *x_prior_next, const double *P_prior_next,
                         const double *x_smooth_next, const double *P_smooth_next,
                         const double *F, double *x_smooth, double *P_smooth);
int kalman_filter_run(const kalman_model_t *model, kalman_engine_t *engine, int N,
                      const double *zk, const double *uk, int smooth,
                      double *x_prior, double *P_prior, double *x_post, double *P_post,
                      double *x_smooth, double *P_smooth, double *innovations, double *transitions);

void kf_predict(int n, int l,
                const double *F, const double *x, const double *P,
                const double *B, const double *u,
                const double *L, const double *Q,
                double *x_out, double *P_out);
int kf_update(int n, int m,
              const double *H, const double *x, const double *P,
              const double *z, const double *R,
              double *x_out, double *P_out);
int kf_update_residual(int n, int m,
                        const double *H, const double *x, const double *P,
                        const double *y, const double *R,
                        double *x_out, double *P_out);
void kf_forward_linear(int n, int m, int l, int N,
                        const double *F, const double *H,
                        const double *Q, const double *R,
                        const double *B, const double *L,
                        const double *zk, const double *uk,
                        const double *x0, const double *P0,
                        double *x_prior, double *P_prior,
                        double *x_post, double *P_post,
                        double *innovations);
void kf_backward_rts(int n, int N,
                      const double *x_post, const double *P_post,
                      const double *x_prior, const double *P_prior,
                      const double *F,
                      double *x_smooth, double *P_smooth);
void pf_weight_update(int n_particles, int m,
                       const double *z, const double *zp,
                       const double *Rinv, double *logw);
void pf_systematic_resample(int n, const double *weights, double u0, int *indices);
void ukf_weighted_outer_sum(int n_sigma, int n,
                             const double *Wc, const double *Y, const double *mean,
                             double *P_out);
void ukf_weighted_cross_sum(int n_sigma, int na, int nb,
                             const double *Wc, const double *A, const double *B,
                             double *C_out);
void diag_nis_batch(int N, int m,
                     const double *nu, const double *S, double *nis);
void diag_nees_batch(int N, int n,
                      const double *e, const double *P, double *nees);
"""

ffi = None
lib = None
LIBRARY_PATH: str | None = None
HAS_CXX_CORE = False
LOAD_ERROR: str | None = None


# --- locating and loading ----------------------------------------------------
def candidate_paths() -> list[str]:
    """Every place the library is looked for, in order; the first that exists wins."""
    out = []
    explicit = os.environ.get("KALMAN_FRAMEWORK_LIBRARY")
    if explicit:
        out.append(explicit)
    prefixes = [os.environ.get("KALMANFRAMEWORK_ROOT"), os.environ.get("KALMAN_ROOT"),
                str(_REPO / "var" / "install"), str(_REPO / "var" / "build")]
    for prefix in prefixes:
        if prefix:
            for ext in _EXTS:
                out.append(os.path.join(prefix, "lib", "libKalmanFramework" + ext))
    system = ctypes.util.find_library("KalmanFramework")
    if system:
        out.append(system)
    return out


def find_library() -> str | None:
    for path in candidate_paths():
        if os.path.isfile(path):
            return path
    return None


def load(path: str | None = None):
    """dlopen the library (idempotent). Returns ``lib`` or raises ``OSError``."""
    global ffi, lib, LIBRARY_PATH, HAS_CXX_CORE, LOAD_ERROR
    if lib is not None and (path is None or path == LIBRARY_PATH):
        return lib
    from cffi import FFI
    path = path or find_library()
    if not path:
        raise OSError("libKalmanFramework not found; build it (make build install) or set "
                      "KALMAN_FRAMEWORK_LIBRARY / KALMANFRAMEWORK_ROOT. Looked in: "
                      + ", ".join(candidate_paths()))
    _ffi = FFI()
    _ffi.cdef(_CDEF)
    _lib = _ffi.dlopen(path)
    ffi, lib, LIBRARY_PATH, HAS_CXX_CORE, LOAD_ERROR = _ffi, _lib, path, True, None
    return lib


try:
    load()
except Exception as _e:                       # missing library, wrong arch, no cffi ...
    LOAD_ERROR = f"{type(_e).__name__}: {_e}"


def require():
    """``lib``, or a ``RuntimeError`` explaining why there is none."""
    if lib is None:
        raise RuntimeError("the C++ core is not loaded — " + (LOAD_ERROR or "unknown reason"))
    return lib


# --- calling conventions -----------------------------------------------------
def _s(cptr) -> str:
    return "" if cptr == ffi.NULL else ffi.string(cptr).decode()


def last_error() -> str:
    return _s(lib.kalman_last_error())


def check(rc: int, what: str):
    """Turn the C convention (``-1`` + message) back into an exception."""
    if rc < 0:
        raise RuntimeError(f"{what}: {last_error() or 'failed'}")
    return rc


def dptr(arr: np.ndarray):
    """``double *`` into a C-contiguous float64 array."""
    return ffi.cast("double *", arr.ctypes.data)


def cdptr(arr: np.ndarray):
    """``const double *`` into a C-contiguous float64 array."""
    return ffi.cast("const double *", arr.ctypes.data)


def version() -> str:
    return _s(require().kalman_version())


# --- factory -----------------------------------------------------------------
def names() -> list[str]:
    L = require()
    count = check(L.kalman_factory_count(), "kalman_factory_count")
    return [_s(L.kalman_factory_name(i)) for i in range(count)]


def has(name: str) -> bool:
    return check(require().kalman_factory_has(name.encode()), "kalman_factory_has") == 1


def describe(name: str) -> dict:
    L = require()
    info = ffi.new("kalman_model_info_t *")
    check(L.kalman_factory_describe(name.encode(), info), f"describe({name!r})")
    return {"name": _s(info.name), "title": _s(info.title), "description": _s(info.description),
            "version": _s(info.version), "plugin": _s(info.plugin) or None,
            "n": int(info.n), "m": int(info.m), "l": int(info.l), "fs": float(info.fs),
            "engine": int(info.engine), "analytic": bool(info.analytic)}


def plugins() -> list[str]:
    L = require()
    count = check(L.kalman_factory_plugin_count(), "kalman_factory_plugin_count")
    return [_s(L.kalman_factory_plugin(i)) for i in range(count)]


def load_plugin(path: str) -> list[str]:
    """dlopen ``path`` (already resolved — see :func:`resolve_plugin`) and return
    the names it registered. Idempotent: a second load returns the same names."""
    L = require()
    before = set(names())
    check(L.kalman_factory_load(path.encode()), f"load_plugin({path!r})")
    after = names()
    added = [n for n in after if n not in before]
    if added:
        return added
    # Already loaded earlier: report what that library owns.
    return [n for n in after if describe(n)["plugin"] == path]


# --- engines and a fused pass --------------------------------------------------
# Kalman::EngineKind by name, the contract shared with Constants.h and models/native.py.
ENGINE_KINDS = {"linear": 0, "extended": 1, "unscented": 2, "particle": 3,
                "ensemble": 4, "combinatorial": 5}


def create_engine(kind="extended", **params):
    """A C++ engine handle (garbage-collected) of ``kind`` — a name or the integer —
    configured by the same parameters as the C++ constructors: ``alpha/beta/kappa``
    for the unscented one, ``n_particles/threshold/seed`` for the particle one,
    ``n_members/inflation/seed`` for the ensemble one."""
    L = require()
    code = ENGINE_KINDS[kind.lower()] if isinstance(kind, str) else int(kind)
    if code == 2 and params:
        h = L.kalman_engine_create_unscented(float(params.get("alpha", 1e-3)),
                                             float(params.get("beta", 2.0)), float(params.get("kappa", 0.0)))
    elif code == 3 and params:
        h = L.kalman_engine_create_particle(int(params.get("n_particles", 500)),
                                            float(params.get("threshold", 0.5)), int(params.get("seed", 0)))
    elif code == 4 and params:
        h = L.kalman_engine_create_ensemble(int(params.get("n_members", 50)),
                                            float(params.get("inflation", 1.0)), int(params.get("seed", 0)))
    else:
        if params:
            raise TypeError(f"engine kind {kind!r} takes no parameters (got {sorted(params)}).")
        h = L.kalman_engine_create(code)
    if h == ffi.NULL:
        raise RuntimeError(f"create_engine({kind!r}): {last_error()}")
    return ffi.gc(h, L.kalman_engine_destroy)


def run_filter(model_handle, engine_handle, n, m, l, zk, uk=None, smooth=False):
    """The fused pass: Kalman::Filter over ``zk`` (N x m; NaN = missing) with ``uk``
    (N x l or None) on a native model and engine. Returns the per-step arrays."""
    L = require()
    zk = np.ascontiguousarray(np.asarray(zk, dtype=np.float64).reshape(-1, m))
    N = zk.shape[0]
    if uk is not None and l > 0:
        uk = np.ascontiguousarray(np.asarray(uk, dtype=np.float64).reshape(N, l))
    out = {"x_prior": np.empty((N, n)), "P_prior": np.empty((N, n, n)),
           "x_post": np.empty((N, n)), "P_post": np.empty((N, n, n)),
           "innovations": np.empty((N, m)), "transitions": np.empty((N, n, n))}
    if smooth:
        out["x_smooth"] = np.empty((N, n))
        out["P_smooth"] = np.empty((N, n, n))
    rc = L.kalman_filter_run(
        model_handle, engine_handle, int(N), cdptr(zk),
        cdptr(uk) if (uk is not None and l > 0) else ffi.NULL, 1 if smooth else 0,
        dptr(out["x_prior"]), dptr(out["P_prior"]), dptr(out["x_post"]), dptr(out["P_post"]),
        dptr(out["x_smooth"]) if smooth else ffi.NULL, dptr(out["P_smooth"]) if smooth else ffi.NULL,
        dptr(out["innovations"]), dptr(out["transitions"]))
    check(rc, "kalman_filter_run")
    return out


# --- plugin policy -----------------------------------------------------------
# Loading a plugin executes code. The registry does not decide what may be
# loaded; this does: only files inside an allow-listed directory, and by *name*
# from the API so a caller never chooses a path. ``KALMAN_PLUGIN_PATH`` (os.pathsep-
# separated) replaces the default list, which is the two trees of this checkout
# and wherever the library itself was found.
def plugin_dirs() -> list[str]:
    env = os.environ.get("KALMAN_PLUGIN_PATH")
    if env:
        dirs = [d for d in env.split(os.pathsep) if d]
    else:
        dirs = [str(_REPO / "var" / "install" / "lib"), str(_REPO / "var" / "build" / "lib")]
        if LIBRARY_PATH:
            dirs.append(os.path.dirname(LIBRARY_PATH))
    out = []
    for d in dirs:
        real = os.path.realpath(d)
        if real not in out:
            out.append(real)
    return out


_NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


def resolve_plugin(name_or_path: str) -> str:
    """The file to load for ``name_or_path``, or an error naming what was tried.

    A bare name (``KalmanPlugin_example``) is looked up as ``lib<name><ext>``,
    ``<name><ext>`` or ``<name>`` in every allow-listed directory. A path is
    accepted only when its real location lies inside one of those directories —
    ``../`` and symlinks out of the tree are resolved before the check.
    """
    if not isinstance(name_or_path, str) or not name_or_path:
        raise ValueError("plugin: give a plugin name or path.")
    dirs = plugin_dirs()
    if os.sep in name_or_path or (os.altsep and os.altsep in name_or_path) \
            or name_or_path.endswith(_EXTS):
        real = os.path.realpath(name_or_path)
        if not os.path.isfile(real):
            raise FileNotFoundError(f"plugin: {name_or_path!r} does not exist.")
        if not any(os.path.dirname(real) == d for d in dirs):
            raise PermissionError(f"plugin: {name_or_path!r} is outside the allowed directories "
                                  f"{dirs}; set KALMAN_PLUGIN_PATH to allow another.")
        return real
    if not _NAME_RE.match(name_or_path):
        raise ValueError(f"plugin: {name_or_path!r} is not a plugin name.")
    tried = []
    for d in dirs:
        for candidate in ([f"lib{name_or_path}{e}" for e in _EXTS]
                          + [f"{name_or_path}{e}" for e in _EXTS] + [name_or_path]):
            path = os.path.join(d, candidate)
            tried.append(path)
            if os.path.isfile(path):
                return os.path.realpath(path)
    raise FileNotFoundError(f"plugin: no {name_or_path!r} in {dirs} (tried {len(tried)} names).")

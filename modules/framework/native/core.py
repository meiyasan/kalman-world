"""kalman/framework/native/core — the Python side of libKalman's C API.

This module ``dlopen``s the standalone C++ library — the same ``libKalman`` that
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

  1. ``KALMAN_LIBRARY`` — the file itself;
  2. ``KALMAN_ROOT`` — an install prefix (``<prefix>/lib``);
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

typedef int (*kalman_step_fn)(void *user, const double *x, const double *u, double t, double *x_next);
typedef int (*kalman_measure_fn)(void *user, const double *x, double t, double *z);
typedef int (*kalman_step_jacobian_fn)(void *user, const double *x, const double *u, double t, double *J);
typedef int (*kalman_measure_jacobian_fn)(void *user, const double *x, double t, double *J);
kalman_model_t *kalman_model_create_callback(const char *name, int n, int m, int l,
                                             kalman_step_fn f, kalman_measure_fn h,
                                             kalman_step_jacobian_fn jacobian_f,
                                             kalman_measure_jacobian_fn jacobian_h, void *user);
typedef int (*kalman_process_noise_fn)(void *user, const double *x, const double *u, double t, double *Q);
typedef int (*kalman_measurement_noise_fn)(void *user, double t, double *R);
int kalman_model_callback_set_noise(kalman_model_t *model, kalman_process_noise_fn process_noise,
                                    kalman_measurement_noise_fn measurement_noise);
typedef int (*kalman_noisy_step_fn)(void *user, const double *x, const double *u, const double *w, double t, double *x_next);
typedef int (*kalman_noisy_measure_fn)(void *user, const double *x, const double *v, double t, double *z);
int kalman_model_callback_set_noise_inputs(kalman_model_t *model, kalman_noisy_step_fn f_with_noise,
                                           kalman_noisy_measure_fn h_with_noise);
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

int kalman_model_parameter_count(const kalman_model_t *model);
const char *kalman_model_parameter_name(const kalman_model_t *model, int i);
int kalman_model_parameter(const kalman_model_t *model, const char *name, double *value);
int kalman_model_set_parameter(kalman_model_t *model, const char *name, double value);
int kalman_model_set_Q(kalman_model_t *model, const double *Q);
int kalman_model_set_R(kalman_model_t *model, const double *R);

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
typedef int (*kalman_control_fn)(void *user, const double *x, const double *P, double t,
                                 const double *nu, double *u);
int kalman_filter_run_closed_loop(const kalman_model_t *model, kalman_engine_t *engine, int N,
                                  const double *zk, kalman_control_fn control, void *user, int smooth,
                                  double *x_prior, double *P_prior, double *x_post, double *P_post,
                                  double *x_smooth, double *P_smooth, double *innovations,
                                  double *transitions, double *controls);
int kalman_matrix_exponential(int n, const double *A, double *out);
int kalman_van_loan(int n, const double *A, const double *G, double dt, double *F, double *Q);
int kalman_model_dynamics(const kalman_model_t *model);
int kalman_model_set_generator(kalman_model_t *model, const double *A);
int kalman_model_set_noise_intensity(kalman_model_t *model, int q, const double *Qc, const double *Lc);
int kalman_model_get_F_Q(const kalman_model_t *model, double *F, double *Q);
int kalman_propagate(const kalman_model_t *model, kalman_engine_t *engine, int N,
                     const double *x0, const double *P0, double t0, double fs,
                     const double *uk, const double *wk, double *xk, double *Pk);
typedef int (*kalman_map_fn)(void *user, const double *x, double *y);
int kalman_jacobian(int kind, double step, kalman_map_fn fn, void *user, int n, int m,
                    const double *x, const double *y0, double *J);
int kalman_model_differentiator(const kalman_model_t *model);
int kalman_model_set_differentiator(kalman_model_t *model, int kind, double step);
typedef int (*kalman_derivative_fn)(void *user, const double *x, double t, double *dx);
int kalman_integrate(int kind, double rtol, double atol, int max_substeps,
                     kalman_derivative_fn f, void *user, int n,
                     const double *x, double t, double dt, int nsteps, double *out);

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

typedef struct kalman_tracks kalman_tracks_t;
kalman_tracks_t *kalman_ckf_find(const kalman_model_t *model, int n_layers, const double *radii,
                                 const int *hits_per_layer, const double *hits,
                                 int n_seeds, const int *seed_layer, const int *seed_index,
                                 const double *seed_x, const double *seed_P,
                                 double chi2_cut, double hole_penalty, int max_candidates,
                                 int max_holes, int min_hits, double shared_hit_frac);
void kalman_tracks_destroy(kalman_tracks_t *tracks);
int kalman_tracks_count(const kalman_tracks_t *tracks);
int kalman_track_hits(const kalman_tracks_t *tracks, int i);
int kalman_track_summary(const kalman_tracks_t *tracks, int i, double *chi2, double *score, int *holes,
                         double *x, double *P, double *Faccum);
int kalman_track_copy(const kalman_tracks_t *tracks, int i, int *layers, int *indices, double *radii,
                      double *xk, double *Pk, double *zk, double *prior_x, double *prior_P, double *Fc);
"""

ffi = None
lib = None
LIBRARY_PATH: str | None = None
HAS_CXX_CORE = False
LOAD_ERROR: str | None = None


# --- locating and loading ----------------------------------------------------
#: The library's name, then the one it had up to release 1.0.
LIBRARY_NAMES = ("Kalman", "KalmanFramework")

def candidate_paths() -> list[str]:
    """Every place the library is looked for, in order; the first that exists wins."""
    out = []
    # KALMAN_LIBRARY / KALMAN_ROOT are the names; the *FRAMEWORK* ones are those
    # of releases up to 1.0, when the library was libKalmanFramework, and are
    # still read.
    for variable in ("KALMAN_LIBRARY", "KALMAN_FRAMEWORK_LIBRARY"):
        explicit = os.environ.get(variable)
        if explicit:
            out.append(explicit)
    prefixes = [os.environ.get("KALMAN_ROOT"), os.environ.get("KALMANFRAMEWORK_ROOT"),
                str(_REPO / "var" / "install"), str(_REPO / "var" / "build")]
    for prefix in prefixes:
        if prefix:
            for name in LIBRARY_NAMES:
                for ext in _EXTS:
                    out.append(os.path.join(prefix, "lib", "lib" + name + ext))
    for name in LIBRARY_NAMES:
        system = ctypes.util.find_library(name)
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
        raise OSError("libKalman not found; build it (make build install) or set "
                      "KALMAN_LIBRARY / KALMAN_ROOT. Looked in: "
                      + ", ".join(candidate_paths()))
    _ffi = FFI()
    _ffi.cdef(_CDEF)
    _lib = _ffi.dlopen(path)
    ffi, lib, LIBRARY_PATH, HAS_CXX_CORE, LOAD_ERROR = _ffi, _lib, path, True, None
    return lib


class PurePythonWarning(FutureWarning):
    """Raised once at import when libKalman is not loaded.

    C++ is the implementation of the framework; the Python arithmetic is kept as
    the reference the tests compare against and will stop being a way to run
    it. Build the library (``make build`` then ``cmake --install var/build``),
    install the conda package, or point ``KALMAN_LIBRARY`` at it. To silence::

        warnings.filterwarnings("ignore", category=kalman.framework.native.core.PurePythonWarning)
    """


try:
    # KALMAN_PURE_PYTHON=1 leaves the library out on purpose (comparisons, debugging).
    if os.environ.get("KALMAN_PURE_PYTHON", "0") == "1":
        raise OSError("libKalman left out on request (KALMAN_PURE_PYTHON=1)")
    load()
except Exception as _e:                       # missing library, wrong arch, no cffi ...
    LOAD_ERROR = f"{type(_e).__name__}: {_e}"
    import warnings as _warnings
    _warnings.warn(
        "libKalman is not loaded (" + LOAD_ERROR.split(". Looked in")[0] + "); the framework is running on its "
        "Python reference code, which is deprecated as a way to run it and slower. Build or install the "
        "library, or set KALMAN_LIBRARY.", PurePythonWarning, stacklevel=2)


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


def run_filter(model_handle, engine_handle, n, m, l, zk, uk=None, smooth=False, control=None):
    """The fused pass: Kalman::Filter over ``zk`` (N x m; NaN = missing) with ``uk``
    (N x l or None) on a native model and engine. Returns the per-step arrays.

    ``control`` closes the loop: a function ``control(x, P, t, nu=nu) -> u`` called
    at every step with the belief before the step (``x`` (n,1), ``P`` (n,n)) and the
    previous innovation (``nu`` (m,1), zeros at first). The commands applied are
    returned as ``controls`` (N x l)."""
    L = require()
    zk = np.ascontiguousarray(np.asarray(zk, dtype=np.float64).reshape(-1, m))
    N = zk.shape[0]
    if control is not None:
        return _run_closed_loop(L, model_handle, engine_handle, n, m, l, zk, N, smooth, control)
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


def _run_closed_loop(L, model_handle, engine_handle, n, m, l, zk, N, smooth, control):
    out = {"x_prior": np.empty((N, n)), "P_prior": np.empty((N, n, n)),
           "x_post": np.empty((N, n)), "P_post": np.empty((N, n, n)),
           "innovations": np.empty((N, m)), "transitions": np.empty((N, n, n)),
           "controls": np.empty((N, l))}
    if smooth:
        out["x_smooth"] = np.empty((N, n))
        out["P_smooth"] = np.empty((N, n, n))
    raised = []

    @ffi.callback("int(void *, const double *, const double *, double, const double *, double *)")
    def policy(_user, x, P, t, nu, u):
        try:
            xs = np.frombuffer(ffi.buffer(x, n * 8), dtype=np.float64).reshape(n, 1).copy()
            Ps = np.frombuffer(ffi.buffer(P, n * n * 8), dtype=np.float64).reshape(n, n).copy()
            nus = np.frombuffer(ffi.buffer(nu, m * 8), dtype=np.float64).reshape(m, 1).copy()
            command = np.asarray(control(xs, Ps, t, nu=nus), dtype=np.float64).reshape(-1)
            if command.size != l:
                raise ValueError(f"the controller returned {command.size} commands, the model takes {l}.")
            ffi.buffer(u, l * 8)[:] = command.tobytes()
            return 0
        except BaseException as e:               # must not cross the C boundary
            raised.append(e)
            return -1

    rc = L.kalman_filter_run_closed_loop(
        model_handle, engine_handle, int(N), cdptr(zk), policy, ffi.NULL, 1 if smooth else 0,
        dptr(out["x_prior"]), dptr(out["P_prior"]), dptr(out["x_post"]), dptr(out["P_post"]),
        dptr(out["x_smooth"]) if smooth else ffi.NULL, dptr(out["P_smooth"]) if smooth else ffi.NULL,
        dptr(out["innovations"]), dptr(out["transitions"]), dptr(out["controls"]))
    if raised:
        raise raised[0]
    check(rc, "kalman_filter_run_closed_loop")
    return out


# --- continuous models: decoding ------------------------------------------------
def matrix_exponential(A):
    """``exp(A)`` in C++ (Kalman::Utils::Exponential)."""
    L = require()
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    out = np.empty_like(A)
    check(L.kalman_matrix_exponential(A.shape[0], cdptr(A), dptr(out)), "kalman_matrix_exponential")
    return out


def van_loan(A, G, dt):
    """``(F, Q)`` for one sample of ``dx/dt = A x + w`` with noise intensity ``G``
    (Kalman::Utils::VanLoan)."""
    L = require()
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    G = np.ascontiguousarray(np.asarray(G, dtype=np.float64))
    F, Q = np.empty_like(A), np.empty_like(A)
    check(L.kalman_van_loan(A.shape[0], cdptr(A), cdptr(G), float(dt), dptr(F), dptr(Q)), "kalman_van_loan")
    return F, Q


# --- sampling ------------------------------------------------------------------
def propagate(model_handle, engine_handle, n, l, N, x0, P0, t0, fs, uk=None, wk=None):
    """The state chain of a sampled run (kalman_propagate): ``(xk, Pk)`` of shapes
    ``(N, n)`` and ``(N, n, n)``, with the caller's process-noise draws ``wk``."""
    L = require()
    x0 = np.ascontiguousarray(np.asarray(x0, dtype=np.float64).reshape(n))
    P0 = np.ascontiguousarray(np.asarray(P0, dtype=np.float64).reshape(n, n))
    if uk is not None and l > 0:
        uk = np.ascontiguousarray(np.asarray(uk, dtype=np.float64).reshape(N, l))
    if wk is not None:
        wk = np.ascontiguousarray(np.asarray(wk, dtype=np.float64).reshape(N, n))
    xk, Pk = np.empty((N, n)), np.empty((N, n, n))
    rc = L.kalman_propagate(model_handle, engine_handle, int(N), cdptr(x0), cdptr(P0), float(t0), float(fs),
                            cdptr(uk) if (uk is not None and l > 0) else ffi.NULL,
                            cdptr(wk) if wk is not None else ffi.NULL, dptr(xk), dptr(Pk))
    check(rc, "kalman_propagate")
    return xk, Pk


# --- differentiators -----------------------------------------------------------
DIFFERENTIATOR_KINDS = {"analytic": 0, "forward": 1, "central": 2}
_maps: list = []                 # the maps being differentiated, innermost last
_map_cb = None


def _map_callback():
    global _map_cb
    if _map_cb is None:
        @ffi.callback("int(void *, const double *, double *)")
        def call(_user, x, y):
            frame = _maps[-1]
            try:
                n, m = frame["n"], frame["m"]
                point = np.frombuffer(ffi.buffer(x, n * 8), dtype=np.float64).reshape(n, 1).copy()
                value = np.asarray(frame["fn"](point), dtype=np.float64).reshape(-1)
                if value.size != m:
                    raise ValueError(f"the map returned {value.size} values, expected {m}.")
                ffi.buffer(y, m * 8)[:] = value.tobytes()
                return 0
            except BaseException as e:           # must not cross the C boundary
                frame["raised"] = e
                return -1
        _map_cb = call
    return _map_cb


def jacobian(scheme, fn, x, y0, step=0.0):
    """``d fn / dx`` at the column ``x`` by the C++ differentiator ``scheme``
    (``"forward"`` or ``"central"``); ``y0 = fn(x)``, which also gives the number of rows."""
    L = require()
    point = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    base = np.ascontiguousarray(np.asarray(y0, dtype=np.float64).reshape(-1))
    J = np.empty((base.size, point.size))
    frame = {"fn": fn, "n": point.size, "m": base.size, "raised": None}
    _maps.append(frame)
    try:
        rc = L.kalman_jacobian(DIFFERENTIATOR_KINDS[scheme], float(step), _map_callback(), ffi.NULL,
                               point.size, base.size, cdptr(point), cdptr(base), dptr(J))
    finally:
        _maps.pop()
    if frame["raised"] is not None:
        raise frame["raised"]
    check(rc, "kalman_jacobian")
    return J


# --- integrators ---------------------------------------------------------------
INTEGRATOR_KINDS = {"euler": 0, "rk4": 1, "dormand_prince": 2}
_derivatives: list = []          # the derivatives being integrated, innermost last
_derivative_cb = None


def _derivative_callback():
    """One C callback for every integration: it serves the derivative on top of
    the stack, so no callback is built per call."""
    global _derivative_cb
    if _derivative_cb is None:
        @ffi.callback("int(void *, const double *, double, double *)")
        def call(_user, x, t, dx):
            frame = _derivatives[-1]
            try:
                n = frame["n"]
                state = np.frombuffer(ffi.buffer(x, n * 8), dtype=np.float64).reshape(frame["shape"]).copy()
                d = np.asarray(frame["f"](state, frame["u"], t), dtype=np.float64).reshape(-1)
                if d.size != n:
                    raise ValueError(f"the derivative returned {d.size} values for a state of {n}.")
                ffi.buffer(dx, n * 8)[:] = d.tobytes()
                return 0
            except BaseException as e:           # must not cross the C boundary
                frame["raised"] = e
                return -1
        _derivative_cb = call
    return _derivative_cb


def integrate(kind, f, x, u=None, t=0.0, dt=1.0, n=1, rtol=0.0, atol=0.0, max_substeps=0):
    """``x`` advanced by ``n`` steps of ``dt`` with the C++ integrator ``kind``
    (``"euler"``, ``"rk4"``, ``"dormand_prince"``); ``f(x, u, t)`` is the derivative,
    called with ``x`` in the shape it was given in. Returns that shape."""
    L = require()
    state = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    out = np.empty_like(state)
    frame = {"f": f, "u": u, "n": state.size, "shape": state.shape, "raised": None}
    _derivatives.append(frame)
    try:
        rc = L.kalman_integrate(INTEGRATOR_KINDS[kind], float(rtol), float(atol), int(max_substeps),
                                _derivative_callback(), ffi.NULL, state.size, cdptr(state),
                                float(t), float(dt), int(n), dptr(out))
    finally:
        _derivatives.pop()
    if frame["raised"] is not None:
        raise frame["raised"]
    check(rc, "kalman_integrate")
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

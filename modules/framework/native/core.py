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

const char *kalman_model_export_json(const kalman_model_t *model, const char *registered, const char *plugin,
                                     int linearize, int steps, const char *origin);
int kalman_model_save(const kalman_model_t *model, const char *path, const char *registered, const char *plugin,
                      int linearize, int steps, const char *origin);
kalman_model_t *kalman_model_load(const char *path);
const char *kalman_model_structure(const kalman_model_t *model);
kalman_model_t *kalman_model_open(const char *path, int settings);
int kalman_model_document_save(const char *document, const char *path);
kalman_model_t *kalman_model_create_matrices(const char *name, int n, int m, int l);
int kalman_model_set_matrix(kalman_model_t *model, const char *which, const double *data);
int kalman_model_set_table(kalman_model_t *model, const char *which, int steps, const double *data);
int kalman_model_set_component(kalman_model_t *model, int state, int i, const char *name, const char *unit);
int kalman_model_set_structure(kalman_model_t *model, int kind);
const char *kalman_model_state_name(const kalman_model_t *model, int i);
const char *kalman_model_state_unit(const kalman_model_t *model, int i);
const char *kalman_model_measurement_name(const kalman_model_t *model, int i);
const char *kalman_model_measurement_unit(const kalman_model_t *model, int i);
int kalman_model_array_count(const kalman_model_t *model);
const char *kalman_model_array_name(const kalman_model_t *model, int i);
int kalman_model_array_size(const kalman_model_t *model, const char *name);
int kalman_model_array(const kalman_model_t *model, const char *name, double *values);
int kalman_model_set_array(kalman_model_t *model, const char *name, int size, const double *values);
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
int kalman_solve_vorticity(int ny, int nx, double dt, double Ly, double Lx,
                           double nu, double beta, const double *forcing_re, const double *forcing_im,
                           int integrator, const double *w, int batch, int nsteps, double *out);
int kalman_solve_diffusion(int ny, int nx, double dt, double Ly, double Lx,
                           int boundary, double boundary_value, double nu,
                           int integrator, const double *w, int nsteps, double *out);
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

/* --- controllers --- */
typedef struct kalman_controller kalman_controller_t;
typedef int (*kalman_clip_fn)(void *user, const double *u, double *u_out);
int kalman_model_callback_set_clip(kalman_model_t *model, kalman_clip_fn clip);
kalman_controller_t *kalman_controller_create_proportional(kalman_model_t *model, const double *Kp,
                                                           const double *x_ref, const double *Qcost,
                                                           const double *Rcost);
kalman_controller_t *kalman_controller_create_pid(kalman_model_t *model, const double *Kp, const double *Ki,
                                                  const double *Kd, const double *x_ref, const double *Qcost,
                                                  const double *Rcost);
kalman_controller_t *kalman_controller_create_lqr(kalman_model_t *model, const double *x_ref,
                                                  const double *Qcost, const double *Rcost);
kalman_controller_t *kalman_controller_create_mppi(kalman_model_t *model, int n_rollouts, int horizon,
                                                   double noise_sigma, double lambda, int n_iters,
                                                   unsigned long long seed, const double *x_ref,
                                                   const double *Qcost, const double *Rcost);
kalman_controller_t *kalman_controller_create_boost(kalman_model_t *model, double boost_factor,
                                                    double threshold, int window, const double *x_ref,
                                                    const double *Qcost, const double *Rcost);
kalman_controller_t *kalman_controller_create_adaptive_qr(kalman_model_t *model, int window, double q_min,
                                                          double q_max, const double *x_ref,
                                                          const double *Qcost, const double *Rcost);
void kalman_controller_destroy(kalman_controller_t *controller);
int kalman_controller_n_control(const kalman_controller_t *controller);
int kalman_controller_policy(kalman_controller_t *controller, const double *x, const double *P, double t,
                             const double *nu, double *u);
int kalman_controller_update(kalman_controller_t *controller, const double *x, const double *P, double t,
                             const double *nu, double *scale);
int kalman_controller_command(kalman_controller_t *controller, const double *x, const double *P, double t,
                              const double *nu, double *u, double *cost);
int kalman_controller_reset(kalman_controller_t *controller);
int kalman_controller_lqr_fit(kalman_controller_t *controller, const double *A, const double *B, double *K);
int kalman_controller_lqr_set_gain(kalman_controller_t *controller, const double *K);
int kalman_controller_mppi_set_draws(kalman_controller_t *controller, int count, const double *draws);
int kalman_solve_dare(int n, int l, const double *A, const double *B, const double *Q, const double *R,
                      double *P, double *K);

/* --- routers --- */
typedef struct kalman_router kalman_router_t;
kalman_router_t *kalman_router_create_bayesian(void);
kalman_router_t *kalman_router_create_threshold(int m, int window, int soft);
kalman_router_t *kalman_router_create_hybrid(const kalman_router_t *a, const kalman_router_t *b, double alpha);
typedef int (*kalman_route_fn)(void *user, int rows, int M, const double *nis, const double *c,
                               const double *likelihoods, double *mu);
kalman_router_t *kalman_router_create_callback(kalman_route_fn route, void *user);
void kalman_router_destroy(kalman_router_t *router);
int kalman_router_route(const kalman_router_t *router, int rows, int M, const double *nis, const double *c,
                        const double *likelihoods, double *mu);

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
int enkf_analysis(int n, int N, int p,
                  const double *X, const double *Y, const double *R, const double *z, const double *eps,
                  const double *rho_xy, const double *rho_yy, double *X_out);

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


# --- solvers (fields) ------------------------------------------------------------
BOUNDARY_KINDS = {"periodic": 0, "dirichlet": 1, "neumann": 2}


def solve_vorticity(shape, dt, length, nu, beta, forcing_hat, integrator, w, nsteps):
    """``w`` (batch, ny, nx) advanced by Kalman::Solvers::Vorticity; ``integrator``
    is ``"euler"``, ``"rk4"`` or ``"dormand_prince"``."""
    L = require()
    ny, nx = int(shape[0]), int(shape[1])
    x = np.ascontiguousarray(np.asarray(w, dtype=np.float64).reshape(-1, ny, nx))
    out = np.empty_like(x)
    fre = fim = ffi.NULL
    if forcing_hat is not None:
        f = np.asarray(forcing_hat, dtype=complex)
        re, im = np.ascontiguousarray(f.real), np.ascontiguousarray(f.imag)
        fre, fim = cdptr(re), cdptr(im)
    rc = L.kalman_solve_vorticity(ny, nx, float(dt), float(length[0]), float(length[1]), float(nu), float(beta),
                                  fre, fim, INTEGRATOR_KINDS[integrator], cdptr(x), x.shape[0], int(nsteps), dptr(out))
    check(rc, "kalman_solve_vorticity")
    return out


def solve_diffusion(shape, dt, length, boundary, boundary_value, nu, integrator, w, nsteps):
    """``w`` (ny, nx) advanced by Kalman::Solvers::FiniteDifference (diffusion)."""
    L = require()
    ny, nx = int(shape[0]), int(shape[1])
    x = np.ascontiguousarray(np.asarray(w, dtype=np.float64).reshape(ny, nx))
    out = np.empty_like(x)
    rc = L.kalman_solve_diffusion(ny, nx, float(dt), float(length[0]), float(length[1]), BOUNDARY_KINDS[boundary],
                                  float(boundary_value), float(nu), INTEGRATOR_KINDS[integrator], cdptr(x),
                                  int(nsteps), dptr(out))
    check(rc, "kalman_solve_diffusion")
    return out


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


# --- controllers ---------------------------------------------------------------
# Kalman::Controllers by the name the Python classes use. Each Python controller
# (framework/controllers/) holds one of these handles and forwards its policy
# and its adaptation to it; see KalmanController._native_handle.
def _optional(array, shape):
    """``(const double *, keepalive)`` for an optional matrix, NULL for ``None``."""
    if array is None:
        return ffi.NULL, None
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float64).reshape(shape))
    return cdptr(a), a


def create_controller(kind, model_handle, n, L, *, x_ref=None, Qcost=None, Rcost=None, **params):
    """A C++ controller (garbage-collected handle) of ``kind`` — ``"proportional"``,
    ``"pid"``, ``"lqr"``, ``"mppi"``, ``"boost"``, ``"adaptive_qr"`` — on the model
    ``model_handle`` (n states, command dimension ``L = max(l, 1)``), with the
    parameters of the C++ constructor. ``model_handle`` must outlive it."""
    lib_ = require()
    keep = []

    def opt(array, shape):
        ptr, alive = _optional(array, shape)
        keep.append(alive)
        return ptr

    costs = (opt(x_ref, (n,)), opt(Qcost, (n, n)), opt(Rcost, (L, L)))
    if kind == "proportional":
        h = lib_.kalman_controller_create_proportional(model_handle, opt(params["Kp"], (L, n)), *costs)
    elif kind == "pid":
        h = lib_.kalman_controller_create_pid(model_handle, opt(params["Kp"], (L, n)), opt(params.get("Ki"), (L, n)),
                                              opt(params.get("Kd"), (L, n)), *costs)
    elif kind == "lqr":
        h = lib_.kalman_controller_create_lqr(model_handle, *costs)
    elif kind == "mppi":
        h = lib_.kalman_controller_create_mppi(model_handle, int(params["n_rollouts"]), int(params["horizon"]),
                                               float(params["noise_sigma"]), float(params["lam"]),
                                               int(params["n_iters"]), int(params.get("seed") or 0), *costs)
    elif kind == "boost":
        h = lib_.kalman_controller_create_boost(model_handle, float(params["boost_factor"]),
                                                float(params["threshold"]), int(params["window"]), *costs)
    elif kind == "adaptive_qr":
        h = lib_.kalman_controller_create_adaptive_qr(model_handle, int(params["window"]), float(params["q_min"]),
                                                      float(params["q_max"]), *costs)
    else:
        raise KeyError(f"no C++ controller {kind!r}")
    if h == ffi.NULL:
        raise RuntimeError(f"create_controller({kind!r}): {last_error()}")
    return ffi.gc(h, lib_.kalman_controller_destroy)


def _buf(a):
    """A ``double[]`` view of a C-contiguous float64 array (cheaper than ``cdptr``,
    which goes through ctypes: 0.15 against 0.5 us, and these run once per step)."""
    return ffi.from_buffer("double[]", a)


def _vector(a, size):
    a = np.asarray(a, dtype=np.float64)
    if a.size != size:
        raise ValueError(f"expected {size} values, got {a.size}.")
    return _buf(np.ascontiguousarray(a).reshape(-1))


def _belief(x, P, nu, n, m):
    return (_vector(x, n), ffi.NULL if P is None else _vector(P, n * n),
            ffi.NULL if nu is None else _vector(nu, m))


def controller_policy(handle, x, P, t, nu, n, m, L):
    """Controller::Policy: the command (L, 1) for the belief ``x`` (n), ``P`` (n, n
    or None) and the last innovation ``nu`` (m, or None on the first step)."""
    u = np.empty((L, 1))
    xs, Ps, nus = _belief(x, P, nu, n, m)
    check(lib.kalman_controller_policy(handle, xs, Ps, float(t), nus, _buf(u)), "kalman_controller_policy")
    return u


def controller_update(handle, x, P, t, nu, n, m):
    """Controller::Update: the factor the controller rescaled its model by in this
    call (R's for a Boost, Q's for an AdaptiveQR), or ``None`` when it did not."""
    xs, Ps, nus = _belief(x, P, nu, n, m)
    scale = ffi.new("double *")
    rc = check(lib.kalman_controller_update(handle, xs, Ps, float(t), nus, scale), "kalman_controller_update")
    return float(scale[0]) if rc == 1 else None


def controller_command(handle, x, P, t, nu, n, m, L):
    """Controller::Command: ``(u (L, 1), cost)`` — the policy, the update and the
    cost the C++ controller logs (charged on the clipped command)."""
    xs, Ps, nus = _belief(x, P, nu, n, m)
    u, cost = np.empty((L, 1)), ffi.new("double *")
    check(lib.kalman_controller_command(handle, xs, Ps, float(t), nus, _buf(u), cost), "kalman_controller_command")
    return u, float(cost[0])


def controller_reset(handle):
    check(lib.kalman_controller_reset(handle), "kalman_controller_reset")


def lqr_fit(handle, A, B, n, L):
    """Fit the C++ LQR to ``A`` (n, n), ``B`` (n, L); returns the gain (L, n)."""
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64).reshape(n, n))
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64).reshape(n, L))
    K = np.empty((L, n))
    check(lib.kalman_controller_lqr_fit(handle, cdptr(A), cdptr(B), dptr(K)), "kalman_controller_lqr_fit")
    return K


def lqr_set_gain(handle, K, n, L):
    K = np.ascontiguousarray(np.asarray(K, dtype=np.float64).reshape(L, n))
    check(lib.kalman_controller_lqr_set_gain(handle, cdptr(K)), "kalman_controller_lqr_set_gain")


def mppi_set_draws(handle, draws):
    """The perturbations the next MPPI policy call uses (iteration, rollout, step, command order)."""
    d = np.ascontiguousarray(np.asarray(draws, dtype=np.float64).reshape(-1))
    check(lib.kalman_controller_mppi_set_draws(handle, int(d.size), cdptr(d)), "kalman_controller_mppi_set_draws")


def solve_dare(A, B, Q, R):
    """``(P, K)``: the stabilising solution of the discrete algebraic Riccati equation
    and its gain ``K = (R + B^T P B)^-1 B^T P A`` (Kalman::Algebra::SolveDARE)."""
    L_ = require()
    A = np.ascontiguousarray(np.asarray(A, dtype=np.float64))
    B = np.ascontiguousarray(np.asarray(B, dtype=np.float64))
    n, l = B.shape
    Q = np.ascontiguousarray(np.asarray(Q, dtype=np.float64).reshape(n, n))
    R = np.ascontiguousarray(np.asarray(R, dtype=np.float64).reshape(l, l))
    P, K = np.empty((n, n)), np.empty((l, n))
    check(L_.kalman_solve_dare(n, l, cdptr(A.reshape(n, n)), cdptr(B), cdptr(Q), cdptr(R), dptr(P), dptr(K)),
          "kalman_solve_dare")
    return P, K


# --- routers -------------------------------------------------------------------
def create_router(kind, *, a=None, b=None, route=None, **params):
    """A C++ IKF router (garbage-collected handle): ``"bayesian"``, ``"threshold"``
    (``m``, ``window``, ``soft``), ``"hybrid"`` (children handles ``a``, ``b`` and
    ``alpha``; the caller keeps them alive) or ``"callback"`` (``route`` a
    ``kalman_route_fn`` callback, kept alive by the caller)."""
    L_ = require()
    if kind == "bayesian":
        h = L_.kalman_router_create_bayesian()
    elif kind == "threshold":
        h = L_.kalman_router_create_threshold(int(params["m"]), int(params["window"]), 1 if params["soft"] else 0)
    elif kind == "hybrid":
        h = L_.kalman_router_create_hybrid(a, b, float(params["alpha"]))
    elif kind == "callback":
        h = L_.kalman_router_create_callback(route, ffi.NULL)
    else:
        raise KeyError(f"no C++ router {kind!r}")
    if h == ffi.NULL:
        raise RuntimeError(f"create_router({kind!r}): {last_error()}")
    return ffi.gc(h, L_.kalman_router_destroy)


def route(handle, nis, c, likelihoods):
    """Router::Route: the mode probabilities (M,) from the NIS history ``nis``
    (rows, M; rows may be 0), ``c`` (M,) and ``likelihoods`` (M,)."""
    c = np.asarray(c, dtype=np.float64)
    M = c.size
    nis = np.asarray(nis, dtype=np.float64)
    rows = nis.size // M if M else 0
    mu = np.empty(M)
    check(lib.kalman_router_route(handle, rows, M, _vector(nis, rows * M) if rows else ffi.NULL, _vector(c, M),
                                  _vector(likelihoods, M), _buf(mu)), "kalman_router_route")
    return mu


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

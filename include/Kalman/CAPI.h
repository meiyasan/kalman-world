/**
 * *********************************************
 *
 * \file CAPI.h
 * \brief The C interface of libKalmanFramework
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details A flat `extern "C"` surface over Kalman::Factory and Kalman::Model, for
 * consumers that cannot see C++ — the Python package (through cffi in ABI mode,
 * so no compiler is needed on the Python side), a Julia or Rust binding, a C
 * program. It is deliberately small: enough to **discover** registered models,
 * **construct** one, and **evaluate** its coefficients and maps. Running a filter
 * on the result is the caller's business — the Python engines do it, and so does
 * Kalman::Filter for a C++ caller who has no reason to come through here.
 *
 * Conventions, the same for every function:
 *
 *  - Matrices are row-major contiguous `double` arrays; the caller allocates them
 *    from the model's dimensions (`kalman_model_dims`).
 *  - Functions returning `int` give `0` on success and `-1` on failure, and a
 *    failure leaves a message in `kalman_last_error()` (thread-local). No
 *    exception ever crosses this boundary.
 *  - Strings returned as `const char *` point to thread-local storage that is
 *    valid until the next call from the same thread; copy them if you keep them.
 *
 * This header is also the contract the Python side (`native/core.py`) copies its
 * `cdef` from, and a test compares the two — so a prototype added here has to
 * appear there as well, which is the point.
 *
 * *********************************************
 */

#ifndef KALMAN_CAPI_H
#define KALMAN_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/// An instance created through the factory. Opaque; destroy it with kalman_model_destroy().
typedef struct kalman_model kalman_model_t;

/// An estimation engine. Opaque; destroy it with kalman_engine_destroy().
typedef struct kalman_engine kalman_engine_t;

/// What kalman_factory_describe() fills in. String fields are thread-local, see above.
typedef struct kalman_model_info
{
    const char *name;
    const char *title;
    const char *description;
    const char *version;
    const char *plugin;      /* library path it came from, "" for the host binary */
    int n;                   /* state dimension */
    int m;                   /* measurement dimension */
    int l;                   /* control dimension */
    double fs;               /* sample rate [Hz] */
    int engine;              /* Kalman::EngineKind as an integer */
    int analytic;            /* 1 = analytic Jacobian, 0 = numerical */
} kalman_model_info_t;

/* --- library ------------------------------------------------------------ */

/// KALMAN_VERSION of the library actually loaded (not of the header you compiled against).
const char *kalman_version(void);

/// Message of the last failure on this thread; "" when the last call succeeded.
const char *kalman_last_error(void);

/* --- factory ------------------------------------------------------------ */

/// Number of registered models, or -1.
int kalman_factory_count(void);

/// Name of the i-th registered model in sorted order, or NULL if i is out of range.
const char *kalman_factory_name(int i);

/// 1 if `name` is registered, 0 if not, -1 on error.
int kalman_factory_has(const char *name);

/// Fill `info` for `name`. Builds and discards an instance to read the dimensions.
int kalman_factory_describe(const char *name, kalman_model_info_t *info);

/**
 * dlopen a shared library whose static initialisers register models.
 * Returns the number of models it added (0 when it was already loaded), or -1
 * with the loader's message in kalman_last_error(). The library is never unloaded.
 *
 * Executes code: the caller decides which paths are acceptable, not this function.
 */
int kalman_factory_load(const char *path);

/// Number of loaded plugins, or -1.
int kalman_factory_plugin_count(void);

/// Path of the i-th loaded plugin, in load order, or NULL.
const char *kalman_factory_plugin(int i);

/* --- model: lifetime and identity ---------------------------------------- */

/// Construct `name` through the factory, or NULL with kalman_last_error() set.
kalman_model_t *kalman_model_create(const char *name);

/// Destroy an instance. NULL is accepted and ignored.
void kalman_model_destroy(kalman_model_t *model);

/// The instance's Name() (what the application called this system).
const char *kalman_model_name(const kalman_model_t *model);

/// Title / Description / Version of the model's *type*.
const char *kalman_model_title(const kalman_model_t *model);
const char *kalman_model_description(const kalman_model_t *model);
const char *kalman_model_model_version(const kalman_model_t *model);

/// Model::Describe() — the one-line summary used in logs.
const char *kalman_model_describe(const kalman_model_t *model);

/* --- model: dimensions and time base -------------------------------------- */

/// (n, m, l). Any output pointer may be NULL.
int kalman_model_dims(const kalman_model_t *model, int *n, int *m, int *l);

double kalman_model_sample_rate(const kalman_model_t *model);
int    kalman_model_set_sample_rate(kalman_model_t *model, double fs);

double kalman_model_t0(const kalman_model_t *model);
int    kalman_model_set_t0(kalman_model_t *model, double t0);

/// Kalman::EngineKind of the model's default engine as an integer, or -1.
int kalman_model_engine(const kalman_model_t *model);

/// 1 when F()/H() are the Jacobians, 0 when they must be differenced numerically, -1 on error.
int kalman_model_analytic_jacobian(const kalman_model_t *model);

/// Model::CheckShapes(); the mismatch message, if any, is in kalman_last_error().
int kalman_model_check_shapes(const kalman_model_t *model);

/* --- model: initial state ------------------------------------------------ */

int kalman_model_x0(const kalman_model_t *model, double *x0);          /* n     */
int kalman_model_P0(const kalman_model_t *model, double *P0);          /* n x n */
int kalman_model_set_x0(kalman_model_t *model, const double *x0);      /* n     */
int kalman_model_set_P0(kalman_model_t *model, const double *P0);      /* n x n */

/* --- model: coefficients at time t ---------------------------------------- */

int kalman_model_F(const kalman_model_t *model, double t, double *F);  /* n x n */
int kalman_model_H(const kalman_model_t *model, double t, double *H);  /* m x n */
int kalman_model_Q(const kalman_model_t *model, double t, double *Q);  /* n x n */
int kalman_model_R(const kalman_model_t *model, double t, double *R);  /* m x m */

/**
 * B (n x l) and L (n x n) may be *empty* in the C++ model — no control map,
 * identity noise coupling. These return 1 and fill the buffer when the matrix is
 * present, 0 and leave it untouched when it is empty, -1 on error.
 */
int kalman_model_B(const kalman_model_t *model, double t, double *B);
int kalman_model_L(const kalman_model_t *model, double t, double *L);

/* --- model: maps ---------------------------------------------------------- */

/// x_next = f(x, u, t). `u` may be NULL when l == 0. Writes n values.
int kalman_model_f(const kalman_model_t *model, const double *x, const double *u, double t, double *x_next);

/// z = h(x, t). Writes m values.
int kalman_model_h(const kalman_model_t *model, const double *x, double t, double *z);

/// df/dx at (x, u, t) — analytic F or central differences. Writes n x n.
int kalman_model_jacobian(const kalman_model_t *model, const double *x, const double *u, double t, double *J);

/// dh/dx at (x, t). Writes m x n.
int kalman_model_jacobian_h(const kalman_model_t *model, const double *x, double t, double *J);

/* --- engines: the C++ estimation step, driven from outside ------------------ */

/**
 * An engine of `kind` (Kalman::EngineKind as an integer) with default settings,
 * or NULL with kalman_last_error() set. The configured constructors below take
 * the same parameters as the C++ classes.
 */
kalman_engine_t *kalman_engine_create(int kind);
kalman_engine_t *kalman_engine_create_unscented(double alpha, double beta, double kappa);
kalman_engine_t *kalman_engine_create_particle(int n_particles, double threshold, unsigned long long seed);
kalman_engine_t *kalman_engine_create_ensemble(int n_members, double inflation, unsigned long long seed);
void kalman_engine_destroy(kalman_engine_t *engine);

/// The engine's kind as an integer, or -1.
int kalman_engine_kind(const kalman_engine_t *engine);
/// 1 when the engine has an RTS smoother, 0 when it is a sample set, -1 on error.
int kalman_engine_has_smoother(const kalman_engine_t *engine);
/// Start a run: a stateful engine reseeds its cloud / ensemble from the model's x0 / P0.
int kalman_engine_reset(kalman_engine_t *engine, const kalman_model_t *model);

/// Predict: (x, P) -> (x_out, P_out) with the transition F_out (n x n). `u` may be NULL.
int kalman_engine_predict(const kalman_engine_t *engine, const kalman_model_t *model,
                          const double *x, const double *P, const double *u, double t,
                          double *x_out, double *P_out, double *F_out);

/// Update with `z` (m; NaN marks a missing channel). Returns the channels used, or -1.
int kalman_engine_update(const kalman_engine_t *engine, const kalman_model_t *model,
                         const double *x, const double *P, const double *z, double t,
                         double *x_out, double *P_out);

/// Innovation nu = z - h(x) (m) and its covariance S (m x m) at the prediction (x, P).
int kalman_engine_innovation(const kalman_engine_t *engine, const kalman_model_t *model,
                             const double *x, const double *P, const double *z, double t,
                             double *nu, double *S);

/// One reverse RTS step (all n / n x n); -1 for an engine without a smoother.
int kalman_engine_smooth(const kalman_engine_t *engine, int n,
                         const double *x_post, const double *P_post,
                         const double *x_prior_next, const double *P_prior_next,
                         const double *x_smooth_next, const double *P_smooth_next,
                         const double *F, double *x_smooth, double *P_smooth);

/* --- a whole pass, fused ---------------------------------------------------- */

/**
 * Run Kalman::Filter over N steps with `engine` on `model`: the forward pass and,
 * when `smooth` is non-zero, the RTS backward pass.
 *
 *   zk: N x m (NaN = missing channel)   uk: N x l or NULL
 *   x_prior, x_post, x_smooth: N x n    P_prior, P_post, P_smooth: N x n x n
 *   innovations: N x m                  transitions: N x n x n
 *
 * Every output pointer may be NULL to skip that quantity; x_smooth / P_smooth are
 * only written when `smooth` is set. Returns 0, or -1 with the reason.
 */
int kalman_filter_run(const kalman_model_t *model, kalman_engine_t *engine, int N,
                      const double *zk, const double *uk, int smooth,
                      double *x_prior, double *P_prior, double *x_post, double *P_post,
                      double *x_smooth, double *P_smooth, double *innovations, double *transitions);

#ifdef __cplusplus
}
#endif

#endif /* KALMAN_CAPI_H */

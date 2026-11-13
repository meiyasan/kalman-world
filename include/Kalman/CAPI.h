/**
 * *********************************************
 *
 * \file CAPI.h
 * \brief The C interface of libKalman
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

/* --- model: maps supplied by the caller ----------------------------------- */

/// `x_next = f(x, u, t)`; `u` is NULL when the model has no control. Return 0 on success.
typedef int (*kalman_step_fn)(void *user, const double *x, const double *u, double t, double *x_next);
/// `z = h(x, t)`. Return 0 on success.
typedef int (*kalman_measure_fn)(void *user, const double *x, double t, double *z);
/// `J = df/dx`, row-major (n x n). Return 0 on success.
typedef int (*kalman_step_jacobian_fn)(void *user, const double *x, const double *u, double t, double *J);
/// `J = dh/dx`, row-major (m x n). Return 0 on success.
typedef int (*kalman_measure_jacobian_fn)(void *user, const double *x, double t, double *J);

/// A model whose f and h are the caller's functions (Kalman::Models::Callback).
/// Either Jacobian may be NULL, in which case it is differenced numerically.
/// The functions and `user` must outlive the model. NULL with kalman_last_error() on failure.
kalman_model_t *kalman_model_create_callback(const char *name, int n, int m, int l,
                                             kalman_step_fn f, kalman_measure_fn h,
                                             kalman_step_jacobian_fn jacobian_f,
                                             kalman_measure_jacobian_fn jacobian_h, void *user);

/// The process-noise covariance over the step from (x, u, t), row-major (n x n),
/// with any noise coupling already applied. Return 0 on success.
typedef int (*kalman_process_noise_fn)(void *user, const double *x, const double *u, double t, double *Q);
/// The measurement-noise covariance at t, row-major (m x m). Return 0 on success.
typedef int (*kalman_measurement_noise_fn)(void *user, double t, double *R);

/// Give a callback model its noise covariances as functions (either may be NULL,
/// which keeps the constant matrix). -1 when `model` is not a callback model.
int kalman_model_callback_set_noise(kalman_model_t *model, kalman_process_noise_fn process_noise,
                                    kalman_measurement_noise_fn measurement_noise);

/// `x_next = f(x, u, w, t)`: the step with its process noise `w` (length n) inside. Return 0 on success.
typedef int (*kalman_noisy_step_fn)(void *user, const double *x, const double *u, const double *w, double t, double *x_next);
/// `z = h(x, v, t)`: the observation with its noise `v` (length m) inside. Return 0 on success.
typedef int (*kalman_noisy_measure_fn)(void *user, const double *x, const double *v, double t, double *z);

/// Give a callback model the maps that take their noise as an argument (either
/// may be NULL). The unscented engine then augments the state with that noise.
int kalman_model_callback_set_noise_inputs(kalman_model_t *model, kalman_noisy_step_fn f_with_noise,
                                           kalman_noisy_measure_fn h_with_noise);

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

/* --- model: files ------------------------------------------------------------ */

/// The model.json text of the model written out (see Kalman::IO::ModelFile), or NULL.
/// `registered` is the factory name a reference refers to and `plugin` the library it
/// came from (either may be NULL); `linearize` != 0 writes the Jacobians at the initial
/// state; `steps` is the number of samples tabulated for a time-varying model; `origin`,
/// a JSON object as text or NULL, replaces the "origin" member (where the model came from).
const char *kalman_model_export_json(const kalman_model_t *model, const char *registered, const char *plugin,
                                     int linearize, int steps, const char *origin);
/// Write the model to `path`: one archive (.kw), or plain JSON when the path ends in .json. 0 or -1.
int kalman_model_save(const kalman_model_t *model, const char *path, const char *registered, const char *plugin,
                      int linearize, int steps, const char *origin);
/// Build the model a file describes (archive or JSON), or NULL. The settings file beside it
/// (`path` + ".json"), when there is one, is applied. Destroy it with kalman_model_destroy().
kalman_model_t *kalman_model_load(const char *path);
/// "maps", "matrices" or "time-varying matrices": what the model is made of. Never NULL for a valid handle.
const char *kalman_model_structure(const kalman_model_t *model);

/// Build the model a file describes; `settings` != 0 applies the settings file beside it. NULL on failure.
kalman_model_t *kalman_model_open(const char *path, int settings);
/// Write a model.json document (text) to `path` as a model file. For a document without tables. 0 or -1.
int kalman_model_document_save(const char *document, const char *path);

/* --- model: given by matrices ------------------------------------------------ */

/// A linear model given by its matrices (Kalman::Models::Tabulated), to be filled with the
/// setters below, kalman_model_set_Q/R/x0/P0 and kalman_model_set_generator. NULL on failure.
kalman_model_t *kalman_model_create_matrices(const char *name, int n, int m, int l);
/// Set "F" (n x n), "H" (m x n), "B" (n x l) or "L" (n x n), row-major. 0 or -1.
int kalman_model_set_matrix(kalman_model_t *model, const char *which, const double *data);
/// One matrix per sample for "F", "H", "Q" or "R": `steps` matrices, row-major, one after
/// the other. Only for a model made by kalman_model_create_matrices or read from a file. 0 or -1.
int kalman_model_set_table(kalman_model_t *model, const char *which, int steps, const double *data);
/// Name component i of the state (`state` != 0) or of the measurement. Same models as above. 0 or -1.
int kalman_model_set_component(kalman_model_t *model, int state, int i, const char *name, const char *unit);
/// Declare what the model is made of: 0 maps, 1 matrices, 2 time-varying matrices. 0 or -1.
int kalman_model_set_structure(kalman_model_t *model, int kind);

/* --- model: named components ------------------------------------------------ */

/// Name of component i of the state (0 <= i < n), or NULL. "x<i>" when the model names nothing.
const char *kalman_model_state_name(const kalman_model_t *model, int i);
/// Unit of component i of the state ("" when not stated), or NULL.
const char *kalman_model_state_unit(const kalman_model_t *model, int i);
/// Name of component i of the measurement (0 <= i < m), or NULL. "z<i>" when the model names nothing.
const char *kalman_model_measurement_name(const kalman_model_t *model, int i);
/// Unit of component i of the measurement ("" when not stated), or NULL.
const char *kalman_model_measurement_unit(const kalman_model_t *model, int i);

/* --- model: named parameters and noise covariances ------------------------- */

/// Number of named arrays the model declares, or -1.
int kalman_model_array_count(const kalman_model_t *model);
/// Name of array i (sorted order), or NULL.
const char *kalman_model_array_name(const kalman_model_t *model, int i);
/// Length of the array `name`, or -1 (and a message) when the name is unknown.
int kalman_model_array_size(const kalman_model_t *model, const char *name);
/// Copy the array `name` into `values` (kalman_model_array_size() doubles). 0 or -1.
int kalman_model_array(const kalman_model_t *model, const char *name, double *values);
/// Set the array `name` to `size` values. The model may resize itself: read
/// kalman_model_dims() again afterwards. 0 or -1.
int kalman_model_set_array(kalman_model_t *model, const char *name, int size, const double *values);

/// Number of named parameters the model declares, or -1.
int kalman_model_parameter_count(const kalman_model_t *model);
/// Name of parameter i (sorted order), or NULL.
const char *kalman_model_parameter_name(const kalman_model_t *model, int i);
/// Read a parameter into *value; -1 (and a message) when the name is unknown.
int kalman_model_parameter(const kalman_model_t *model, const char *name, double *value);
/// Set a parameter; -1 (and a message) when the name is unknown.
int kalman_model_set_parameter(kalman_model_t *model, const char *name, double value);

/// Replace the constant process / measurement noise covariances.
int kalman_model_set_Q(kalman_model_t *model, const double *Q);        /* n x n */
int kalman_model_set_R(kalman_model_t *model, const double *R);        /* m x m */

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

/* --- the ensemble engine: what a caller that draws itself hands in ---------- */

/**
 * The initial ensemble of every run of an ensemble engine: `X` is n x N, row-major,
 * the members as columns (numpy's `(n, N)`). NULL (or N = 0) draws from x0 / P0 again.
 */
int kalman_engine_ensemble_set_members(kalman_engine_t *engine, int n, int N, const double *X);
/// The current members, n x N row-major, members as columns.
int kalman_engine_ensemble_members(const kalman_engine_t *engine, int n, int N, double *X);
/**
 * Standard normal draws in place of the engine's own stream, read as numpy's
 * `rng.standard_normal((rows, N))` blocks (see Engines::Ensemble::SetNormals).
 */
int kalman_engine_ensemble_set_normals(kalman_engine_t *engine, long long count, const double *normals);
/// The draws handed in and not used yet, or -1.
long long kalman_engine_ensemble_normals_left(const kalman_engine_t *engine);
/// Localization tapers rho_xy (n x m) and rho_yy (m x m); both NULL to switch it off.
int kalman_engine_ensemble_set_localization(kalman_engine_t *engine, int n, int m,
                                            const double *rho_xy, const double *rho_yy);
/**
 * The data-assimilation cycles of Kalman::Filters::EnKF::Cycle(): from a fresh
 * reset, one forecast per measurement `zk` (N x m, NaN = silent) and, when
 * `analyse`, its analysis. Writes the forecast and the analysis means (N x n
 * each) and the spread after each cycle (N); no covariance is formed.
 */
int kalman_ensemble_cycle(const kalman_model_t *model, kalman_engine_t *engine, int N, const double *zk,
                          int analyse, double *forecast, double *analysis, double *spread);

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

/**
 * A control policy for a closed loop: from the belief before the step (`x`, n;
 * `P`, n x n), the time and the previous step's innovation (`nu`, m; zeros on the
 * first step), write the command (`u`, l). Return 0, or non-zero to stop the run.
 */
typedef int (*kalman_control_fn)(void *user, const double *x, const double *P, double t,
                                 const double *nu, double *u);

/**
 * kalman_filter_run() in closed loop: at every step the command comes from
 * `control` instead of an array. `controls` (N x l, may be NULL) receives the
 * commands applied. The model must take a control (l > 0).
 */
int kalman_filter_run_closed_loop(const kalman_model_t *model, kalman_engine_t *engine, int N,
                                  const double *zk, kalman_control_fn control, void *user, int smooth,
                                  double *x_prior, double *P_prior, double *x_post, double *P_post,
                                  double *x_smooth, double *P_smooth, double *innovations,
                                  double *transitions, double *controls);

/* --- continuous models: decoding into one sample ------------------------------ */

/// exp(A) for a square matrix A (n x n), written to `out` (n x n).
int kalman_matrix_exponential(int n, const double *A, double *out);

/**
 * Decode dx/dt = A x + w, w white noise of intensity G (n x n, per unit time),
 * over a step dt: the transition F = exp(A dt) and the accumulated noise
 * covariance Q (Van Loan). Either output may be NULL.
 */
int kalman_van_loan(int n, const double *A, const double *G, double dt, double *F, double *Q);

/// 0 for a discrete model, 1 for a continuous one (a derivative or a generator).
int kalman_model_dynamics(const kalman_model_t *model);

/// State a linear model as dx/dt = A x (n x n); F is decoded for the sample rate.
int kalman_model_set_generator(kalman_model_t *model, const double *A);

/// State the process noise as an intensity Qc (q x q) entering through Lc (n x q, or NULL
/// for the identity, then q = n); Q is decoded for the sample rate.
int kalman_model_set_noise_intensity(kalman_model_t *model, int q, const double *Qc, const double *Lc);

/// The model's current F (n x n) and Q (n x n), as decoded. Either may be NULL.
int kalman_model_get_F_Q(const kalman_model_t *model, double *F, double *Q);

/* --- solvers (fields) ---------------------------------------------------------- */

/**
 * Advance `batch` vorticity fields (batch x ny x nx) by `nsteps` steps of `dt` with
 * Kalman::Solvers::Vorticity: dw/dt = -u.grad(w) - beta v + nu lap(w) + F, on a
 * periodic Ly x Lx box. `forcing_re` / `forcing_im` are the transform of F (ny x nx
 * each) or both NULL. `integrator` is 0 Euler, 1 RK4, 2 Dormand-Prince (its
 * tableau, at a fixed step). `out` has the layout of `w` and may alias it.
 */
int kalman_solve_vorticity(int ny, int nx, double dt, double Ly, double Lx,
                           double nu, double beta, const double *forcing_re, const double *forcing_im,
                           int integrator, const double *w, int batch, int nsteps, double *out);

/**
 * Advance one field (ny x nx) by `nsteps` steps of `dt` with
 * Kalman::Solvers::FiniteDifference and its default physics, diffusion
 * df/dt = nu lap(f). `boundary` is 0 periodic, 1 Dirichlet, 2 Neumann.
 */
int kalman_solve_diffusion(int ny, int nx, double dt, double Ly, double Lx,
                           int boundary, double boundary_value, double nu,
                           int integrator, const double *w, int nsteps, double *out);

/* --- sampling ------------------------------------------------------------------ */

/**
 * The state chain of a sampled run: N times, predict with `engine` (the mean
 * through the model, the covariance through its linearisation), then add the
 * process-noise draw the caller supplies. Sample k is the state after the k-th
 * step, at time t0 + k / fs.
 *
 *   x0: n            P0: n x n         uk: N x l or NULL     wk: N x n or NULL (no noise)
 *   xk: N x n        Pk: N x n x n (may be NULL)
 *
 * The draws come from the caller so that a seeded run is the same run whoever
 * executes the loop.
 */
int kalman_propagate(const kalman_model_t *model, kalman_engine_t *engine, int N,
                     const double *x0, const double *P0, double t0, double fs,
                     const double *uk, const double *wk, double *xk, double *Pk);

/* --- differentiators ----------------------------------------------------------- */

/// A map to differentiate: write y = fn(x) (m values) for x (n values). Return 0, or non-zero to stop.
typedef int (*kalman_map_fn)(void *user, const double *x, double *y);

/**
 * The Jacobian d fn / dx at `x` (n), written to `J` (m x n), by a
 * Kalman::Differentiator: `kind` 1 forward differences, 2 central differences;
 * `step` is the relative step (<= 0: the scheme's default). `y0` is fn(x) (m) when
 * the caller has it, which saves the forward scheme one evaluation; else NULL.
 */
int kalman_jacobian(int kind, double step, kalman_map_fn fn, void *user, int n, int m,
                    const double *x, const double *y0, double *J);

/// How a model's Jacobians are obtained: 0 analytic, 1 forward, 2 central.
int kalman_model_differentiator(const kalman_model_t *model);
int kalman_model_set_differentiator(kalman_model_t *model, int kind, double step);

/* --- integrators ------------------------------------------------------------ */

/**
 * The derivative of a state: write dx/dt (n) at state `x` (n) and time `t`.
 * The control, if any, is the caller's to carry in `user`. Return 0, or non-zero
 * to stop.
 */
typedef int (*kalman_derivative_fn)(void *user, const double *x, double t, double *dx);

/**
 * Advance `x` (n) by `nsteps` steps of `dt` from time `t` with a Kalman::Integrator:
 * `kind` is 0 Euler, 1 RK4, 2 Dormand-Prince (adaptive; `rtol`, `atol` and
 * `max_substeps` apply to it only, and a non-positive value means its default).
 * Writes the result to `out` (n; may alias `x`).
 */
int kalman_integrate(int kind, double rtol, double atol, int max_substeps,
                     kalman_derivative_fn f, void *user, int n,
                     const double *x, double t, double dt, int nsteps, double *out);

/* --- controllers ------------------------------------------------------------ */

/// A closed-loop controller (Kalman::Controller). Opaque; destroy it with kalman_controller_destroy().
typedef struct kalman_controller kalman_controller_t;

/// `u_out = clip(u)`, l values each: what the actuator of a callback model delivers. Return 0 on success.
typedef int (*kalman_clip_fn)(void *user, const double *u, double *u_out);

/// Give a callback model its actuator limit (Model::ClipControl); NULL restores the identity.
/// -1 when `model` is not a callback model.
int kalman_model_callback_set_clip(kalman_model_t *model, kalman_clip_fn clip);

/**
 * The shipped controllers (Kalman::Controllers), with the parameters of their C++
 * constructors. `model` is not owned and must outlive the controller. With L =
 * max(l, 1) the command dimension: gains are L x n; `x_ref` (n), `Qcost` (n x n)
 * and `Rcost` (L x L) may be NULL for the origin and the identities; `Ki`, `Kd`
 * may be NULL for zero. NULL with kalman_last_error() on failure.
 */
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

/// The command dimension L = max(l, 1), or -1.
int kalman_controller_n_control(const kalman_controller_t *controller);

/**
 * Controller::Policy(): the command `u` (L) for the belief `x` (n), `P` (n x n; NULL
 * for zeros, for a policy that does not read it), the time `t` and the last
 * innovation `nu` (m; NULL on the first step).
 */
int kalman_controller_policy(kalman_controller_t *controller, const double *x, const double *P, double t,
                             const double *nu, double *u);

/**
 * Controller::Update(), the adaptation after the policy. Returns 1 when the
 * controller rescaled its model in this call and writes the factor to `scale`
 * (may be NULL): R's for a Boost, Q's for an AdaptiveQR. 0 when it did not, -1 on error.
 */
int kalman_controller_update(kalman_controller_t *controller, const double *x, const double *P, double t,
                             const double *nu, double *scale);

/// Controller::Command(): policy, update and the logged cost, charged on the clipped
/// command. `u` (L) and `cost` may be NULL.
int kalman_controller_command(kalman_controller_t *controller, const double *x, const double *P, double t,
                              const double *nu, double *u, double *cost);

/// Controller::Reset(): clear the logs and the internal state, restore what Update() changed.
int kalman_controller_reset(kalman_controller_t *controller);

/// LQR: fit the gain to `A` (n x n) and `B` (n x L), or to the model's F and B at T0
/// when both are NULL, and write it to `K` (L x n; may be NULL). -1 for another controller.
int kalman_controller_lqr_fit(kalman_controller_t *controller, const double *A, const double *B, double *K);

/// LQR: use the gain `K` (L x n) computed elsewhere.
int kalman_controller_lqr_set_gain(kalman_controller_t *controller, const double *K);

/// MPPI: the `count` perturbations the next policy call uses instead of its own draws,
/// in the order iteration, rollout, horizon step, command (count = n_iters x n_rollouts x horizon x L).
int kalman_controller_mppi_set_draws(kalman_controller_t *controller, int count, const double *draws);

/// The stabilising solution `P` (n x n) of the discrete algebraic Riccati equation for
/// `A` (n x n), `B` (n x l), `Q` (n x n), `R` (l x l), and its gain `K` (l x n). Either output may be NULL.
int kalman_solve_dare(int n, int l, const double *A, const double *B, const double *Q, const double *R,
                      double *P, double *K);

/* --- routers ---------------------------------------------------------------- */

/// An IKF mode router (Kalman::Routers::Router). Opaque; destroy it with kalman_router_destroy().
typedef struct kalman_router kalman_router_t;

/// Routers::Bayesian: `mu_j` proportional to `c_j L_j`.
kalman_router_t *kalman_router_create_bayesian(void);

/// Routers::Threshold: the model whose NIS over the last `window` steps (0 = all) is closest to `m`;
/// `soft` non-zero for inverse-distance weights.
kalman_router_t *kalman_router_create_threshold(int m, int window, int soft);

/// Routers::Hybrid: `mu_a^(1-alpha) mu_b^alpha`, normalised. `a` and `b` are not owned and must outlive it.
kalman_router_t *kalman_router_create_hybrid(const kalman_router_t *a, const kalman_router_t *b, double alpha);

/// A router written elsewhere: write `mu` (M) from `nis` (rows x M), `c` (M) and `likelihoods` (M). Return 0 on success.
typedef int (*kalman_route_fn)(void *user, int rows, int M, const double *nis, const double *c,
                               const double *likelihoods, double *mu);

/// A router whose decision is the caller's function (to blend it in a Hybrid). `route` and `user` must outlive it.
kalman_router_t *kalman_router_create_callback(kalman_route_fn route, void *user);

void kalman_router_destroy(kalman_router_t *router);

/// Router::Route(): `mu` (M) from the NIS history `nis` (rows x M, rows may be 0), `c` (M), `likelihoods` (M).
int kalman_router_route(const kalman_router_t *router, int rows, int M, const double *nis, const double *c,
                        const double *likelihoods, double *mu);

/* --- kernels: the framework's arithmetic for a model that lives elsewhere ---
 *
 * The Python engines evaluate the model themselves -- f at the sigma points, F/H/Q/R
 * at the current point, h at every particle -- and hand the arrays here for the
 * step: covariance prediction, the NaN-masked Joseph update, a whole linear pass,
 * the RTS sweep, the sigma-point moments, particle weights and resampling, NIS and
 * NEES. One crossing per step with arrays, never a callback per point. Every one of
 * these is the same code as the C++ engines above (Kalman::Filter, Engine::Smooth,
 * Utils::Correct, Utils::WeightedOuterSum, Utils::SystematicResample), so a Python
 * model on these kernels and a C++ model in ROOT+ run one implementation.
 *
 * All arrays row-major double; dimensions n (state), m (measurement), l (control);
 * a NaN in a measurement marks a channel that did not arrive. The void functions
 * leave their outputs NaN and kalman_last_error() set if anything goes wrong.
 */

/* x_out = F x (+ B u when B and u are given and l > 0); P_out = F P F^T + L Q L^T (L NULL => + Q). */
void kf_predict(int n, int l,
                const double *F, const double *x, const double *P,
                const double *B, const double *u,
                const double *L, const double *Q,
                double *x_out, double *P_out);

/* Joseph-form update; NaN channels of z are dropped. Returns the channels used, 0 = pass-through. */
int kf_update(int n, int m,
              const double *H, const double *x, const double *P,
              const double *z, const double *R,
              double *x_out, double *P_out);

/* The same from a precomputed residual y = z - h(x) (the nonlinear engines' path). */
int kf_update_residual(int n, int m,
                        const double *H, const double *x, const double *P,
                        const double *y, const double *R,
                        double *x_out, double *P_out);

/* A whole linear forward pass -- Kalman::Filter over Engines::Linear -- with constant matrices.
 * zk: N x m, uk: N x l or NULL; outputs N x n, N x n x n, innovations N x m (NaN kept). */
void kf_forward_linear(int n, int m, int l, int N,
                        const double *F, const double *H,
                        const double *Q, const double *R,
                        const double *B, const double *L,
                        const double *zk, const double *uk,
                        const double *x0, const double *P0,
                        double *x_prior, double *P_prior,
                        double *x_post, double *P_post,
                        double *innovations);

/* The RTS sweep over N stored steps with one constant transition F. */
void kf_backward_rts(int n, int N,
                      const double *x_post, const double *P_post,
                      const double *x_prior, const double *P_prior,
                      const double *F,
                      double *x_smooth, double *P_smooth);

/* logw[i] += -0.5 (z - zp[i])^T Rinv (z - zp[i]); zp is n_particles x m. */
void pf_weight_update(int n_particles, int m,
                       const double *z, const double *zp,
                       const double *Rinv, double *logw);

/* indices[i] = the particle at cumulative position (i + u0) / n; u0 one uniform draw. */
void pf_systematic_resample(int n, const double *weights, double u0, int *indices);

/* P_out = sum_s Wc[s] (Y[s] - mean)(Y[s] - mean)^T; Y is n_sigma x n. */
void ukf_weighted_outer_sum(int n_sigma, int n,
                             const double *Wc, const double *Y, const double *mean,
                             double *P_out);

/* C_out = sum_s Wc[s] A[s] B[s]^T for already-centred rows A (n_sigma x na), B (n_sigma x nb). */
void ukf_weighted_cross_sum(int n_sigma, int na, int nb,
                             const double *Wc, const double *A, const double *B,
                             double *C_out);

/* nis[k] = nu_k^T S_k^-1 nu_k over N steps (nu: N x m, S: N x m x m). */
void diag_nis_batch(int N, int m,
                     const double *nu, const double *S, double *nis);

/* nees[k] = e_k^T P_k^-1 e_k over N steps (e: N x n, P: N x n x n), pseudo-inverse when singular. */
void diag_nees_batch(int N, int n,
                      const double *e, const double *P, double *nees);

/*
 * The ensemble (perturbed-observation) analysis, for a caller that holds the
 * ensemble and evaluates the model itself:
 *
 *   X: n x N members (already inflated)     Y: p x N, the members in observation space
 *   R: p x p        z: p (the channels that arrived)       eps: p x N draws from N(0, R)
 *   rho_xy: n x p and rho_yy: p x p localization tapers, or NULL for none
 *
 * Writes X + K (z + eps - Y) to X_out (n x N), K = (rho_xy . P H^T) (rho_yy . H P H^T + R)^-1,
 * with the covariances taken from the ensemble. Returns 0, or -1 (X_out is then NaN).
 */
int enkf_analysis(int n, int N, int p,
                  const double *X, const double *Y, const double *R, const double *z, const double *eps,
                  const double *rho_xy, const double *rho_yy, double *X_out);

/* --- track finding (the combinatorial engine) -------------------------------- */

/// The tracks one search returned; destroy with kalman_tracks_destroy().
typedef struct kalman_tracks kalman_tracks_t;

/**
 * Find tracks in one event with Kalman::Engines::Combinatorial.
 *
 * `radii` holds the n_layers layer radii, innermost first; `hits_per_layer` the
 * number of hits on each; `hits` every hit, layer after layer, m values each (NaN
 * marks a missing channel). The n_seeds seeds are given explicitly: the layer
 * and index of the hit each starts from, its state `seed_x` (n each) and
 * covariance `seed_P` (n x n each). `chi2_cut` and `hole_penalty` may be NaN to
 * take the defaults (99% of a chi-square with m degrees of freedom; one gate).
 * Returns the arbitrated tracks, best first, or NULL with kalman_last_error().
 */
kalman_tracks_t *kalman_ckf_find(const kalman_model_t *model, int n_layers, const double *radii,
                                 const int *hits_per_layer, const double *hits,
                                 int n_seeds, const int *seed_layer, const int *seed_index,
                                 const double *seed_x, const double *seed_P,
                                 double chi2_cut, double hole_penalty, int max_candidates,
                                 int max_holes, int min_hits, double shared_hit_frac);

void kalman_tracks_destroy(kalman_tracks_t *tracks);

/// Number of tracks, or -1.
int kalman_tracks_count(const kalman_tracks_t *tracks);
/// Number of hits assigned to track i, or -1.
int kalman_track_hits(const kalman_tracks_t *tracks, int i);
/// Track i's totals and final state: x (n), P (n x n), Faccum (n x n).
int kalman_track_summary(const kalman_tracks_t *tracks, int i, double *chi2, double *score, int *holes,
                         double *x, double *P, double *Faccum);
/// Track i hit by hit (H = kalman_track_hits): layers and indices (H each), radii (H),
/// xk (H x n), Pk (H x n x n), zk (H x m), prior_x (H x n), prior_P (H x n x n), Fc (H x n x n).
int kalman_track_copy(const kalman_tracks_t *tracks, int i, int *layers, int *indices, double *radii,
                      double *xk, double *Pk, double *zk, double *prior_x, double *prior_P, double *Fc);

#ifdef __cplusplus
}
#endif

#endif /* KALMAN_CAPI_H */

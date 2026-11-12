/**
 * *********************************************
 *
 * \file CAPI.h
 * \brief The C interface of libKalmanLearn
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * What the Python package calls in libKalmanLearn. Same conventions as
 * Kalman/CAPI.h: an int is 0 on success and -1 on failure, a pointer is NULL
 * on failure, and kalman_learn_last_error() holds the message.
 *
 * *********************************************
 */

#ifndef KALMAN_LEARN_CAPI_H
#define KALMAN_LEARN_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/// The last error on this thread, or "".
const char *kalman_learn_last_error(void);

/// 1 when this build of libKalmanLearn has libTorch, 0 otherwise.
int kalman_learn_has_torch(void);

/// A batched 2-D vorticity solver on libTorch (Kalman::Learn::Solvers::Vorticity).
typedef struct kalman_vorticity kalman_vorticity_t;

/// Every array holds ny * nx values, row-major. The forcing pair may be NULL.
/// `device` is "cpu", "cuda" or "mps"; one that is unavailable falls back to "cpu".
kalman_vorticity_t *kalman_vorticity_create(int ny, int nx, const double *kx, const double *ky,
                                            const double *k2, const double *k2inv, const double *dealias,
                                            double nu, double beta,
                                            const double *forcing_re, const double *forcing_im,
                                            const char *device);

void kalman_vorticity_destroy(kalman_vorticity_t *solver);

/// The device in use ("cpu", "cuda", "mps"), or NULL.
const char *kalman_vorticity_device(const kalman_vorticity_t *solver);

/// Advance `batch` fields (batch * ny * nx values) by `nsteps` steps of `dt` with the
/// explicit Runge-Kutta scheme c (stages), A (stages x stages), b (stages).
int kalman_vorticity_advance(const kalman_vorticity_t *solver, const double *w, int batch, int nsteps,
                             double dt, int stages, const double *c, const double *A, const double *b,
                             double *out);

#ifdef __cplusplus
}
#endif

#endif /* KALMAN_LEARN_CAPI_H */

/* kalman/framework/native/kalman.h
 *
 * Small-matrix Kalman-filter acceleration kernels.
 *
 * All matrices are row-major contiguous `double` arrays. State/measurement/control
 * dimensions (n, m, l) are expected to be small (<= ~16): every routine works on
 * fixed-size stack buffers (C99 VLAs sized by the caller's n/m/l), so there is no
 * heap allocation and no dependency on BLAS/LAPACK.
 */
#ifndef KALMAN_NATIVE_H
#define KALMAN_NATIVE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Single linear predict step.
 *   x_out = F @ x + (B @ u, if u/B given)
 *   P_out = F @ P @ F^T + (L @ Q @ L^T, if L given, else + Q)
 *
 * F: n x n   x: n   P: n x n
 * B: n x l (NULL if u is NULL or l == 0)   u: l
 * L: n x n (NULL => identity, i.e. P_out += Q directly)   Q: n x n
 */
void kf_predict(int n, int l,
                const double *F, const double *x, const double *P,
                const double *B, const double *u,
                const double *L, const double *Q,
                double *x_out, double *P_out);

/* Single Joseph-form update step. NaN entries in z mark missing channels and are
 * dropped (H/R/z are compressed to the valid channels internally). If every channel
 * is missing, (x_out, P_out) is a copy of (x, P) (predict-only) and 0 is returned.
 *
 * H: m x n   x: n   P: n x n   z: m   R: m x m
 * Returns the number of valid (non-NaN) measurement channels used.
 */
int kf_update(int n, int m,
              const double *H, const double *x, const double *P,
              const double *z, const double *R,
              double *x_out, double *P_out);

/* Joseph-form update from a precomputed residual ``y`` (= z - h(x)) rather than
 * ``z`` itself -- the variant for nonlinear engines (EKF/CKF), where the caller has
 * already evaluated the true measurement function ``h(x)`` and ``y`` is NOT
 * necessarily ``z - H @ x`` (only ``kf_update``'s linear-model assumption is). NaN
 * entries in ``y`` mark missing channels, same convention as ``kf_update``.
 *
 * H: m x n   x: n   P: n x n   y: m   R: m x m
 * Returns the number of valid (non-NaN) measurement channels used.
 */
int kf_update_residual(int n, int m,
                        const double *H, const double *x, const double *P,
                        const double *y, const double *R,
                        double *x_out, double *P_out);

/* Fused N-step linear forward pass (predict + update, every step), with NO Python
 * callbacks -- F/H/Q/R are constant matrices for the whole run.
 *
 * F: n x n   H: m x n   Q: n x n   R: m x m
 * B: n x l or NULL   L: n x n or NULL (=> identity)
 * zk: N x m (row-major; NaN marks a missing channel at that step)
 * uk: N x l or NULL
 * x0: n   P0: n x n
 *
 * Outputs (caller-allocated):
 *   x_prior, x_post: N x n
 *   P_prior, P_post: N x n x n
 *   innovations:     N x m   (z - H @ x_prior; NaN preserved where z was NaN)
 */
void kf_forward_linear(int n, int m, int l, int N,
                        const double *F, const double *H,
                        const double *Q, const double *R,
                        const double *B, const double *L,
                        const double *zk, const double *uk,
                        const double *x0, const double *P0,
                        double *x_prior, double *P_prior,
                        double *x_post, double *P_post,
                        double *innovations);

/* RTS smoother backward pass over N steps, constant transition F (matches the
 * `kf_forward_linear` output layout).
 *
 * x_post, x_prior:  N x n      (as produced by kf_forward_linear)
 * P_post, P_prior:  N x n x n
 * F: n x n constant transition
 *
 * Outputs:
 *   x_smooth: N x n
 *   P_smooth: N x n x n
 */
void kf_backward_rts(int n, int N,
                      const double *x_post, const double *P_post,
                      const double *x_prior, const double *P_prior,
                      const double *F,
                      double *x_smooth, double *P_smooth);

/* Particle filter: batch log-weight update.
 *   logw[i] += -0.5 * (z - zp[i])^T @ Rinv @ (z - zp[i])
 *
 * z: m   zp: n_particles x m   Rinv: m x m   logw: n_particles (in/out)
 */
void pf_weight_update(int n_particles, int m,
                       const double *z, const double *zp,
                       const double *Rinv, double *logw);

/* Particle filter: systematic resampling.
 *   weights: n (normalised, sums to 1)   u0: uniform offset in [0, 1)
 *   indices: n (output)
 */
void pf_systematic_resample(int n, const double *weights, double u0, int *indices);

/* UKF: weighted outer-product covariance accumulation.
 *   P_out = sum_i Wc[i] * (Y[i] - mean) outer (Y[i] - mean)
 *
 * Wc: n_sigma   Y: n_sigma x n   mean: n   P_out: n x n
 */
void ukf_weighted_outer_sum(int n_sigma, int n,
                             const double *Wc, const double *Y, const double *mean,
                             double *P_out);

/* UKF: weighted cross-covariance accumulation, for already-centered deviations.
 *   C_out = sum_i Wc[i] * A[i] outer B[i]
 *
 * Wc: n_sigma   A: n_sigma x na   B: n_sigma x nb   C_out: na x nb
 */
void ukf_weighted_cross_sum(int n_sigma, int na, int nb,
                             const double *Wc, const double *A, const double *B,
                             double *C_out);

/* Diagnostics: batch Normalized Innovation Squared, nis[k] = nu_k^T @ inv(S_k) @ nu_k.
 *   nu: N x m   S: N x m x m   nis: N (output)
 */
void diag_nis_batch(int N, int m,
                     const double *nu, const double *S, double *nis);

/* Diagnostics: batch Normalized Estimation Error Squared, nees[k] = e_k^T @ inv(P_k) @ e_k.
 *   e: N x n   P: N x n x n   nees: N (output)
 */
void diag_nees_batch(int N, int n,
                      const double *e, const double *P, double *nees);

#ifdef __cplusplus
}
#endif

#endif /* KALMAN_NATIVE_H */

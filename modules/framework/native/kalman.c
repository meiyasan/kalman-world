/* kalman/framework/native/kalman.c
 *
 * Implementation notes:
 *   - Row-major dense linear algebra, hand-written for small n/m/l (<= ~16).
 *   - C99 VLAs sized by the caller's dimensions -- no malloc, no BLAS.
 *   - Symmetric solves/inverses go through Cholesky with Tikhonov-jitter retries,
 *     mirroring the numerical fallback used on the Python side.
 */
#include "kalman.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Small dense linear algebra helpers (row-major)                     */
/* ------------------------------------------------------------------ */

/* C(r x c2) = A(r x c1) @ B(c1 x c2) */
static void mat_mul(int r, int c1, int c2, const double *A, const double *B, double *C) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c2; j++) {
            double s = 0.0;
            for (int k = 0; k < c1; k++) s += A[i * c1 + k] * B[k * c2 + j];
            C[i * c2 + j] = s;
        }
    }
}

/* C(r x c2) = A(r x c1) @ B^T, where B is (c2 x c1) */
static void mat_mul_bt(int r, int c1, int c2, const double *A, const double *B, double *C) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c2; j++) {
            double s = 0.0;
            for (int k = 0; k < c1; k++) s += A[i * c1 + k] * B[j * c1 + k];
            C[i * c2 + j] = s;
        }
    }
}

/* y(r) = A(r x c) @ x(c) */
static void mat_vec(int r, int c, const double *A, const double *x, double *y) {
    for (int i = 0; i < r; i++) {
        double s = 0.0;
        for (int k = 0; k < c; k++) s += A[i * c + k] * x[k];
        y[i] = s;
    }
}

/* Cholesky factor (lower-triangular L, S = L @ L^T) of a symmetric matrix S (n x n).
 * Retries with growing Tikhonov jitter on the diagonal if S is not (numerically) PD,
 * mirroring the `_chol` fallback in the UKF engine. */
static void cholesky(int n, const double *S, double *L) {
    double Sreg[n * n];
    memcpy(Sreg, S, sizeof(double) * (size_t)(n * n));
    double jitter = 0.0;

    for (int attempt = 0; attempt < 8; attempt++) {
        memset(L, 0, sizeof(double) * (size_t)(n * n));
        int ok = 1;
        for (int i = 0; i < n && ok; i++) {
            for (int j = 0; j <= i; j++) {
                double s = Sreg[i * n + j];
                for (int k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
                if (i == j) {
                    if (s <= 0.0) { ok = 0; break; }
                    L[i * n + j] = sqrt(s);
                } else {
                    L[i * n + j] = s / L[j * n + j];
                }
            }
        }
        if (ok) return;
        jitter = (jitter == 0.0) ? 1e-12 : jitter * 10.0;
        memcpy(Sreg, S, sizeof(double) * (size_t)(n * n));
        for (int i = 0; i < n; i++) Sreg[i * n + i] += jitter;
    }
    /* Last attempt's L (possibly degenerate) is left in place -- callers operate on
     * already near-singular covariances only in pathological cases. */
}

/* Forward substitution: solve L @ y = b for lower-triangular L (n x n). */
static void forward_subst(int n, const double *L, const double *b, double *y) {
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i * n + k] * y[k];
        y[i] = s / L[i * n + i];
    }
}

/* Backward substitution: solve L^T @ x = y (L lower-triangular, n x n). */
static void backward_subst(int n, const double *L, const double *y, double *x) {
    for (int i = n - 1; i >= 0; i--) {
        double s = y[i];
        for (int k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
        x[i] = s / L[i * n + i];
    }
}

/* Solve S @ x = b for symmetric PD S (n x n) via Cholesky. */
static void chol_solve_vec(int n, const double *S, const double *b, double *x) {
    double L[n * n];
    cholesky(n, S, L);
    double y[n];
    forward_subst(n, L, b, y);
    backward_subst(n, L, y, x);
}

/* Inverse of symmetric PD S (n x n) via Cholesky. */
static void chol_inverse(int n, const double *S, double *Sinv) {
    double L[n * n];
    cholesky(n, S, L);
    double e[n], y[n], col[n];
    for (int j = 0; j < n; j++) {
        memset(e, 0, sizeof(double) * (size_t)n);
        e[j] = 1.0;
        forward_subst(n, L, e, y);
        backward_subst(n, L, y, col);
        for (int i = 0; i < n; i++) Sinv[i * n + j] = col[i];
    }
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */

void kf_predict(int n, int l,
                const double *F, const double *x, const double *P,
                const double *B, const double *u,
                const double *L, const double *Q,
                double *x_out, double *P_out) {
    double Fx[n];
    mat_vec(n, n, F, x, Fx);
    if (B != NULL && u != NULL && l > 0) {
        double Bu[n];
        mat_vec(n, l, B, u, Bu);
        for (int i = 0; i < n; i++) x_out[i] = Fx[i] + Bu[i];
    } else {
        memcpy(x_out, Fx, sizeof(double) * (size_t)n);
    }

    double FP[n * n];
    mat_mul(n, n, n, F, P, FP);
    double FPFt[n * n];
    mat_mul_bt(n, n, n, FP, F, FPFt);

    if (L != NULL) {
        double LQ[n * n];
        mat_mul(n, n, n, L, Q, LQ);
        double LQLt[n * n];
        mat_mul_bt(n, n, n, LQ, L, LQLt);
        for (int i = 0; i < n * n; i++) P_out[i] = FPFt[i] + LQLt[i];
    } else {
        for (int i = 0; i < n * n; i++) P_out[i] = FPFt[i] + Q[i];
    }
}

/* Joseph-form update given an already NaN-masked, compressed system (Hc, Rc, yc of
 * size mv): x_out = x + K @ yc, P_out = (I-KH) P (I-KH)^T + K Rc K^T. Shared by
 * kf_update (derives yc from z - Hc@x) and kf_update_residual (yc given directly). */
static void update_core(int n, int mv, const double *Hc, const double *Rc, const double *yc,
                         const double *x, const double *P,
                         double *x_out, double *P_out) {
    /* S = Rc + Hc @ P @ Hc^T */
    double HP[mv * n];
    mat_mul(mv, n, n, Hc, P, HP);
    double HPHt[mv * mv];
    mat_mul_bt(mv, n, mv, HP, Hc, HPHt);
    double S[mv * mv];
    for (int a = 0; a < mv * mv; a++) S[a] = Rc[a] + HPHt[a];

    /* K = P @ Hc^T @ inv(S)  (n x mv) */
    double PHt[n * mv];
    for (int i = 0; i < n; i++) {
        for (int a = 0; a < mv; a++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += P[i * n + k] * Hc[a * n + k];
            PHt[i * mv + a] = s;
        }
    }
    double Sinv[mv * mv];
    chol_inverse(mv, S, Sinv);
    double K[n * mv];
    mat_mul(n, mv, mv, PHt, Sinv, K);

    /* x_out = x + K @ yc */
    double Ky[n];
    mat_vec(n, mv, K, yc, Ky);
    for (int i = 0; i < n; i++) x_out[i] = x[i] + Ky[i];

    /* Joseph form: P_out = (I - K H) P (I - K H)^T + K R K^T */
    double KH[n * n];
    mat_mul(n, mv, n, K, Hc, KH);
    double IKH[n * n];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            IKH[i * n + j] = (i == j ? 1.0 : 0.0) - KH[i * n + j];

    double IKH_P[n * n];
    mat_mul(n, n, n, IKH, P, IKH_P);
    double term1[n * n];
    mat_mul_bt(n, n, n, IKH_P, IKH, term1);

    double KR[n * mv];
    mat_mul(n, mv, mv, K, Rc, KR);
    double term2[n * n];
    mat_mul_bt(n, mv, n, KR, K, term2);

    for (int i = 0; i < n * n; i++) P_out[i] = term1[i] + term2[i];
}

int kf_update(int n, int m,
              const double *H, const double *x, const double *P,
              const double *z, const double *R,
              double *x_out, double *P_out) {
    int idx[m];
    int mv = 0;
    for (int j = 0; j < m; j++) {
        if (!isnan(z[j])) idx[mv++] = j;
    }
    if (mv == 0) {
        memcpy(x_out, x, sizeof(double) * (size_t)n);
        memcpy(P_out, P, sizeof(double) * (size_t)(n * n));
        return 0;
    }

    double Hc[mv * n], Rc[mv * mv], zc[mv];
    for (int a = 0; a < mv; a++) {
        int j = idx[a];
        for (int k = 0; k < n; k++) Hc[a * n + k] = H[j * n + k];
        zc[a] = z[j];
        for (int b = 0; b < mv; b++) Rc[a * mv + b] = R[j * m + idx[b]];
    }

    /* yc = zc - Hc @ x */
    double Hx[mv];
    mat_vec(mv, n, Hc, x, Hx);
    double yc[mv];
    for (int a = 0; a < mv; a++) yc[a] = zc[a] - Hx[a];

    update_core(n, mv, Hc, Rc, yc, x, P, x_out, P_out);
    return mv;
}

int kf_update_residual(int n, int m,
                        const double *H, const double *x, const double *P,
                        const double *y, const double *R,
                        double *x_out, double *P_out) {
    int idx[m];
    int mv = 0;
    for (int j = 0; j < m; j++) {
        if (!isnan(y[j])) idx[mv++] = j;
    }
    if (mv == 0) {
        memcpy(x_out, x, sizeof(double) * (size_t)n);
        memcpy(P_out, P, sizeof(double) * (size_t)(n * n));
        return 0;
    }

    double Hc[mv * n], Rc[mv * mv], yc[mv];
    for (int a = 0; a < mv; a++) {
        int j = idx[a];
        for (int k = 0; k < n; k++) Hc[a * n + k] = H[j * n + k];
        yc[a] = y[j];
        for (int b = 0; b < mv; b++) Rc[a * mv + b] = R[j * m + idx[b]];
    }

    update_core(n, mv, Hc, Rc, yc, x, P, x_out, P_out);
    return mv;
}

void kf_forward_linear(int n, int m, int l, int N,
                        const double *F, const double *H,
                        const double *Q, const double *R,
                        const double *B, const double *L,
                        const double *zk, const double *uk,
                        const double *x0, const double *P0,
                        double *x_prior, double *P_prior,
                        double *x_post, double *P_post,
                        double *innovations) {
    double x[n], P[n * n];
    memcpy(x, x0, sizeof(double) * (size_t)n);
    memcpy(P, P0, sizeof(double) * (size_t)(n * n));

    for (int k = 0; k < N; k++) {
        const double *u = (uk != NULL && l > 0) ? (uk + (size_t)k * l) : NULL;
        double xp[n], Pp[n * n];
        kf_predict(n, l, F, x, P, B, u, L, Q, xp, Pp);

        memcpy(x_prior + (size_t)k * n, xp, sizeof(double) * (size_t)n);
        memcpy(P_prior + (size_t)k * n * n, Pp, sizeof(double) * (size_t)(n * n));

        const double *z = zk + (size_t)k * m;
        double Hx[m];
        mat_vec(m, n, H, xp, Hx);
        for (int j = 0; j < m; j++) {
            innovations[(size_t)k * m + j] = isnan(z[j]) ? NAN : (z[j] - Hx[j]);
        }

        double xu[n], Pu[n * n];
        kf_update(n, m, H, xp, Pp, z, R, xu, Pu);

        memcpy(x_post + (size_t)k * n, xu, sizeof(double) * (size_t)n);
        memcpy(P_post + (size_t)k * n * n, Pu, sizeof(double) * (size_t)(n * n));

        memcpy(x, xu, sizeof(double) * (size_t)n);
        memcpy(P, Pu, sizeof(double) * (size_t)(n * n));
    }
}

void kf_backward_rts(int n, int N,
                      const double *x_post, const double *P_post,
                      const double *x_prior, const double *P_prior,
                      const double *F,
                      double *x_smooth, double *P_smooth) {
    memcpy(x_smooth + (size_t)(N - 1) * n, x_post + (size_t)(N - 1) * n,
           sizeof(double) * (size_t)n);
    memcpy(P_smooth + (size_t)(N - 1) * n * n, P_post + (size_t)(N - 1) * n * n,
           sizeof(double) * (size_t)(n * n));

    for (int k = N - 2; k >= 0; k--) {
        const double *Pk  = P_post   + (size_t)k * n * n;
        const double *xk  = x_post   + (size_t)k * n;
        const double *Ppn = P_prior  + (size_t)(k + 1) * n * n;
        const double *xpn = x_prior  + (size_t)(k + 1) * n;
        const double *xsn = x_smooth + (size_t)(k + 1) * n;
        const double *Psn = P_smooth + (size_t)(k + 1) * n * n;

        /* C = Pk @ F^T @ inv(Ppn) */
        double PkFt[n * n];
        mat_mul_bt(n, n, n, Pk, F, PkFt);
        double Ppn_inv[n * n];
        chol_inverse(n, Ppn, Ppn_inv);
        double C[n * n];
        mat_mul(n, n, n, PkFt, Ppn_inv, C);

        /* x_s = xk + C @ (xsn - xpn) */
        double dx[n];
        for (int i = 0; i < n; i++) dx[i] = xsn[i] - xpn[i];
        double Cdx[n];
        mat_vec(n, n, C, dx, Cdx);
        for (int i = 0; i < n; i++) x_smooth[(size_t)k * n + i] = xk[i] + Cdx[i];

        /* P_s = Pk + C @ (Psn - Ppn) @ C^T */
        double dP[n * n];
        for (int i = 0; i < n * n; i++) dP[i] = Psn[i] - Ppn[i];
        double CdP[n * n];
        mat_mul(n, n, n, C, dP, CdP);
        double CdPCt[n * n];
        mat_mul_bt(n, n, n, CdP, C, CdPCt);
        for (int i = 0; i < n * n; i++) P_smooth[(size_t)k * n * n + i] = Pk[i] + CdPCt[i];
    }
}

void pf_weight_update(int n_particles, int m,
                       const double *z, const double *zp,
                       const double *Rinv, double *logw) {
    for (int i = 0; i < n_particles; i++) {
        double dz[m];
        for (int j = 0; j < m; j++) dz[j] = z[j] - zp[(size_t)i * m + j];
        double Rinv_dz[m];
        mat_vec(m, m, Rinv, dz, Rinv_dz);
        double quad = 0.0;
        for (int j = 0; j < m; j++) quad += dz[j] * Rinv_dz[j];
        logw[i] += -0.5 * quad;
    }
}

void pf_systematic_resample(int n, const double *weights, double u0, int *indices) {
    double cumsum[n];
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        acc += weights[i];
        cumsum[i] = acc;
    }
    cumsum[n - 1] = 1.0; /* guard against floating round-off */

    int j = 0;
    for (int i = 0; i < n; i++) {
        double pos = (i + u0) / (double)n;
        while (j < n - 1 && cumsum[j] < pos) j++;
        indices[i] = j;
    }
}

void ukf_weighted_outer_sum(int n_sigma, int n,
                             const double *Wc, const double *Y, const double *mean,
                             double *P_out) {
    memset(P_out, 0, sizeof(double) * (size_t)(n * n));
    for (int s = 0; s < n_sigma; s++) {
        double d[n];
        for (int i = 0; i < n; i++) d[i] = Y[(size_t)s * n + i] - mean[i];
        double w = Wc[s];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                P_out[i * n + j] += w * d[i] * d[j];
    }
}

void ukf_weighted_cross_sum(int n_sigma, int na, int nb,
                             const double *Wc, const double *A, const double *B,
                             double *C_out) {
    memset(C_out, 0, sizeof(double) * (size_t)(na * nb));
    for (int s = 0; s < n_sigma; s++) {
        double w = Wc[s];
        for (int i = 0; i < na; i++)
            for (int j = 0; j < nb; j++)
                C_out[i * nb + j] += w * A[(size_t)s * na + i] * B[(size_t)s * nb + j];
    }
}

void diag_nis_batch(int N, int m,
                     const double *nu, const double *S, double *nis) {
    for (int k = 0; k < N; k++) {
        double x[m];
        chol_solve_vec(m, S + (size_t)k * m * m, nu + (size_t)k * m, x);
        double s = 0.0;
        for (int j = 0; j < m; j++) s += nu[(size_t)k * m + j] * x[j];
        nis[k] = s;
    }
}

void diag_nees_batch(int N, int n,
                      const double *e, const double *P, double *nees) {
    for (int k = 0; k < N; k++) {
        double Pinv[n * n];
        chol_inverse(n, P + (size_t)k * n * n, Pinv);
        double Pinv_e[n];
        mat_vec(n, n, Pinv, e + (size_t)k * n, Pinv_e);
        double s = 0.0;
        for (int i = 0; i < n; i++) s += e[(size_t)k * n + i] * Pinv_e[i];
        nees[k] = s;
    }
}

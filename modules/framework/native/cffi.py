"""cffi out-of-line API-mode builder for the kalman C acceleration kernels.

Compiles ``kalman.c`` (declared in ``kalman.h``) into a small CPython extension,
``_kalman_native``, that lives next to this file. Run directly to (re)build::

    python modules/framework/native/cffi.py

or let the build system run it via the ``cffi_modules`` hook in ``pyproject.toml``.
The compiled extension is optional: ``native/__init__.py`` falls back to pure
Python everywhere if it is missing or fails to import.
"""
from __future__ import annotations
import pathlib

from cffi import FFI

_HERE = pathlib.Path(__file__).parent

ffi = FFI()

ffi.cdef("""
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
""")

ffi.set_source(
    "modules.framework.native._kalman_native",
    '#include "kalman.h"',
    sources=[str(_HERE / "kalman.c")],
    include_dirs=[str(_HERE)],
)

if __name__ == "__main__":
    ffi.compile(verbose=True)

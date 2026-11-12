/**
 * *********************************************
 *
 * \file Cholesky.h
 * \brief Cholesky factorisation, and the solves and inverses built on it
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details The linear solver of the framework. A covariance is symmetric
 * positive-definite, so its factorisation is Cholesky's, `S = L L^T`; the
 * Kalman gain, the sigma points and the likelihoods all go through it. There
 * is no LU: nothing here factors a general matrix.
 *
 * A covariance that has drifted slightly out of positive-definiteness must not
 * abort a run, so Inverse() degrades instead of failing: Cholesky, then an
 * eigenvalue pseudo-inverse that drops the directions carrying no information.
 *
 * *********************************************
 */

#ifndef KALMAN_ALGEBRA_CHOLESKY_H
#define KALMAN_ALGEBRA_CHOLESKY_H

#include "Kalman/Matrix.h"

namespace Kalman::Algebra
{
    /// Lower-triangular factor `L` of `S = L L^T`, or an empty matrix when \p S is not positive-definite.
    Matrix Cholesky(const Matrix &S);

    /**
     * \brief Lower-triangular Cholesky factor `L` of a symmetric matrix, `S = L L^T`.
     *
     * \param S      the matrix to factor; only its symmetric part is used.
     * \param jitter when true, a growing Tikhonov ridge is added on failure. A
     *               positive *semi*-definite matrix is legitimate — the
     *               white-noise-acceleration process-noise block is exactly that —
     *               so sampling from one must degrade, not throw.
     * \return the factor, or an empty matrix when even the jittered attempts fail.
     */
    Matrix CholeskyFactor(const Matrix &S, bool jitter = true);

    /**
     * \brief Inverse of a symmetric positive-definite matrix.
     *
     * Cholesky first; on failure the diagonal is jittered by a growing Tikhonov
     * term and retried; if that still fails, a pseudo-inverse is built from the
     * symmetric eigendecomposition, dropping directions whose eigenvalue is below
     * a relative tolerance. That last step is the dependency-free equivalent of
     * numpy's `pinv`, and it is what keeps a degenerate innovation covariance from
     * turning into NaN.
     *
     * \param S  the matrix to invert; only its symmetric part is used.
     * \param ok when given, reports whether the plain (unjittered) Cholesky worked.
     */
    Matrix Inverse(const Matrix &S, bool *ok = nullptr);

    /// Solve `S x = b` for symmetric positive-definite \p S (see Inverse()).
    Vector Solve(const Matrix &S, const Vector &b);
}

#endif // KALMAN_ALGEBRA_CHOLESKY_H

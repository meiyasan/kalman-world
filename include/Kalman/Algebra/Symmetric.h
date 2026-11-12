/**
 * *********************************************
 *
 * \file Symmetric.h
 * \brief Operations on symmetric matrices (covariances)
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Covariances are symmetric by definition and only approximately so in
 * floating point. These keep them symmetric, diagonalise them, and measure how
 * close to singular they are.
 *
 * *********************************************
 */

#ifndef KALMAN_ALGEBRA_SYMMETRIC_H
#define KALMAN_ALGEBRA_SYMMETRIC_H

#include "Kalman/Matrix.h"

namespace Kalman::Algebra
{
    /// Symmetric part of \p S, `(S + S^T) / 2`, as a new matrix.
    Matrix SymmetricPart(const Matrix &S);

    /// Congruence transform `A P A^T` (the covariance workhorse).
    Matrix Similarity(const Matrix &A, const Matrix &P);

    /// Average a matrix with its transpose, killing round-off asymmetry in place.
    void Symmetrize(Matrix &P);

    /**
     * \brief Eigendecomposition of a symmetric matrix, by cyclic Jacobi rotations.
     *
     * \param S             the symmetric matrix.
     * \param[out] values   the eigenvalues.
     * \param[out] vectors  the eigenvectors, one per column.
     * \return true when the sweeps converged.
     *
     * Jacobi is not the fastest method at large n, but it is short, has no
     * dependency, and is accurate for the small symmetric matrices a Kalman filter
     * actually decomposes.
     */
    bool SymmetricEigen(const Matrix &S, Vector &values, Matrix &vectors);

    /// Condition number from the symmetric eigenvalues; infinity when singular.
    double ConditionNumber(const Matrix &M);
}

#endif // KALMAN_ALGEBRA_SYMMETRIC_H

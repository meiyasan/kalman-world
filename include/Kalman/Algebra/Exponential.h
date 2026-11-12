/**
 * *********************************************
 *
 * \file Exponential.h
 * \brief The matrix exponential, and decoding a continuous linear system
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details What turns a model stated in continuous time into one sample:
 * `F = exp(A dt)` for the dynamics and the Van Loan block for the noise.
 *
 * *********************************************
 */

#ifndef KALMAN_ALGEBRA_EXPONENTIAL_H
#define KALMAN_ALGEBRA_EXPONENTIAL_H

#include "Kalman/Matrix.h"

namespace Kalman::Algebra
{
    /// The matrix exponential `exp(A)` (scaling and squaring).
    Matrix Exponential(const Matrix &A);

    /**
     * \brief Decode a continuous linear system into one sample (Van Loan, 1978).
     *
     * For `dx/dt = A x + w`, with `w` white noise of intensity \p G (per unit
     * time; `G = L Qc L^T`), gives the transition `F = exp(A dt)` and the
     * covariance \p Q of the noise accumulated over \p dt, exactly, including
     * the cross terms a plain `G dt` misses. Works for a singular \p A.
     */
    void VanLoan(const Matrix &A, const Matrix &G, double dt, Matrix &F, Matrix &Q);
}

#endif // KALMAN_ALGEBRA_EXPONENTIAL_H

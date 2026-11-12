/**
 * *********************************************
 *
 * \file Riccati.h
 * \brief The discrete algebraic Riccati equation, and the Stein equation it is refined with
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details What a linear-quadratic regulator needs: the stabilising solution of
 *
 * \code{.unparsed}
 *     P = Q + A^T P A - A^T P B (R + B^T P B)^-1 B^T P A
 * \endcode
 *
 * Solved without a general factorisation: the Riccati recursion is iterated
 * until its gain stabilises the pair (its own convergence is linear, and slow
 * when the closed loop has a pole near the unit circle), then Hewer's Newton
 * step takes over: for the gain `K` of the current `P`, the cost of that gain is
 * the solution of the Stein equation `X = (A - B K)^T X (A - B K) + Q + K^T R K`,
 * which converges quadratically. The Stein equation is summed by doubling,
 * with matrix products only. The one inverse is of `R + B^T P B`, a covariance-
 * like SPD matrix, through Cholesky. Agrees with scipy's Schur-based
 * `solve_discrete_are` to rounding (docs/15 has the measurements).
 *
 * *********************************************
 */

#ifndef KALMAN_ALGEBRA_RICCATI_H
#define KALMAN_ALGEBRA_RICCATI_H

#include "Kalman/Matrix.h"

#include <cstddef>

namespace Kalman::Algebra
{
    /**
     * \brief The solution of the Stein (discrete Lyapunov) equation `X = A^T X A + W`.
     *
     * `X = sum_k (A^T)^k W A^k`, summed by doubling (`X += A_j^T X A_j`,
     * `A_{j+1} = A_j^2`), so `2^j` terms cost `j` steps.
     * \throws std::runtime_error when \p A is not Schur stable (the sum diverges).
     */
    Matrix SolveStein(const Matrix &A, const Matrix &W, std::size_t maxDoublings = 64);

    /**
     * \brief The stabilising solution of the discrete algebraic Riccati equation.
     *
     * \param A, B         the pair (n x n, n x l); it must be stabilisable.
     * \param Q, R         the state (n x n, PSD) and control (l x l, SPD) weights.
     * \param tolerance    relative change at which the recursion stops if Newton never takes over.
     * \param maxIterations bound on the Riccati iteration.
     * \throws std::runtime_error when the iteration does not converge.
     */
    Matrix SolveDARE(const Matrix &A, const Matrix &B, const Matrix &Q, const Matrix &R,
                     double tolerance = 1e-10, std::size_t maxIterations = 100000);

    /// The gain `K = (R + B^T P B)^-1 B^T P A` of a Riccati solution \p P.
    Matrix RiccatiGain(const Matrix &A, const Matrix &B, const Matrix &R, const Matrix &P);
}

#endif // KALMAN_ALGEBRA_RICCATI_H

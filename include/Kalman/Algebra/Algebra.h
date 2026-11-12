/**
 * *********************************************
 *
 * \file Algebra.h
 * \brief The linear algebra of the framework
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Free functions over Kalman::Matrix and Kalman::Vector, in
 * Kalman::Algebra:
 *
 * - Cholesky.h: the factorisation `S = L L^T`, Solve() and Inverse();
 * - Symmetric.h: symmetrising, the symmetric eigen-decomposition, conditioning;
 * - Exponential.h: `exp(A)` and the Van Loan decoding of a continuous system.
 *
 * These are utilities, not a role: nobody chooses between them. The roles a
 * user does choose are the Integrator (time), the Solver (space, for a field)
 * and the Differentiator (Jacobians).
 *
 * *********************************************
 */

#ifndef KALMAN_ALGEBRA_ALGEBRA_H
#define KALMAN_ALGEBRA_ALGEBRA_H

#include "Kalman/Algebra/Cholesky.h"
#include "Kalman/Algebra/Symmetric.h"
#include "Kalman/Algebra/Exponential.h"

#endif // KALMAN_ALGEBRA_ALGEBRA_H

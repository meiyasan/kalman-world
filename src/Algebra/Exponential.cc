/**
 * *********************************************
 *
 * \file Exponential.cc
 * \brief Implementation of the matrix exponential and the Van Loan decoding
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Algebra/Exponential.h"
#include "Kalman/Algebra/Symmetric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Kalman::Algebra
{
    Matrix Exponential(const Matrix &A)
    {
        const std::size_t n = A.Rows();
        if (A.Cols() != n) throw std::invalid_argument("Utils::Exponential: the matrix must be square.");

        // Scaling and squaring: halve A until its 1-norm is at most 1/2, sum the
        // series there (it converges to rounding in under 20 terms), square back.
        double norm = 0.;
        for (std::size_t j = 0; j < n; ++j)
        {
            double column = 0.;
            for (std::size_t i = 0; i < n; ++i) column += std::abs(A(i, j));
            norm = std::max(norm, column);
        }
        if (!std::isfinite(norm)) throw std::invalid_argument("Utils::Exponential: the matrix is not finite.");
        int squarings = 0;
        double scale = 1.;
        while (norm * scale > 0.5) { scale *= 0.5; ++squarings; }

        Matrix scaled = A;
        scaled *= scale;
        Matrix result = Matrix::Identity(n), term = Matrix::Identity(n);
        for (int k = 1; k <= 20; ++k)
        {
            term = term * scaled;
            term *= 1. / static_cast<double>(k);
            result += term;
        }
        for (int s = 0; s < squarings; ++s) result = result * result;
        return result;
    }

    void VanLoan(const Matrix &A, const Matrix &G, double dt, Matrix &F, Matrix &Q)
    {
        const std::size_t n = A.Rows();
        if (A.Cols() != n || G.Rows() != n || G.Cols() != n)
            throw std::invalid_argument("Utils::VanLoan: A and G must both be (n, n).");

        // exp([[-A, G], [0, A^T]] dt) = [[., M12], [0, M22]]:  F = M22^T,  Q = F M12.
        Matrix Z(2 * n, 2 * n, 0.);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
                Z(i, j) = -A(i, j) * dt;
                Z(i, n + j) = G(i, j) * dt;
                Z(n + i, n + j) = A(j, i) * dt;
            }
        const Matrix M = Exponential(Z);

        F = Matrix(n, n, 0.);
        Matrix M12(n, n, 0.);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
                F(i, j) = M(n + j, n + i);
                M12(i, j) = M(i, n + j);
            }
        Q = F * M12;
        Symmetrize(Q);
    }
}
